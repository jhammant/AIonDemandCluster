"""Headless launch/destroy — the non-interactive core used by the auto-spin-up
proxy (and reusable elsewhere). Emits phase events to `events` so progress is
visible from the CLI/TUI/proxy while it runs in the background.
"""

from __future__ import annotations

import time

from . import events, model_configs, providers, state
from .bootstrap import CONTAINER_PORT, ServerConfig
from .config import Settings
from .health import wait_until_ready
from .sizing import size_model
from .vast import recommend_disk_gb


def launch(
    s: Settings,
    *,
    model: str,
    quant: str = "bf16",
    provider: str = "vast",
    max_price: float | None = None,
    ttl_hours: float | None = None,
    idle_minutes: int | None = None,
    context: int | None = None,
    concurrency: int = 4,
    tool_parser: str | None = None,  # None -> resolve from model_configs
    extra_args: list[str] | None = None,
    on_event=None,
) -> state.Instance | None:
    """Size → find cheapest fit → rent → wait for boot + model load. Saves state
    and returns the Instance (status 'running' on success, else 'error'/None)."""

    def emit(phase: str, msg: str = "") -> None:
        events.append(phase, msg)
        if on_event:
            on_event(phase, msg)

    max_p = max_price if max_price is not None else s.max_price
    ttl_h = ttl_hours if ttl_hours is not None else s.ttl_hours

    events.clear()
    emit("sizing", model)
    try:
        sizing = size_model(
            model, hf_token=s.hf_token, quants=[quant], context_len=context, concurrency=concurrency
        )
    except Exception as e:  # noqa: BLE001 - background task; report and bail
        emit("error", f"sizing failed: {e}")
        return None

    m = sizing.model
    plan = sizing.plan(quant)
    disk = recommend_disk_gb(plan.weights_gb)

    inst: state.Instance | None = None
    try:
        with providers.get_client(provider, s) as client:
            emit("searching", f"cheapest {quant} fit ≤ ${max_p:.2f}/hr")
            priced = client.price_plan(plan, disk, max_price=max_p)
            best = min(
                (x for x in priced if x.offer), key=lambda x: x.offer.dph_total, default=None
            )
            if not best:
                emit("error", f"no offer ≤ ${max_p:.2f}/hr for {m.repo_id} ({quant})")
                return None
            offer = best.offer

            mc = model_configs.resolve(m.repo_id)
            cfg = ServerConfig(
                repo_id=m.repo_id,
                num_gpus=best.option.num_gpus,
                quant=quant,
                api_key=s.vllm_api_key,
                port=CONTAINER_PORT,
                max_model_len=context,
                tool_call_parser=tool_parser or mc.tool_call_parser,
                extra_args=(extra_args or []) + mc.vllm_serving_args(),
                hf_token=s.hf_token,
            )
            emit("renting", f"{offer.desc} @ ${offer.dph_total:.2f}/hr")
            instance_id = client.create_instance(
                offer.id, cfg, disk_gb=disk, max_price=max_p, label="aiod-vllm"
            )
            inst = state.Instance(
                instance_id=instance_id,
                repo_id=m.repo_id,
                quant=quant,
                gpu_desc=offer.desc,
                price_per_hr=offer.dph_total,
                created_at=time.time(),
                ttl_hours=ttl_h,
                api_key=s.vllm_api_key,
                status="creating",
                provider=provider,
                idle_minutes=idle_minutes,
                weights_gb=plan.weights_gb,
            )
            state.save(inst)

            ep = None
            start = time.time()
            while time.time() - start < 1200:  # slow nodes pull the image slowly
                vi = client.get_instance(instance_id)
                ep = client.endpoint_of(vi, CONTAINER_PORT)
                emit("booting", f"{client.status_of(vi)} ({int(time.time() - start)}s)")
                if ep:
                    break
                time.sleep(10)
            if not ep:
                emit("error", "port never mapped (machine may lack free direct ports)")
                inst.status = "error"
                state.save(inst)
                return inst
            inst.host, inst.port, inst.status = ep[0], ep[1], "loading"
            state.save(inst)
    except providers.PROVIDER_ERRORS as e:
        emit("error", str(e))
        return inst

    emit("loading", "downloading weights / loading model")
    ok = wait_until_ready(
        inst.base_url,
        api_key=inst.api_key,
        on_progress=lambda hs, el: emit("loading", f"{hs.detail} ({int(el)}s)"),
    )
    inst.status = "running" if ok else "error"
    state.save(inst)
    emit("ready" if ok else "error", inst.base_url if ok else "model did not come up in time")
    return inst


def destroy(s: Settings, inst: state.Instance) -> bool:
    try:
        with providers.get_client(inst.provider, s) as client:
            client.destroy_instance(inst.instance_id)
    except providers.PROVIDER_ERRORS as e:
        events.append("error", f"destroy failed: {e}")
        return False
    state.clear()
    events.append("destroyed", f"instance {inst.instance_id}")
    return True
