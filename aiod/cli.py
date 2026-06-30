"""`aiod` command-line interface.

    aiod estimate <hf-link>     # size the model + show live vast.ai $/hr
    aiod spin <hf-link>         # rent a GPU, serve with vLLM, wire up CCR
    aiod status                 # show the running instance + cost so far
    aiod teardown               # destroy the instance (stop billing)
    aiod ccr-config             # re-write the CCR config from saved state
    aiod tui                    # interactive Textual wizard
"""

from __future__ import annotations

import time
import webbrowser

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import branding, ccr, events, model_configs, onboard, profiles, providers, state
from .bootstrap import CONTAINER_PORT, ServerConfig
from .config import Settings
from .health import wait_until_ready
from .sizing import QUANT_LABELS, size_any
from .vast import PricedOption, recommend_disk_gb

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Spin up a HuggingFace model on vast.ai and drive it from Claude Code via CCR.",
)
console = Console()

profile_app = typer.Typer(no_args_is_help=True, help="Manage spin-up profiles (named presets).")
app.add_typer(profile_app, name="profile")

ONLINE_QUANTS = {"bf16", "fp16", "fp8"}  # work on any repo; int4 needs a pre-quantized checkpoint


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _warn_quant(repo_id: str, quant: str) -> None:
    if quant in ("awq-int4", "gptq-int4"):
        kind = quant.split("-")[0]
        if kind not in repo_id.lower():
            console.print(
                f"[yellow]![/] '{quant}' needs a pre-quantized {kind.upper()} checkpoint. "
                f"'{repo_id}' doesn't look quantized — vLLM will likely fail to load it. "
                f"Use a '*-{kind.upper()}' repo, or pick 'fp8' (works online) / 'bf16'."
            )


def _pick_cheapest(priced: list[PricedOption]) -> PricedOption | None:
    with_offers = [p for p in priced if p.offer is not None]
    if not with_offers:
        return None
    return min(with_offers, key=lambda p: p.offer.dph_total)


def _settings_or_exit() -> Settings:
    s = Settings.load()
    if not s.vast_api_key:
        console.print("[red]VAST_API_KEY is not set.[/] Add it to .env (see .env.example).")
        raise typer.Exit(1)
    return s


def _require_provider_key(s: Settings, provider: str) -> None:
    if not providers.api_key_for(provider, s):
        env = "VAST_API_KEY" if provider == "vast" else f"{provider.upper()}_API_KEY"
        console.print(f"[red]{env} is not set.[/] Run `aiod init` or add it to .env.")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# init  (guided key setup)
# --------------------------------------------------------------------------- #

def _maybe_open(url: str) -> None:
    if typer.confirm(f"Open {url} in your browser?", default=True):
        try:
            webbrowser.open(url)
        except Exception:
            console.print(f"[dim]Couldn't open a browser — visit: {url}[/]")


@app.command()
def init():
    """Guided setup: walk through getting your vast.ai (and optional HF) keys."""
    console.print(
        Panel(
            "Let's get you set up. You'll need a [bold]vast.ai[/] account + API key.\n"
            "A HuggingFace token is optional (only for gated models like Llama).",
            title="aiod init",
            border_style="cyan",
        )
    )

    existing = onboard.read_env()
    updates: dict[str, str] = {}

    # --- vast.ai key -------------------------------------------------------
    console.print("\n[bold]1) vast.ai API key[/] [red](required)[/]")
    if not typer.confirm("Do you already have a vast.ai account?", default=True):
        url = branding.signup_url()
        console.print(f"   Sign up here: [link={url}]{url}[/]")
        if branding.VAST_REFERRAL_URL.strip():
            console.print("   [dim](that's the maintainer's referral link — thanks for using it!)[/]")
        _maybe_open(url)
        typer.confirm("   Press Enter once your account is created", default=True, show_default=False)

    cur = existing.get("VAST_API_KEY", "")
    if cur and typer.confirm("   A VAST_API_KEY is already in .env. Keep it?", default=True):
        key = cur
    else:
        console.print(f"   Get your key at: [link={branding.VAST_KEYS_URL}]{branding.VAST_KEYS_URL}[/]")
        _maybe_open(branding.VAST_KEYS_URL)
        key = typer.prompt("   Paste your VAST_API_KEY").strip()

    while True:
        with console.status("   Validating with vast.ai..."):
            ok, msg = onboard.validate_vast_key(key)
        if ok:
            console.print(f"   [green]✓ vast.ai key {msg}[/]")
            updates["VAST_API_KEY"] = key
            break
        console.print(f"   [red]✗ {msg}[/]")
        if not typer.confirm("   Try a different key?", default=True):
            raise typer.Exit(1)
        key = typer.prompt("   Paste your VAST_API_KEY").strip()

    # --- HuggingFace token (optional) -------------------------------------
    console.print("\n[bold]2) HuggingFace token[/] [dim](optional — gated models only)[/]")
    if typer.confirm("   Set a HuggingFace token now?", default=False):
        console.print(f"   Create one at: [link={branding.HF_TOKENS_URL}]{branding.HF_TOKENS_URL}[/]")
        _maybe_open(branding.HF_TOKENS_URL)
        token = typer.prompt("   Paste your HF_TOKEN", default="", show_default=False).strip()
        if token:
            with console.status("   Validating with HuggingFace..."):
                ok, msg = onboard.validate_hf_token(token)
            console.print(f"   [{'green' if ok else 'yellow'}]{'✓' if ok else '!'} {msg}[/]")
            if ok:
                updates["HF_TOKEN"] = token

    # --- defaults ----------------------------------------------------------
    console.print("\n[bold]3) Safety defaults[/]")
    ttl = typer.prompt("   Auto-destroy reminder (hours)", default=existing.get("AIOD_TTL_HOURS", "4"))
    cap = typer.prompt("   Hard price cap ($/hr)", default=existing.get("AIOD_MAX_PRICE", "6.0"))
    updates["AIOD_TTL_HOURS"] = str(ttl)
    updates["AIOD_MAX_PRICE"] = str(cap)

    onboard.set_env_values(updates)
    console.print(f"\n[green]✓ Saved to {onboard.ENV_FILE}[/] [dim](gitignored)[/]")

    ccr_path = onboard.ccr_installed()
    if not ccr_path:
        console.print(
            f"\n[yellow]![/] Claude Code Router not found. Install it with:\n   "
            f"[bold]{branding.CCR_INSTALL_CMD}[/]"
        )

    console.print(
        Panel(
            "You're ready! Estimate cost for a model (no spend):\n"
            "  [bold]aiod estimate Qwen/Qwen2.5-Coder-32B-Instruct[/]\n"
            "…then launch it:\n"
            "  [bold]aiod spin Qwen/Qwen2.5-Coder-32B-Instruct -q fp8[/]",
            title="Next",
            border_style="green",
        )
    )


# --------------------------------------------------------------------------- #
# doctor  (health check)
# --------------------------------------------------------------------------- #

@app.command()
def doctor():
    """Check that keys, the router, and the environment are all good to go."""
    # Validate the EFFECTIVE config (env vars + project .env + global ~/.config/aiod/.env),
    # so this works the same whether run inside the project or from anywhere.
    s = Settings.load()
    table = Table(title="aiod doctor", show_header=True)
    table.add_column("Check")
    table.add_column("Status")

    def row(name: str, ok: bool, detail: str) -> None:
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        table.add_row(name, f"{mark} {detail}")

    if s.vast_api_key:
        ok, msg = onboard.validate_vast_key(s.vast_api_key)
        row("vast.ai key", ok, msg)
    else:
        row("vast.ai key", False, "not set — run `aiod init`")

    ok, msg = onboard.validate_runpod_key(s.runpod_api_key)
    row("RunPod key", ok, msg)

    ok, msg = onboard.validate_hf_token(s.hf_token or "")
    row("HuggingFace token", ok, msg)

    ccr_path = onboard.ccr_installed()
    row(
        "Claude Code Router",
        bool(ccr_path),
        ccr_path or f"missing — `{branding.CCR_INSTALL_CMD}`",
    )

    inst = state.load()
    if inst:
        table.add_row("Tracked instance", f"#{inst.instance_id} {inst.repo_id} ({inst.status})")

    console.print(table)


# --------------------------------------------------------------------------- #
# estimate
# --------------------------------------------------------------------------- #

@app.command()
def estimate(
    model: str = typer.Argument(..., help="HuggingFace link or org/name"),
    quant: list[str] = typer.Option(
        ["bf16", "fp8", "awq-int4"], "--quant", "-q", help="Quant schemes to compare"
    ),
    provider: str = typer.Option("vast", "--provider", help="vast | runpod"),
    engine: str = typer.Option("auto", "--engine", help="auto | vllm | llamacpp (GGUF)"),
    context: int = typer.Option(None, "--context", help="Context length for the KV estimate"),
    concurrency: int = typer.Option(4, "--concurrency", help="Concurrent sequences for KV estimate"),
    max_price: float = typer.Option(None, "--max-price", help="Cap $/hr in the offer search"),
    gpu: list[str] = typer.Option(
        None, "--gpu", help="Only consider GPUs whose name contains this (repeatable), e.g. --gpu rtx6000"
    ),
):
    """Size a model from its HuggingFace link and show live provider cost ($0)."""
    provider = provider.lower()
    s = Settings.load()
    _require_provider_key(s, provider)
    with console.status("Fetching model metadata from HuggingFace..."):
        sizing = size_any(
            model, engine=engine, hf_token=s.hf_token, quants=quant,
            context_len=context, concurrency=concurrency,
        )
    m = sizing.model

    if sizing.engine == "llamacpp":
        body = (
            f"[bold]{m.repo_id}[/]\n"
            f"GGUF · {len(sizing.plans)} quant build(s) · engine: llama.cpp"
        )
    else:
        src = "" if m.params_source == "safetensors" else " [dim](size guessed from name)[/]"
        body = (
            f"[bold]{m.repo_id}[/]\n"
            f"{m.params_b:.1f}B params · {m.dtype} · "
            f"ctx {m.max_context or '?'} · {'gated' if m.gated else 'open'}{src}\n"
            f"[dim]KV estimate uses {sizing.context_tokens:,} tokens "
            f"({concurrency} concurrent seqs)[/]"
        )
    console.print(Panel(body, title="Model", expand=False))

    table = Table(title=f"VRAM & live {provider} cost", show_lines=False)
    table.add_column("Quant")
    table.add_column("VRAM need", justify="right")
    table.add_column("Cheapest fit")
    table.add_column("$/hr", justify="right")
    table.add_column("$/4h", justify="right")

    max_p = max_price if max_price is not None else s.max_price
    try:
        with providers.get_client(provider, s) as client:
            with console.status(f"Querying live {provider} offers..."):
                for p in sizing.plans:
                    disk = recommend_disk_gb(p.weights_gb)
                    priced = client.price_plan(p, disk, max_price=max_p, gpu_match=gpu or None)
                    best = _pick_cheapest(priced)
                    if best:
                        fit = f"{best.option.num_gpus}x {best.option.tier.name}"
                        hr = f"${best.offer.dph_total:.2f}"
                        four = f"${best.offer.dph_total * 4:.2f}"
                    else:
                        fit, hr, four = "[red]no offer found[/]", "—", "—"
                    table.add_row(f"{p.quant}", f"{p.required_vram_gb:.0f} GB", fit, hr, four)
    except providers.PROVIDER_ERRORS as e:
        console.print(f"[red]{provider} error:[/] {e}")
        raise typer.Exit(1) from e
    console.print(table)
    console.print(
        "[dim]Tip: int4 (awq/gptq) needs a pre-quantized repo; fp8 works online on any model.[/]"
    )


# --------------------------------------------------------------------------- #
# spin
# --------------------------------------------------------------------------- #

@app.command()
def spin(
    model: str = typer.Argument(None, help="HuggingFace link or org/name (or use --profile)"),
    profile: str = typer.Option(None, "--profile", "-p", help="Use a saved profile's settings"),
    provider: str = typer.Option(None, "--provider", help="Backend: vast | runpod"),
    engine: str = typer.Option("auto", "--engine", help="auto | vllm | llamacpp (GGUF)"),
    quant: str = typer.Option(
        None, "--quant", "-q", help="vLLM: bf16/fp8/awq-int4 · GGUF: a repo quant tag"
    ),
    max_price: float = typer.Option(None, "--max-price", help="Hard cap $/hr (default from .env)"),
    gpu: list[str] = typer.Option(
        None, "--gpu", help="Only rent GPUs whose name contains this (repeatable), e.g. --gpu rtx6000"
    ),
    ttl: float = typer.Option(None, "--ttl", help="Auto-destroy reminder window, hours"),
    idle: int = typer.Option(
        None, "--idle", help="Auto-shutdown after N idle minutes (starts a local watcher)"
    ),
    context: int = typer.Option(None, "--context", help="Max model length to serve"),
    concurrency: int = typer.Option(None, "--concurrency", help="Concurrency for KV sizing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    no_ccr: bool = typer.Option(False, "--no-ccr", help="Don't touch the CCR config"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the chosen offer + exact payload without renting ($0)"
    ),
):
    """Rent a GPU, serve the model with vLLM, and wire up Claude Code Router."""
    s = Settings.load()

    prof = None
    if profile:
        prof = profiles.get(profile)
        if not prof:
            console.print(f"[red]No profile '{profile}'.[/] See [bold]aiod profile list[/].")
            raise typer.Exit(1)

    # Resolution: CLI flag  >  profile value  >  built-in/.env default.
    model = model or (prof.model if prof else None)
    if not model:
        console.print("[red]Provide a model or --profile.[/] See [bold]aiod profile list[/].")
        raise typer.Exit(1)
    quant = quant or (prof.quant if prof else None)
    provider = (provider or (prof.provider if prof else "vast")).lower()
    _require_provider_key(s, provider)
    context = context if context is not None else (prof.context if prof else None)
    concurrency = concurrency if concurrency is not None else (prof.concurrency if prof else 4)
    max_p = (
        max_price if max_price is not None
        else (prof.max_price if prof and prof.max_price is not None else s.max_price)
    )
    ttl_h = (
        ttl if ttl is not None
        else (prof.ttl_hours if prof and prof.ttl_hours is not None else s.ttl_hours)
    )
    idle_m = idle if idle is not None else (prof.idle_minutes if prof else None)
    mc = model_configs.resolve(model)
    tool_parser = (prof.tool_call_parser if prof and prof.tool_call_parser else None) or mc.tool_call_parser
    extra_args = (list(prof.extra_vllm_args) if prof else []) + mc.vllm_serving_args()

    if state.load() is not None:
        console.print(
            "[yellow]An instance is already tracked.[/] Run `aiod status` or `aiod teardown` first."
        )
        raise typer.Exit(1)

    try:
        client_cm = providers.get_client(provider, s)
    except providers.ProviderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e

    with console.status("Sizing model..."):
        sizing = size_any(
            model, engine=engine, hf_token=s.hf_token,
            quants=[quant] if quant else None, context_len=context, concurrency=concurrency,
        )
    eng = sizing.engine
    m = sizing.model

    if quant is None:
        if eng == "llamacpp":
            avail = ", ".join(p.quant for p in sizing.plans)
            console.print(f"[yellow]GGUF model[/] — pick a quant with -q. Available: {avail}")
            raise typer.Exit(1)
        quant = "bf16"
    plan = sizing.plan(quant)
    if plan is None:
        avail = ", ".join(p.quant for p in sizing.plans)
        console.print(f"[red]Quant '{quant}' not available.[/] Options: {avail}")
        raise typer.Exit(1)
    if eng == "vllm":
        _warn_quant(model, quant)
    disk = recommend_disk_gb(plan.weights_gb)

    with client_cm as client:
        with console.status("Finding the cheapest GPU that fits..."):
            priced = client.price_plan(plan, disk, max_price=max_p, gpu_match=gpu or None)
        best = _pick_cheapest(priced)
        if not best:
            gpu_note = f" matching --gpu {' '.join(gpu)}" if gpu else ""
            console.print(
                f"[red]No {provider} offer found[/] under ${max_p:.2f}/hr for {m.repo_id} "
                f"({quant}){gpu_note}. Try a higher --max-price or a smaller quant."
            )
            raise typer.Exit(1)

        offer = best.offer
        console.print(
            Panel(
                f"[bold]{m.repo_id}[/]  ·  {quant} ({QUANT_LABELS.get(quant, quant)})\n"
                f"GPU:   {offer.desc}  ·  reliability {offer.reliability:.0%}  ·  "
                f"{offer.geolocation or '?'}\n"
                f"Disk:  {disk} GB\n"
                f"Price: [bold]${offer.dph_total:.2f}/hr[/]  (~${offer.dph_total * ttl_h:.2f} "
                f"over a {ttl_h:g}h session)\n"
                f"VRAM:  need ~{plan.required_vram_gb:.0f} GB / have {offer.total_vram_gb:.0f} GB",
                title="Launch plan",
                border_style="cyan",
                expand=False,
            )
        )
        cfg = ServerConfig(
            repo_id=m.repo_id,
            num_gpus=best.option.num_gpus,
            quant=quant,
            api_key=s.vllm_api_key,
            engine=eng,
            port=CONTAINER_PORT,
            max_model_len=context,
            tool_call_parser=tool_parser,
            extra_args=extra_args,
            hf_token=s.hf_token,
            gguf_quant=quant if eng == "llamacpp" else None,
        )

        if dry_run:
            _print_dry_run(client, offer, cfg, disk, max_p)
            raise typer.Exit(0)

        if not yes and not typer.confirm("Rent this machine and start serving?"):
            raise typer.Exit(0)

        with console.status("Renting instance on vast.ai..."):
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
            idle_minutes=idle_m,
        )
        state.save(inst)
        console.print(f"[green]✓[/] Instance [bold]{instance_id}[/] created. Waiting for boot...")

        endpoint = _wait_for_endpoint(client, instance_id)
        if endpoint is None:
            console.print(
                "[red]Port never mapped.[/] The machine may lack free direct ports. "
                "Run `aiod teardown` and try again."
            )
            raise typer.Exit(1)
        host, port = endpoint
        inst.host, inst.port, inst.status = host, port, "loading"
        state.save(inst)
        console.print(f"[green]✓[/] Endpoint reachable at [bold]{inst.base_url}[/] — loading weights...")

    # Health: wait for the model to download + load (GGUF downloads can be huge).
    ok = _wait_for_health(inst, timeout_s=5400.0 if eng == "llamacpp" else 2400.0)
    if not ok:
        console.print(
            "[yellow]Model still not serving after the timeout.[/] "
            "Big models can take a while — check `aiod status` again shortly."
        )
        raise typer.Exit(1)

    inst.status = "running"
    state.save(inst)
    console.print(f"[green]✓ Model is live:[/] {inst.base_url}")

    if not no_ccr:
        path = ccr.write_config(inst.base_url, inst.api_key, inst.repo_id)
        console.print(f"[green]✓[/] CCR config written: {path}")

    if idle_m:
        if _launch_watcher(idle_m):
            console.print(
                f"[green]✓[/] Idle watcher started — auto-shutdown after {idle_m} idle min."
            )
        else:
            console.print(
                f"[yellow]![/] Couldn't auto-start the watcher. Run it yourself:\n"
                f"   [bold]aiod watch --idle {idle_m}[/]"
            )

    _print_next_steps(inst, wrote_ccr=not no_ccr)


def _launch_watcher(idle_minutes: int) -> bool:
    """Spawn `aiod watch` detached so idle-shutdown keeps running in the background."""
    from .watch import spawn_detached

    return spawn_detached(idle_minutes, state.STATE_DIR / "watch.log")


def _wait_for_endpoint(client, instance_id, timeout_s: float = 1200.0, startup_grace: float = 540.0):
    """Wait for the public port to map. Aborts early if the container never even
    reaches 'running' within startup_grace — that's a bad host (stuck pulling the
    image), and waiting the full timeout just wastes money."""
    start = time.time()
    seen_running = False
    with console.status("Booting container / pulling image...") as status:
        while time.time() - start < timeout_s:
            inst = client.get_instance(instance_id)
            st = client.status_of(inst)
            if "running" in st.lower():
                seen_running = True
            ep = client.endpoint_of(inst, CONTAINER_PORT)
            status.update(f"vast status: {st} ({int(time.time() - start)}s)")
            if ep:
                return ep
            if not seen_running and (time.time() - start) > startup_grace:
                console.print(
                    "[yellow]Host never started the container (slow/bad node) — aborting early.[/]"
                )
                return None
            time.sleep(8)
    return None


def _wait_for_health(inst: state.Instance, timeout_s: float = 2400.0) -> bool:
    from .vast import extract_download_progress

    # A client just for log-tailing (live download %). Best-effort — only vast has it.
    log_client = None
    try:
        log_client = providers.get_client(inst.provider, Settings.load())
    except Exception:  # noqa: BLE001
        log_client = None
    box = {"dl": None, "last": 0.0}

    def on_progress(hs, elapsed, status):
        msg = f"{hs.detail} ({int(elapsed)}s)"
        if log_client is not None and hasattr(log_client, "fetch_logs") and elapsed - box["last"] > 20:
            box["last"] = elapsed
            try:
                p = extract_download_progress(log_client.fetch_logs(inst.instance_id))
                if p:
                    box["dl"] = p
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass
        if box["dl"]:
            msg += f"  ·  ⬇ {box['dl']}"
        status.update(msg)

    try:
        with console.status("Downloading weights / loading model...") as status:
            return wait_until_ready(
                inst.base_url, api_key=inst.api_key, timeout_s=timeout_s,
                on_progress=lambda hs, el: on_progress(hs, el, status),
            )
    finally:
        if log_client is not None:
            try:
                log_client.close()
            except Exception:  # noqa: BLE001
                pass


def _print_dry_run(client, offer, cfg, disk, max_price) -> None:
    import json

    body = client.build_create_body(cfg, disk_gb=disk, max_price=max_price)
    redacted = json.loads(json.dumps(body))  # deep copy
    for k in list(redacted.get("env", {})):
        if "TOKEN" in k:
            redacted["env"][k] = "***redacted***"
    redacted["args"] = _redact_args(redacted.get("args", []))

    console.print(
        Panel(
            f"Would rent offer [bold]{offer.id}[/] — {offer.desc} @ ${offer.dph_total:.2f}/hr\n"
            f"Image: {cfg.image}  ·  disk {disk} GB  ·  price cap ${max_price:.2f}",
            title="DRY RUN — nothing rented",
            border_style="yellow",
        )
    )
    console.print("[bold]vast.ai create payload:[/]")
    console.print_json(data=redacted)
    console.print(
        "\n[dim]Looks right? Re-run without --dry-run to actually launch.[/]"
    )


def _redact_args(args: list[str]) -> list[str]:
    out = list(args)
    for i, a in enumerate(out):
        if a == "--api-key" and i + 1 < len(out):
            out[i + 1] = "***redacted***"
    return out


def _print_next_steps(inst: state.Instance, wrote_ccr: bool) -> None:
    lines = [
        "[bold green]Ready.[/] Point Claude Code at your model:\n",
        "  [bold]ccr restart && ccr code[/]\n" if wrote_ccr else "",
        f"Endpoint: {inst.base_url}",
        f"Model:    {inst.repo_id}",
        f"Cost:     ${inst.price_per_hr:.2f}/hr  ·  auto-destroy reminder in {inst.ttl_hours:g}h",
        "\n[dim]When done:[/] [bold]aiod teardown[/]  (stops billing)",
    ]
    console.print(Panel("\n".join(x for x in lines if x), title="Next steps", border_style="green"))


# --------------------------------------------------------------------------- #
# status / teardown / ccr-config
# --------------------------------------------------------------------------- #

def _print_events(evs: list[dict]) -> None:
    t = Table(title="Recent activity", show_header=False)
    for e in evs:
        t.add_row(f"[cyan]{e.get('phase', '?')}[/]", str(e.get("msg", "")))
    console.print(t)


@app.command()
def status():
    """Show the tracked instance, its health, cost so far, TTL, and warm-up trail."""
    inst = state.load()
    evs = events.read(10)
    if inst is None:
        if evs:
            _print_events(evs)
            console.print("[dim]No instance tracked yet (warm-up above, or last attempt ended).[/]")
        else:
            console.print("No instance tracked. Use `aiod spin <model>` or `aiod proxy`.")
        raise typer.Exit(0)

    s = Settings.load()
    live_status = "?"
    if providers.api_key_for(inst.provider, s):
        try:
            with providers.get_client(inst.provider, s) as client:
                vi = client.get_instance(inst.instance_id)
                live_status = client.status_of(vi)
                ep = client.endpoint_of(vi, CONTAINER_PORT)
                if ep and not inst.base_url:
                    inst.host, inst.port = ep
                    state.save(inst)
        except providers.PROVIDER_ERRORS as e:
            live_status = f"error: {e}"

    over = inst.expires_in_hours < 0
    table = Table(show_header=False)
    table.add_row("Instance", str(inst.instance_id))
    table.add_row("Model", inst.repo_id)
    table.add_row("GPU", f"{inst.gpu_desc} ({inst.quant})")
    table.add_row("Endpoint", inst.base_url or "[dim]not mapped yet[/]")
    table.add_row(f"{inst.provider} status", live_status)
    table.add_row("Price", f"${inst.price_per_hr:.2f}/hr")
    table.add_row("Running for", f"{inst.age_hours:.2f} h  (~${inst.est_cost_so_far:.2f} so far)")
    table.add_row(
        "TTL",
        f"[red]exceeded by {-inst.expires_in_hours:.2f} h — consider teardown[/]"
        if over
        else f"{inst.expires_in_hours:.2f} h left",
    )
    console.print(table)
    if evs:
        _print_events(evs)


@app.command()
def teardown(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Destroy the tracked instance and stop billing."""
    inst = state.load()
    if inst is None:
        console.print("Nothing to tear down.")
        raise typer.Exit(0)
    if not yes and not typer.confirm(
        f"Destroy instance {inst.instance_id} ({inst.gpu_desc}, ${inst.price_per_hr:.2f}/hr)?"
    ):
        raise typer.Exit(0)

    s = Settings.load()
    _require_provider_key(s, inst.provider)
    try:
        with providers.get_client(inst.provider, s) as client:
            client.destroy_instance(inst.instance_id)
        console.print(f"[green]✓[/] Instance {inst.instance_id} destroyed.")
    except providers.PROVIDER_ERRORS as e:
        console.print(f"[red]Failed to destroy:[/] {e}")
        console.print("Check your provider console manually to avoid charges.")
        raise typer.Exit(1) from e
    finally:
        state.clear()


@app.command(name="ccr-config")
def ccr_config():
    """Re-write the Claude Code Router config from the tracked instance."""
    inst = state.load()
    if inst is None or not inst.base_url:
        console.print("No running instance with a mapped endpoint. Run `aiod spin` first.")
        raise typer.Exit(1)
    path = ccr.write_config(inst.base_url, inst.api_key, inst.repo_id)
    console.print(f"[green]✓[/] Wrote {path}\nRun: [bold]ccr restart && ccr code[/]")


@app.command()
def ping(
    tools: bool = typer.Option(False, "--tools", help="Also test a tool/function call"),
):
    """Send a test prompt to the running model to confirm it actually serves."""
    from .health import sample_completion

    inst = state.load()
    if inst is None or not inst.base_url:
        console.print("No running instance with a mapped endpoint. Run `aiod spin` first.")
        raise typer.Exit(1)

    console.print(f"Pinging [bold]{inst.base_url}[/] ({inst.repo_id})...")
    res = sample_completion(inst.base_url, inst.repo_id, api_key=inst.api_key)
    if not res["ok"]:
        console.print(f"[red]✗ Chat failed:[/] {res['error']}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/] Replied in {res['latency_s']:.1f}s: [italic]{res['text']!r}[/]")

    if tools:
        tr = sample_completion(inst.base_url, inst.repo_id, api_key=inst.api_key, with_tool=True)
        if tr["ok"] and tr["tool_call"]:
            console.print(f"[green]✓[/] Tool call works: [italic]{tr['tool_call']}[/]")
        elif tr["ok"]:
            console.print(
                "[yellow]![/] No tool call returned — Claude Code may misbehave. "
                "Try a different --tool-call-parser for this model family."
            )
        else:
            console.print(f"[red]✗ Tool test failed:[/] {tr['error']}")
    console.print("\n[dim]Looks good? Run [bold]ccr restart && ccr code[/] to use it in Claude Code.[/]")


@app.command()
def watch(
    idle: int = typer.Option(20, "--idle", help="Idle minutes before auto-shutdown"),
    poll: float = typer.Option(30.0, "--poll", help="Seconds between metric polls"),
):
    """Watch the running instance and destroy it once idle (TTL is the backstop)."""
    from .watch import watch_loop

    inst = state.load()
    if inst is None or not inst.base_url:
        console.print("No running instance with an endpoint to watch.")
        raise typer.Exit(1)

    s = _settings_or_exit()
    try:
        client_cm = providers.get_client(inst.provider, s)
    except providers.ProviderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e

    console.print(
        f"Watching {inst.base_url} — idle {idle}m, TTL {inst.ttl_hours:g}h. Ctrl-C to stop."
    )
    reason = "gone"
    with client_cm as client:
        try:
            reason = watch_loop(
                inst.base_url,
                inst.api_key,
                idle_minutes=idle,
                created_at=inst.created_at,
                ttl_hours=inst.ttl_hours,
                destroy=lambda: client.destroy_instance(inst.instance_id),
                poll_seconds=poll,
                on_event=lambda msg: console.log(msg),
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch stopped[/] — instance left running.")
            raise typer.Exit(0) from None

    if reason in ("idle", "ttl"):
        state.clear()
        console.print(f"[green]✓[/] Instance destroyed ({reason}).")
    else:
        console.print("[yellow]Watcher exited without destroying (endpoint gone?).[/]")


# --------------------------------------------------------------------------- #
# profile subcommands
# --------------------------------------------------------------------------- #

@profile_app.command("list")
def profile_list():
    """List built-in and user profiles."""
    table = Table(title="Profiles")
    for col in ("Name", "Source", "Model", "Provider", "Quant", "Idle", "Notes"):
        table.add_column(col)
    for name, p in sorted(profiles.all_profiles().items()):
        table.add_row(
            name,
            "built-in" if profiles.is_builtin(name) else "user",
            p.model,
            p.provider,
            p.quant,
            f"{p.idle_minutes}m" if p.idle_minutes else "—",
            p.description,
        )
    console.print(table)
    console.print(f"[dim]User profiles file: {profiles.PROFILE_FILE}[/]")
    console.print("[dim]Use one: [bold]aiod spin --profile <name>[/]  ·  or in the TUI.[/]")


@profile_app.command("show")
def profile_show(name: str):
    """Print a profile as JSON."""
    p = profiles.get(name)
    if not p:
        console.print(f"[red]No profile '{name}'.[/]")
        raise typer.Exit(1)
    console.print_json(data={"name": p.name, **p.body()})


@profile_app.command("add")
def profile_add(
    name: str = typer.Argument(..., help="Profile name"),
    model: str = typer.Option(..., "--model", help="HuggingFace link or org/name"),
    provider: str = typer.Option("vast", "--provider"),
    quant: str = typer.Option("bf16", "--quant", "-q"),
    max_price: float = typer.Option(None, "--max-price"),
    context: int = typer.Option(None, "--context"),
    ttl: float = typer.Option(None, "--ttl"),
    idle: int = typer.Option(None, "--idle"),
    tool_call_parser: str = typer.Option("hermes", "--tool-call-parser"),
    description: str = typer.Option("", "--desc"),
):
    """Create or overwrite a user profile."""
    profiles.save(
        profiles.Profile(
            name=name,
            model=model,
            provider=provider,
            quant=quant,
            max_price=max_price,
            context=context,
            ttl_hours=ttl,
            idle_minutes=idle,
            tool_call_parser=tool_call_parser,
            description=description,
        )
    )
    console.print(f"[green]✓[/] Saved profile '{name}' → {profiles.PROFILE_FILE}")


@profile_app.command("rm")
def profile_rm(name: str):
    """Remove a user profile."""
    if profiles.is_builtin(name):
        console.print(f"[yellow]'{name}' is built-in — can't remove it.[/]")
        raise typer.Exit(1)
    if profiles.remove(name):
        console.print(f"[green]✓[/] Removed '{name}'.")
    else:
        console.print(f"[red]No user profile '{name}'.[/]")
        raise typer.Exit(1)


@profile_app.command("path")
def profile_path():
    """Print the user profiles file path."""
    console.print(str(profiles.PROFILE_FILE))


@app.command()
def proxy(
    profile: str = typer.Option(None, "--profile", "-p", help="Profile to spin on first request"),
    model: str = typer.Option(None, "--model", help="Model to spin (if no profile)"),
    quant: str = typer.Option(None, "--quant", "-q"),
    provider: str = typer.Option(None, "--provider"),
    max_price: float = typer.Option(None, "--max-price"),
    ttl: float = typer.Option(None, "--ttl"),
    idle: int = typer.Option(20, "--idle", help="Auto-destroy after N idle minutes"),
    context: int = typer.Option(None, "--context"),
    port: int = typer.Option(4000, "--port"),
    write_ccr: bool = typer.Option(True, "--ccr/--no-ccr", help="Point CCR at the proxy"),
):
    """Run a local auto-spin-up proxy: the first Claude Code message spins the box,
    streams warm-up progress into the chat, then answers. Auto-destroys on idle."""
    s = Settings.load()

    prof = profiles.get(profile) if profile else None
    if profile and not prof:
        console.print(f"[red]No profile '{profile}'.[/] See [bold]aiod profile list[/].")
        raise typer.Exit(1)
    model = model or (prof.model if prof else None)
    if not model:
        console.print("[red]Provide --model or --profile.[/]")
        raise typer.Exit(1)
    provider = (provider or (prof.provider if prof else "vast")).lower()
    _require_provider_key(s, provider)

    spin_kwargs = dict(
        model=model,
        quant=quant or (prof.quant if prof else "bf16"),
        provider=provider,
        max_price=max_price if max_price is not None else (prof.max_price if prof else None),
        ttl_hours=ttl if ttl is not None else (prof.ttl_hours if prof else None),
        idle_minutes=idle,
        context=context if context is not None else (prof.context if prof else None),
        concurrency=prof.concurrency if prof else 4,
        tool_parser=prof.tool_call_parser if prof else None,  # None -> model_configs
        extra_args=list(prof.extra_vllm_args) if prof else [],
    )

    base = f"http://127.0.0.1:{port}/v1"
    if write_ccr:
        ccr.write_config(base, s.vllm_api_key, model)
        console.print("[green]✓[/] CCR pointed at the proxy. Run [bold]ccr restart && ccr code[/].")

    console.print(
        Panel(
            f"Listening on [bold]http://127.0.0.1:{port}[/]\n"
            f"model: {model} ({spin_kwargs['quant']}) · provider: {spin_kwargs['provider']} · "
            f"idle-shutdown: {idle}m\n"
            f"First message streams warm-up progress, then answers.\n"
            f"Status: [bold]aiod status[/] / the TUI / GET /aiod/status  ·  Ctrl-C stops the proxy.",
            title="aiod proxy (auto spin-up)",
            border_style="green",
        )
    )
    from .proxy import run_proxy

    try:
        run_proxy(
            s, spin_kwargs, idle_minutes=idle, port=port,
            on_event=lambda phase, msg: console.log(f"[{phase}] {msg}"),
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Proxy stopped[/] — any running box is left up (aiod status/teardown).")


@app.command()
def bench(
    n: int = typer.Option(8, "--n", help="Number of requests"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Parallel requests"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Output tokens per request"),
    prompt: str = typer.Option(None, "--prompt", help="Override the benchmark prompt"),
):
    """Benchmark the running model: TTFT, tokens/sec, throughput, $/1M tokens."""
    from .bench import DEFAULT_PROMPT, run_benchmark

    inst = state.load()
    if inst is None or not inst.base_url:
        console.print("No running instance with an endpoint. Run `aiod spin` first.")
        raise typer.Exit(1)

    console.print(
        f"Benchmarking [bold]{inst.repo_id}[/] @ {inst.base_url}\n"
        f"[dim]n={n} · concurrency={concurrency} · max_tokens={max_tokens}[/]"
    )
    with console.status("Running benchmark..."):
        res = run_benchmark(
            inst.base_url, inst.repo_id, api_key=inst.api_key, n=n, concurrency=concurrency,
            max_tokens=max_tokens, prompt=prompt or DEFAULT_PROMPT, price_per_hr=inst.price_per_hr,
        )

    ok = len(res.ok)
    if not ok:
        errs = {r.error for r in res.results if r.error}
        console.print(f"[red]All {n} requests failed.[/] {', '.join(list(errs)[:2])}")
        raise typer.Exit(1)

    def fmt(v, unit="", nd=1):
        return f"{v:.{nd}f}{unit}" if v is not None else "—"

    table = Table(title=f"Benchmark — {inst.gpu_desc} @ ${inst.price_per_hr:.2f}/hr")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Requests ok", f"{ok}/{n}")
    table.add_row("TTFT p50", fmt(res.ttft_p50, "s", 2))
    table.add_row("TTFT p95", fmt(res.ttft_p95, "s", 2))
    table.add_row("Decode speed (per req)", fmt(res.avg_decode_tok_s, " tok/s"))
    table.add_row("Throughput (aggregate)", fmt(res.throughput_tok_s, " tok/s"))
    table.add_row("Output tokens total", str(res.total_completion_tokens))
    table.add_row("[bold]$ / 1M output tokens[/]", f"[bold]{fmt(res.cost_per_million, '', 2)}[/]")
    console.print(table)
    if concurrency == 1:
        console.print("[dim]Tip: re-run with `-c 8` to measure throughput + cheaper $/1M.[/]")


@app.command()
def tui():
    """Launch the interactive Textual control center."""
    from .tui import run_tui

    run_tui()


if __name__ == "__main__":
    app()
