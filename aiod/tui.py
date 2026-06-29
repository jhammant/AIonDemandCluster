"""Interactive Textual control center — the primary way to use aiod.

One screen to:
  * pick a profile (or type a HuggingFace link) + quant + provider + idle window
  * estimate live vast.ai cost, then launch
  * see the running instance (status, cost so far, idle/TTL) refreshed live
  * ping it, or tear it down

Blocking HTTP/vast work runs in thread workers so the UI stays responsive; UI
updates from those threads go through `call_from_thread`.
"""

from __future__ import annotations

import time

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    RichLog,
    Select,
    Static,
)

from . import ccr, events, model_configs, profiles, providers, state, watch
from .bootstrap import CONTAINER_PORT, ServerConfig
from .config import Settings
from .health import check_once, sample_completion
from .sizing import QUANT_LABELS, size_any
from .vast import PricedOption, recommend_disk_gb

QUANTS = ["bf16", "fp8", "awq-int4"]


class AiodTUI(App):
    CSS = """
    .row { height: auto; }
    .hdr { padding: 1 1 0 1; text-style: bold; }
    #link { width: 2fr; }
    #profile, #quant, #provider, #engine { width: 1fr; }
    #idle { width: 18; }
    #info { height: auto; padding: 0 1; color: $text-muted; }
    DataTable { height: 9; }
    #status { height: auto; padding: 0 1; border: round $secondary; }
    #log { height: 1fr; border: round $primary; }
    Button { margin: 0 1; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings.load()
        self.sizing = None
        self.priced: dict[str, PricedOption] = {}
        self.row_quants: list[str] = []
        self.busy = False
        self._last_event_ts = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("Launch", classes="hdr")
            with Horizontal(classes="row"):
                yield Select(
                    [(f"{n}", n) for n in sorted(profiles.all_profiles())],
                    prompt="profile…",
                    id="profile",
                    allow_blank=True,
                )
                yield Input(placeholder="…or HuggingFace link / org/name", id="link")
                yield Select([(q, q) for q in QUANTS], prompt="quant", id="quant", allow_blank=True)
            with Horizontal(classes="row"):
                yield Select(
                    [("vast", "vast"), ("runpod", "runpod")],
                    value="vast",
                    id="provider",
                )
                yield Select(
                    [("engine: auto", "auto"), ("vLLM", "vllm"), ("llama.cpp (GGUF)", "llamacpp")],
                    value="auto",
                    id="engine",
                )
                yield Input(placeholder="idle min", id="idle")
            with Horizontal(classes="row"):
                yield Button("Estimate", id="estimate", variant="primary")
                yield Button("Dry-run", id="dryrun")
                yield Button("Launch", id="launch", variant="success", disabled=True)
            yield Static("", id="info")
            yield DataTable(id="table", cursor_type="row")
            yield Static("Running instance", classes="hdr")
            yield Static("No instance running.", id="status")
            with Horizontal(classes="row"):
                yield Button("Ping", id="ping")
                yield Button("Teardown", id="teardown", variant="error")
                yield Button("Refresh", id="refresh")
            yield RichLog(id="log", highlight=True, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#table", DataTable)
        t.add_columns("Quant", "Detail", "GPUs", "VRAM need", "$/hr", f"$/{self._ttl():g}h")
        if not self.settings.vast_api_key:
            self._log("[red]VAST_API_KEY not set.[/] Run `aiod init` (or add it to .env).")
        else:
            self._log("[dim]Pick a profile or enter a model, then Estimate.[/]")
        self.refresh_status()
        self.set_interval(5.0, self.refresh_status)

    def _ttl(self) -> float:
        return self.settings.ttl_hours

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def _drain_events(self) -> None:
        """Stream any new warm-up events (from a proxy/spin) into the log."""
        for e in events.read(40):
            if e["ts"] > self._last_event_ts:
                self._last_event_ts = e["ts"]
                self._log(f"[dim][{e['phase']}][/] {e['msg']}")

    # ------------------------------------------------------------------ #
    # Profile select fills the launch fields
    # ------------------------------------------------------------------ #

    @on(Select.Changed, "#profile")
    def _on_profile(self, event: Select.Changed) -> None:
        if event.value in (None, Select.BLANK):
            return
        p = profiles.get(str(event.value))
        if not p:
            return
        self.query_one("#link", Input).value = p.model
        self.query_one("#quant", Select).value = p.quant if p.quant in QUANTS else Select.BLANK
        self.query_one("#provider", Select).value = p.provider
        self.query_one("#idle", Input).value = str(p.idle_minutes or "")
        self._log(f"[cyan]Profile[/] {p.name}: {p.model} ({p.quant})")

    # ------------------------------------------------------------------ #
    # Estimate
    # ------------------------------------------------------------------ #

    @on(Button.Pressed, "#estimate")
    @on(Input.Submitted, "#link")
    def _do_estimate(self) -> None:
        link = self.query_one("#link", Input).value.strip()
        if not link:
            self._log("[red]Enter a model or pick a profile first.[/]")
            return
        if not self.settings.vast_api_key:
            self._log("[red]VAST_API_KEY not set.[/]")
            return
        sel = self.query_one("#quant", Select).value
        quants = [sel] if sel and sel != Select.BLANK else QUANTS
        provider = str(self.query_one("#provider", Select).value or "vast")
        engine = str(self.query_one("#engine", Select).value or "auto")
        self.query_one("#estimate", Button).disabled = True
        self.query_one("#launch", Button).disabled = True
        self._log(f"[cyan]Sizing[/] {link} ({provider} · {engine}) ...")
        self._estimate_worker(link, quants, provider, engine)

    @work(thread=True, exclusive=True)
    def _estimate_worker(self, link: str, quants: list[str], provider: str, engine: str) -> None:
        try:
            sizing = size_any(link, engine=engine, hf_token=self.settings.hf_token, quants=quants)
        except Exception as e:  # noqa: BLE001 - surface any HF lookup failure to the UI
            self.call_from_thread(self._log, f"[red]Sizing failed:[/] {e}")
            self.call_from_thread(self._enable_estimate)
            return

        m = sizing.model
        if sizing.engine == "llamacpp":
            info = f"{m.repo_id} — GGUF · {len(sizing.plans)} quant build(s) · engine: llama.cpp"
        else:
            info = (
                f"{m.repo_id} — {m.params_b:.1f}B params · {m.dtype} · "
                f"ctx {m.max_context or '?'} · {'gated' if m.gated else 'open'}"
                + ("  (size guessed from name)" if m.params_source != "safetensors" else "")
            )
        self.call_from_thread(self.query_one("#info", Static).update, info)

        rows: list[tuple] = []
        priced_map: dict[str, PricedOption] = {}
        try:
            with providers.get_client(provider, self.settings) as client:
                for p in sizing.plans:
                    disk = recommend_disk_gb(p.weights_gb)
                    priced = client.price_plan(p, disk, max_price=self.settings.max_price)
                    best = min(
                        (x for x in priced if x.offer), key=lambda x: x.offer.dph_total, default=None
                    )
                    priced_map[p.quant] = best
                    if best:
                        rows.append(
                            (
                                p.quant,
                                QUANT_LABELS.get(p.quant, p.quant),
                                best.offer.desc,
                                f"{p.required_vram_gb:.0f} GB",
                                f"${best.offer.dph_total:.2f}",
                                f"${best.offer.dph_total * self._ttl():.2f}",
                            )
                        )
                    else:
                        rows.append(
                            (p.quant, QUANT_LABELS.get(p.quant, p.quant), "no offer",
                             f"{p.required_vram_gb:.0f} GB", "—", "—")
                        )
        except providers.PROVIDER_ERRORS as e:
            self.call_from_thread(self._log, f"[red]provider error:[/] {e}")
            self.call_from_thread(self._enable_estimate)
            return

        self.sizing = sizing
        self.priced = priced_map
        self.call_from_thread(self._populate_table, rows)
        self.call_from_thread(self._log, "[green]Estimate ready.[/] Select a row and Launch.")
        self.call_from_thread(self._enable_estimate)

    def _populate_table(self, rows: list[tuple]) -> None:
        t = self.query_one("#table", DataTable)
        t.clear()
        self.row_quants = []
        for r in rows:
            t.add_row(*r)
            self.row_quants.append(r[0])
        launchable = any(self.priced.get(q) for q in self.row_quants)
        self.query_one("#launch", Button).disabled = not launchable or bool(state.load())

    def _enable_estimate(self) -> None:
        self.query_one("#estimate", Button).disabled = False

    # ------------------------------------------------------------------ #
    # Launch
    # ------------------------------------------------------------------ #

    @on(Button.Pressed, "#launch")
    def _do_launch(self) -> None:
        if self.busy:
            return
        if state.load():
            self._log("[yellow]Instance already tracked — teardown first.[/]")
            return
        provider = str(self.query_one("#provider", Select).value or "vast")
        t = self.query_one("#table", DataTable)
        if t.cursor_row is None or t.cursor_row >= len(self.row_quants):
            self._log("[red]Select a row first.[/]")
            return
        quant = self.row_quants[t.cursor_row]
        best = self.priced.get(quant)
        if not best or not best.offer:
            self._log(f"[red]No offer for {quant}.[/]")
            return
        idle_m = self._parse_idle()
        self.busy = True
        self.query_one("#launch", Button).disabled = True
        self._log(
            f"[cyan]Launching[/] {self.sizing.model.repo_id} ({quant}) on {best.offer.desc} "
            f"@ ${best.offer.dph_total:.2f}/hr"
            + (f" · idle-shutdown {idle_m}m" if idle_m else "")
        )
        self._launch_worker(quant, best, provider, idle_m)

    def _parse_idle(self) -> int | None:
        raw = self.query_one("#idle", Input).value.strip()
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def _make_cfg(self, quant: str, best: PricedOption):
        s = self.settings
        m = self.sizing.model
        eng = self.sizing.engine
        mc = model_configs.resolve(m.repo_id)
        disk = recommend_disk_gb(self.sizing.plan(quant).weights_gb)
        cfg = ServerConfig(
            repo_id=m.repo_id,
            num_gpus=best.option.num_gpus,
            quant=quant,
            api_key=s.vllm_api_key,
            engine=eng,
            port=CONTAINER_PORT,
            tool_call_parser=mc.tool_call_parser,
            extra_args=mc.vllm_serving_args(),
            hf_token=s.hf_token,
            gguf_quant=quant if eng == "llamacpp" else None,
        )
        return cfg, disk, eng

    def _selected_quant_offer(self):
        """The (quant, PricedOption) for the highlighted table row, or (None, None)."""
        if not self.sizing:
            self._log("[yellow]Run Estimate first.[/]")
            return None, None
        t = self.query_one("#table", DataTable)
        if t.cursor_row is None or t.cursor_row >= len(self.row_quants):
            self._log("[red]Select a row first.[/]")
            return None, None
        quant = self.row_quants[t.cursor_row]
        best = self.priced.get(quant)
        if not best or not best.offer:
            self._log(f"[red]No offer for {quant}.[/]")
            return None, None
        return quant, best

    @on(Button.Pressed, "#dryrun")
    def _do_dryrun(self) -> None:
        import json

        quant, best = self._selected_quant_offer()
        if quant is None:
            return
        provider = str(self.query_one("#provider", Select).value or "vast")
        cfg, disk, eng = self._make_cfg(quant, best)
        offer = best.offer
        try:
            with providers.get_client(provider, self.settings) as client:
                if provider == "runpod":
                    body = client.build_create_body(
                        cfg, disk_gb=disk, max_price=self.settings.max_price,
                        gpu_type_id=str(offer.id),
                    )
                else:
                    body = client.build_create_body(
                        cfg, disk_gb=disk, max_price=self.settings.max_price
                    )
        except providers.PROVIDER_ERRORS as e:
            self._log(f"[red]{e}[/]")
            return

        redacted = json.loads(json.dumps(body))
        for k in list(redacted.get("env", {})):
            if "TOKEN" in k:
                redacted["env"][k] = "***redacted***"
        for key in ("args", "dockerStartCmd"):
            arr = redacted.get(key)
            if isinstance(arr, list):
                for i, v in enumerate(arr):
                    if v == "--api-key" and i + 1 < len(arr):
                        arr[i + 1] = "***redacted***"
        self._log(
            f"[yellow]DRY RUN[/] — would rent [bold]{offer.desc}[/] @ ${offer.dph_total:.2f}/hr "
            f"· {cfg.image} · disk {disk}GB (nothing rented)"
        )
        self._log(json.dumps(redacted, indent=2))

    @work(thread=True, exclusive=True)
    def _launch_worker(self, quant: str, best: PricedOption, provider: str, idle_m: int | None) -> None:
        s = self.settings
        m = self.sizing.model
        cfg, disk, eng = self._make_cfg(quant, best)
        offer = best.offer
        try:
            with providers.get_client(provider, s) as client:
                instance_id = client.create_instance(
                    offer.id, cfg, disk_gb=disk, max_price=s.max_price, label="aiod-vllm"
                )
                inst = state.Instance(
                    instance_id=instance_id,
                    repo_id=m.repo_id,
                    quant=quant,
                    gpu_desc=offer.desc,
                    price_per_hr=offer.dph_total,
                    created_at=time.time(),
                    ttl_hours=s.ttl_hours,
                    api_key=s.vllm_api_key,
                    status="creating",
                    provider=provider,
                    idle_minutes=idle_m,
                )
                state.save(inst)
                self.call_from_thread(self._log, f"[green]✓[/] Instance {instance_id} created.")

                ep = None
                start = time.time()
                while time.time() - start < 1200:  # slow nodes pull the image slowly
                    vi = client.get_instance(instance_id)
                    ep = client.endpoint_of(vi, CONTAINER_PORT)
                    self.call_from_thread(
                        self._log, f"  vast: {client.status_of(vi)} ({int(time.time()-start)}s)"
                    )
                    if ep:
                        break
                    time.sleep(10)
                if not ep:
                    self.call_from_thread(self._log, "[red]Port never mapped — teardown & retry.[/]")
                    self.call_from_thread(self._done_launch)
                    return
                inst.host, inst.port, inst.status = ep[0], ep[1], "loading"
                state.save(inst)
                self.call_from_thread(self._log, f"[green]✓[/] {inst.base_url} — loading weights...")
        except providers.PROVIDER_ERRORS as e:
            self.call_from_thread(self._log, f"[red]Provider error:[/] {e}")
            self.call_from_thread(self._done_launch)
            return

        start = time.time()
        health_timeout = 5400 if eng == "llamacpp" else 2400  # GGUF downloads are huge
        log_client = None
        try:
            log_client = providers.get_client(provider, s)
        except Exception:  # noqa: BLE001
            log_client = None
        dl, last_log = None, 0.0
        while time.time() - start < health_timeout:
            hs = check_once(inst.base_url, api_key=inst.api_key)
            el = time.time() - start
            if log_client is not None and hasattr(log_client, "fetch_logs") and el - last_log > 20:
                last_log = el
                try:
                    from .vast import extract_download_progress
                    p = extract_download_progress(log_client.fetch_logs(inst.instance_id))
                    if p:
                        dl = p
                except Exception:  # noqa: BLE001 - progress is best-effort
                    pass
            self.call_from_thread(
                self._log, f"  {hs.detail} ({int(el)}s)" + (f"  ·  ⬇ {dl}" if dl else "")
            )
            if hs.ready:
                break
            time.sleep(12)
        if log_client is not None:
            try:
                log_client.close()
            except Exception:  # noqa: BLE001
                pass
        else:
            self.call_from_thread(self._log, "[yellow]Timed out waiting for model. Check status.[/]")
            self.call_from_thread(self._done_launch)
            return

        inst.status = "running"
        state.save(inst)
        path = ccr.write_config(inst.base_url, inst.api_key, inst.repo_id)
        msg = (
            f"[bold green]✓ Live:[/] {inst.base_url}\n"
            f"[green]✓[/] CCR config: {path} — run [bold]ccr restart && ccr code[/]"
        )
        if idle_m:
            ok = watch.spawn_detached(idle_m, state.STATE_DIR / "watch.log")
            msg += (
                f"\n[green]✓[/] Idle watcher started ({idle_m}m)"
                if ok
                else f"\n[yellow]![/] Start watcher manually: aiod watch --idle {idle_m}"
            )
        self.call_from_thread(self._log, msg)
        self.call_from_thread(self._done_launch)

    def _done_launch(self) -> None:
        self.busy = False
        self.refresh_status()

    # ------------------------------------------------------------------ #
    # Running-instance panel + management
    # ------------------------------------------------------------------ #

    def action_refresh(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        self._drain_events()
        inst = state.load()
        st = self.query_one("#status", Static)
        if inst is None:
            ev = events.latest()
            if ev and ev["phase"] not in ("ready", "destroyed"):
                st.update(f"Warming up… [b]{ev['phase']}[/] {ev['msg']}")
            else:
                st.update("No instance running.")
            return
        idle = f" · idle-shutdown {inst.idle_minutes}m" if inst.idle_minutes else ""
        ttl = (
            f"[red]TTL exceeded {-inst.expires_in_hours:.1f}h[/]"
            if inst.expires_in_hours < 0
            else f"{inst.expires_in_hours:.1f}h left"
        )
        st.update(
            f"[bold]{inst.repo_id}[/] ({inst.quant}) on {inst.gpu_desc} via {inst.provider}\n"
            f"status: {inst.status}{idle}  ·  {inst.base_url or 'no endpoint yet'}\n"
            f"${inst.price_per_hr:.2f}/hr · up {inst.age_hours:.2f}h "
            f"(~${inst.est_cost_so_far:.2f}) · {ttl}"
        )

    @on(Button.Pressed, "#refresh")
    def _refresh_btn(self) -> None:
        self.refresh_status()
        self._log("[dim]status refreshed[/]")

    @on(Button.Pressed, "#ping")
    def _do_ping(self) -> None:
        inst = state.load()
        if inst is None or not inst.base_url:
            self._log("[yellow]Nothing running to ping.[/]")
            return
        self._log("[cyan]Pinging…[/]")
        self._ping_worker(inst.base_url, inst.repo_id, inst.api_key)

    @work(thread=True, exclusive=True)
    def _ping_worker(self, base_url: str, model: str, api_key: str | None) -> None:
        res = sample_completion(base_url, model, api_key=api_key)
        if res["ok"]:
            self.call_from_thread(
                self._log, f"[green]✓[/] {res['latency_s']:.1f}s: {res['text']!r}"
            )
        else:
            self.call_from_thread(self._log, f"[red]✗ ping failed:[/] {res['error']}")

    @on(Button.Pressed, "#teardown")
    def _do_teardown(self) -> None:
        inst = state.load()
        if not inst:
            self._log("Nothing tracked to tear down.")
            return
        self._log(f"[cyan]Destroying[/] instance {inst.instance_id} …")
        self._teardown_worker(inst.instance_id, inst.provider)

    @work(thread=True, exclusive=True)
    def _teardown_worker(self, instance_id: int, provider: str) -> None:
        try:
            with providers.get_client(provider, self.settings) as client:
                client.destroy_instance(instance_id)
            state.clear()
            self.call_from_thread(self._log, f"[green]✓[/] Destroyed {instance_id}.")
        except providers.PROVIDER_ERRORS as e:
            self.call_from_thread(
                self._log,
                f"[red]Destroy failed:[/] {e}\nCheck the provider console manually.",
            )
        self.call_from_thread(self.refresh_status)


def run_tui() -> None:
    AiodTUI().run()
