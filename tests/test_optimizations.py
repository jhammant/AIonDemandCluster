"""Tests for the pluggable optimization layer (interface, serving wire, sizing
hook, hardware gate, CLI/profile wiring) including the load-bearing invariants."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aiod import optimizations as opt
from aiod import profiles
from aiod.bootstrap import ServerConfig
from aiod.optimizations import (
    BaseOpt,
    OptContext,
    SizingKnobs,
    _check_conflicts,
    parse_selection,
    resolve,
    vllm_flags,
)
from aiod.profiles import Profile
from aiod.sizing import ModelSpec, QuantPlan, estimate_vram, plan_gpus, size_model

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _spec(params: int, **kw) -> ModelSpec:
    base = dict(
        repo_id="test/model",
        params=params,
        dtype="BF16",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        max_context=8192,
        architecture="llama",
        gated=False,
        params_source="safetensors",
    )
    base.update(kw)
    return ModelSpec(**base)


def _cfg(**kw) -> ServerConfig:
    base = dict(repo_id="org/model", num_gpus=2, quant="bf16", api_key="sk-tok")
    base.update(kw)
    return ServerConfig(**base)


# Golden argv with NO optimizations (the pre-change output). Byte-identity here
# is the back-compat invariant: an empty opt list must not perturb a single byte.
_VLLM_GOLDEN = [
    "--host", "0.0.0.0",
    "--port", "8000",
    "--model", "org/model",
    "--served-model-name", "org/model",
    "--api-key", "sk-tok",
    "--tensor-parallel-size", "2",
    "--gpu-memory-utilization", "0.92",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
]
_LLAMACPP_GOLDEN = [
    "-hf", "org/m-GGUF:Q4_K_M",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--api-key", "sk-tok",
    "-ngl", "999",
    "--jinja",
    "-c", "32768",
]


# --------------------------------------------------------------------------- #
# Byte-identity / back-compat
# --------------------------------------------------------------------------- #

def test_vllm_args_empty_opts_byte_identical():
    assert _cfg().server_args() == _VLLM_GOLDEN
    assert _cfg(optimizations=[]).server_args() == _VLLM_GOLDEN


def test_llamacpp_args_empty_opts_byte_identical():
    cfg = _cfg(engine="llamacpp", gguf_quant="Q4_K_M", repo_id="org/m-GGUF")
    assert cfg.server_args() == _LLAMACPP_GOLDEN
    cfg2 = _cfg(engine="llamacpp", gguf_quant="Q4_K_M", repo_id="org/m-GGUF", optimizations=[])
    assert cfg2.server_args() == _LLAMACPP_GOLDEN


def test_default_selection_is_empty():
    # All built-ins ship default_on=False -> the no-opt path is trivially proven.
    assert opt.default_selection() == []


# --------------------------------------------------------------------------- #
# Serving flag emission
# --------------------------------------------------------------------------- #

def test_prefix_caching_appends_exactly_one_flag_vllm():
    args = _cfg(optimizations=["prefix-caching"]).server_args()
    assert args == _VLLM_GOLDEN + ["--enable-prefix-caching"]


def test_prefix_caching_is_noop_on_llamacpp():
    cfg = _cfg(engine="llamacpp", gguf_quant="Q4_K_M", repo_id="org/m-GGUF",
               optimizations=["prefix-caching"])
    # engine != vllm => applies() False => no flag, argv byte-identical.
    assert cfg.server_args() == _LLAMACPP_GOLDEN


def test_registry_order_argv_reproducible():
    a = _cfg(optimizations=["prefix-caching", "kv-cache-fp8"]).server_args()
    b = _cfg(optimizations=["kv-cache-fp8", "prefix-caching"]).server_args()
    assert a == b
    # kv-cache-fp8 precedes prefix-caching in the registry, so that's the order.
    assert a == _VLLM_GOLDEN + ["--kv-cache-dtype", "fp8", "--enable-prefix-caching"]


def test_extra_args_still_last_after_opt_flags():
    args = _cfg(optimizations=["prefix-caching"], extra_args=["--seed", "1"]).server_args()
    assert args[-2:] == ["--seed", "1"]
    assert args.index("--enable-prefix-caching") < args.index("--seed")


def test_chunked_prefill_value_form():
    args = _cfg(optimizations=["chunked-prefill"],
               opt_values={"chunked-prefill": "2048"}).server_args()
    assert args[-3:] == ["--enable-chunked-prefill", "--max-num-batched-tokens", "2048"]


def test_max_num_seqs_flag_uses_value_or_default():
    a = _cfg(optimizations=["max-num-seqs"]).server_args()
    assert a[-2:] == ["--max-num-seqs", "256"]
    b = _cfg(optimizations=["max-num-seqs"], opt_values={"max-num-seqs": "512"}).server_args()
    assert b[-2:] == ["--max-num-seqs", "512"]


# --------------------------------------------------------------------------- #
# parse_selection / resolve / conflicts
# --------------------------------------------------------------------------- #

def test_parse_selection_splits_keys_and_values():
    keys, values = parse_selection(["max-num-seqs=256", "prefix-caching"])
    assert keys == ["max-num-seqs", "prefix-caching"]
    assert values == {"max-num-seqs": "256"}


def test_resolve_rejects_value_on_toggle_only_opt():
    ctx = OptContext(engine="vllm", quant="bf16", repo_id="org/m")
    with pytest.raises(ValueError):
        resolve(["prefix-caching"], {"prefix-caching": "x"}, ctx)


def test_resolve_records_unknown_as_skipped():
    ctx = OptContext(engine="vllm", quant="bf16", repo_id="org/m")
    r = resolve(["bogus"], {}, ctx)
    assert ("bogus", "unknown") in r.skipped
    assert r.keys == []


def test_check_conflicts_duplicate_provides():
    a = BaseOpt(key="a", summary="", tradeoff="", provides=("x",))
    b = BaseOpt(key="b", summary="", tradeoff="", provides=("x",))
    with pytest.raises(ValueError):
        _check_conflicts([a, b])


def test_check_conflicts_conflicts_with():
    a = BaseOpt(key="a", summary="", tradeoff="")
    c = BaseOpt(key="c", summary="", tradeoff="", conflicts_with=("a",))
    with pytest.raises(ValueError):
        _check_conflicts([a, c])


# --------------------------------------------------------------------------- #
# Entry-point loading (fail-soft + warn, later-wins override)
# --------------------------------------------------------------------------- #

class _BadEP:
    name = "broken-plugin"

    def load(self):
        raise RuntimeError("boom")


class _GoodEP:
    name = "override-plugin"

    def load(self):
        return BaseOpt(key="kv-cache-fp8", summary="overridden", tradeoff="")


def test_entrypoint_failure_is_skipped_and_warns(monkeypatch, capsys):
    monkeypatch.setattr(opt, "_iter_entry_points", lambda: [_BadEP()])
    reg = opt.registry()
    # Built-ins still resolve despite the broken plugin.
    assert "kv-cache-fp8" in reg and "prefix-caching" in reg
    err = capsys.readouterr().err
    assert "broken-plugin" in err and "skipping" in err.lower()


def test_plugin_overrides_builtin_later_wins(monkeypatch, capsys):
    monkeypatch.setattr(opt, "_iter_entry_points", lambda: [_GoodEP()])
    reg = opt.registry()
    assert reg["kv-cache-fp8"].summary == "overridden"
    err = capsys.readouterr().err
    assert "overrides built-in" in err


# --------------------------------------------------------------------------- #
# Sizing: float-identity, kv halving, growth, order independence
# --------------------------------------------------------------------------- #

def test_estimate_vram_knobs_none_float_identical():
    for params in (7_000_000_000, 32_000_000_000, 70_000_000_000):
        spec = _spec(params)
        assert estimate_vram(spec, "bf16", 8192) == estimate_vram(spec, "bf16", 8192, knobs=None)


def test_estimate_vram_knobs_none_float_identical_fallback_path():
    # No config shape -> the weights*0.15 fallback branch.
    spec = _spec(7_000_000_000, num_layers=None, hidden_size=None, num_attention_heads=None)
    assert estimate_vram(spec, "bf16", 8192) == estimate_vram(spec, "bf16", 8192, knobs=None)


def test_kv_cache_fp8_halves_kv_shaped_path():
    spec = _spec(70_000_000_000)
    _, kv0, _ = estimate_vram(spec, "bf16", 100_000)
    knobs = opt.get("kv-cache-fp8").sizing(SizingKnobs(), OptContext("vllm", "bf16", "org/m"))
    _, kv1, _ = estimate_vram(spec, "bf16", 100_000, knobs=knobs)
    assert kv1 == pytest.approx(kv0 * 0.5)


def test_kv_cache_fp8_halves_kv_fallback_path():
    # Config-less: kv = weights*0.15, and kv-cache-fp8 scales the OUTPUT.
    spec = _spec(7_000_000_000, num_layers=None, hidden_size=None, num_attention_heads=None)
    _, kv0, _ = estimate_vram(spec, "bf16", 8192)
    knobs = opt.get("kv-cache-fp8").sizing(SizingKnobs(), OptContext("vllm", "bf16", "org/m"))
    _, kv1, _ = estimate_vram(spec, "bf16", 8192, knobs=knobs)
    assert kv1 == pytest.approx(kv0 * 0.5)
    # weights unchanged (weight_scale stays 1.0).
    w0, _, _ = estimate_vram(spec, "bf16", 8192)
    w1, _, _ = estimate_vram(spec, "bf16", 8192, knobs=knobs)
    assert w1 == w0


def test_max_num_seqs_grows_kv_and_is_noop_at_concurrency():
    spec = _spec(7_000_000_000)
    o = opt.get("max-num-seqs")
    # value=256, concurrency=4 -> factor 64.
    ctx = OptContext("vllm", "bf16", "org/m", concurrency=4, value="256")
    knobs = o.sizing(SizingKnobs(), ctx)
    _, kv0, _ = estimate_vram(spec, "bf16", 8192)
    _, kv1, _ = estimate_vram(spec, "bf16", 8192, knobs=knobs)
    assert kv1 == pytest.approx(kv0 * 64)
    # value==concurrency -> no-op.
    ctx2 = OptContext("vllm", "bf16", "org/m", concurrency=4, value="4")
    assert o.sizing(SizingKnobs(), ctx2).kv_scale == 1.0


def test_kv_scale_order_independent():
    ctx = OptContext("vllm", "bf16", "org/m", concurrency=4, value="256")
    fp8 = opt.get("kv-cache-fp8")
    mns = opt.get("max-num-seqs")
    forward = mns.sizing(fp8.sizing(SizingKnobs(), ctx), ctx)
    reverse = fp8.sizing(mns.sizing(SizingKnobs(), ctx), ctx)
    assert forward.kv_scale == pytest.approx(reverse.kv_scale)


# --------------------------------------------------------------------------- #
# Sizing end-to-end: tier flip + compute-cap stamp (size_model)
# --------------------------------------------------------------------------- #

def _patch_spec(monkeypatch, spec):
    monkeypatch.setattr("aiod.sizing.fetch_model_spec", lambda *a, **k: spec)


def test_kv_cache_fp8_flips_tier_and_sets_min_cc(monkeypatch):
    # A KV-heavy 7B at high concurrency lands on a bigger tier at baseline; the
    # fp8 KV shrink drops it to a cheaper tier.
    spec = _spec(7_000_000_000)
    _patch_spec(monkeypatch, spec)

    base = size_model("test/model", quants=["bf16"], context_len=8192, concurrency=12).plan("bf16")
    optimized = size_model(
        "test/model", quants=["bf16"], context_len=8192, concurrency=12,
        opts=["kv-cache-fp8"], opt_values={},
    ).plan("bf16")

    # required shrinks
    assert optimized.required_vram_gb < base.required_vram_gb
    # tier flips to something cheaper (different tier name)
    assert base.cheapest_fit.tier.name != optimized.cheapest_fit.tier.name
    # hardware floor stamped + applied keys recorded
    assert optimized.min_compute_cap == 890
    assert optimized.applied_opts == ["kv-cache-fp8"]
    # baseline untouched
    assert base.min_compute_cap == 800
    assert base.applied_opts == []


def test_size_model_no_opts_float_identical(monkeypatch):
    spec = _spec(32_000_000_000)
    _patch_spec(monkeypatch, spec)
    a = size_model("test/model", quants=["bf16", "fp8"], context_len=8192, concurrency=4)
    b = size_model("test/model", quants=["bf16", "fp8"], context_len=8192, concurrency=4,
                   opts=None)
    for pa, pb in zip(a.plans, b.plans, strict=True):
        assert pa.required_vram_gb == pb.required_vram_gb
        assert pa.kv_gb == pb.kv_gb
        assert pa.min_compute_cap == 800


# --------------------------------------------------------------------------- #
# Hardware gate flows into the vast offer search
# --------------------------------------------------------------------------- #

def _fitting_plan(quant, min_cc) -> QuantPlan:
    opts = plan_gpus(30.0)  # something small that fits on a single GPU
    return QuantPlan(
        quant=quant, label=quant, weights_gb=14.0, kv_gb=2.0,
        required_vram_gb=30.0, options=opts, min_compute_cap=min_cc,
    )


def test_vast_min_cc_uses_plan_floor(monkeypatch):
    from aiod.vast import VastClient

    client = VastClient(api_key="x")
    seen = {}

    def fake_search(**kw):
        seen["min_compute_cap"] = kw["min_compute_cap"]
        return []

    monkeypatch.setattr(client, "search_offers", fake_search)

    # bf16 + opt floor 890 -> 890 (kv-cache-fp8 case).
    client.price_plan(_fitting_plan("bf16", 890), min_disk_gb=40)
    assert seen["min_compute_cap"] == 890

    # bf16 default floor -> 800 (every existing path unchanged).
    client.price_plan(_fitting_plan("bf16", 800), min_disk_gb=40)
    assert seen["min_compute_cap"] == 800

    # fp8 weights still force 890 even with default plan floor.
    client.price_plan(_fitting_plan("fp8", 800), min_disk_gb=40)
    assert seen["min_compute_cap"] == 890


# --------------------------------------------------------------------------- #
# Split-brain regression: opt in sizing IFF its flag in argv
# --------------------------------------------------------------------------- #

def test_split_brain_sizing_keys_match_argv_flags():
    spec = _spec(7_000_000_000)
    selection = ["kv-cache-fp8", "speculative-decoding"]  # latter is inert (no draft model)

    # Sizing path: spec is set, applies() ignores it by contract.
    sizing_ctx = OptContext("vllm", "bf16", "test/model", concurrency=4, spec=spec)
    resolved = resolve(selection, {}, sizing_ctx)

    # Serving path: spec is None.
    serving_ctx = OptContext("vllm", "bf16", "test/model")
    argv = vllm_flags(selection, {}, serving_ctx)

    # kv-cache-fp8 applies on both -> in sizing keys AND its flag is in argv.
    assert "kv-cache-fp8" in resolved.keys
    assert "--kv-cache-dtype" in argv
    # speculative-decoding is inert -> in NEITHER.
    assert "speculative-decoding" not in resolved.keys
    assert "--speculative-config" not in argv


# --------------------------------------------------------------------------- #
# Profiles round-trip
# --------------------------------------------------------------------------- #

def test_profile_optimizations_roundtrip(tmp_path, monkeypatch):
    pf = tmp_path / "profiles.yaml"
    monkeypatch.setattr(profiles, "PROFILE_FILE", pf)
    profiles.save(Profile(name="opt-prof", model="org/m", optimizations=["kv-cache-fp8", "prefix-caching"]))
    loaded = profiles.all_profiles()["opt-prof"]
    assert loaded.optimizations == ["kv-cache-fp8", "prefix-caching"]


def test_profile_from_dict_defaults_empty():
    p = Profile.from_dict("x", {"model": "org/m"})
    assert p.optimizations == []


# --------------------------------------------------------------------------- #
# CLI: opt list, unknown --opt exits, up --detach forwards --opt
# --------------------------------------------------------------------------- #

def test_opt_list_runs_without_model():
    from aiod.cli import app
    result = runner.invoke(app, ["opt", "list"])
    assert result.exit_code == 0, result.output
    assert "kv-cache-fp8" in result.output
    assert "max-num-seqs" in result.output


def test_unknown_opt_exits_nonzero_and_lists_valid(monkeypatch):
    from aiod.cli import app
    fake_settings = SimpleNamespace(hf_token=None, max_price=6.0)
    monkeypatch.setattr("aiod.cli.Settings.load", staticmethod(lambda: fake_settings))
    monkeypatch.setattr("aiod.cli._require_provider_key", lambda s, provider: None)
    result = runner.invoke(app, ["estimate", "org/repo", "--opt", "no-such-opt"])
    assert result.exit_code != 0
    assert "Unknown optimization" in result.output
    assert "kv-cache-fp8" in result.output  # lists valid keys


def test_estimate_opt_shows_baseline_vs_optimized(monkeypatch):
    from rich.console import Console

    from aiod import providers
    from aiod.cli import app
    from aiod.sizing import SizingResult

    # Wide console so the fit cell (tier flip) isn't truncated under CliRunner.
    monkeypatch.setattr("aiod.cli.console", Console(width=240))
    fake_settings = SimpleNamespace(hf_token=None, max_price=6.0)
    monkeypatch.setattr("aiod.cli.Settings.load", staticmethod(lambda: fake_settings))
    monkeypatch.setattr("aiod.cli._require_provider_key", lambda s, provider: None)

    spec = _spec(7_000_000_000)

    def make_result(required, fit_required):
        plan = QuantPlan(
            quant="bf16", label="bf16", weights_gb=14.0, kv_gb=2.0,
            required_vram_gb=required, options=plan_gpus(fit_required),
        )
        return SizingResult(model=spec, context_tokens=8192, plans=[plan])

    baseline = make_result(60.0, 60.0)      # cheapest_fit -> A100 80GB (1x)
    optimized = make_result(40.0, 40.0)     # cheapest_fit -> RTX 6000 Ada (1x)

    def fake_size_any(*a, **k):
        return optimized if k.get("opts") else baseline

    monkeypatch.setattr("aiod.cli.size_any", fake_size_any)

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def price_plan(self, *a, **k):
            return []  # no live offers; sizing-tier flip still shown

    monkeypatch.setattr(providers, "get_client", lambda provider, s: _FakeClient())

    result = runner.invoke(
        app, ["estimate", "org/repo", "-q", "bf16", "--opt", "kv-cache-fp8"],
        env={"COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.output
    # baseline != optimized required, shown as an arrow
    assert "60→40 GB" in result.output
    # flipped tier (A100 -> RTX 6000 Ada) appears in the fit cell
    assert "A100" in result.output and "6000" in result.output


@pytest.fixture
def up_detach(monkeypatch):
    """Stub the detached-gateway side effects and capture the child argv."""
    captured = {}
    fake_settings = SimpleNamespace(vllm_api_key="sk-aiod-test")
    monkeypatch.setattr("aiod.cli.Settings.load", staticmethod(lambda: fake_settings))
    monkeypatch.setattr("aiod.cli._require_provider_key", lambda s, provider: None)
    monkeypatch.setattr("aiod.cli.was_token_minted", lambda s: False)
    monkeypatch.setattr(profiles, "get", lambda name: None)
    monkeypatch.setattr("aiod.cli.ccr.write_config", lambda *a, **k: None)
    monkeypatch.setattr("aiod.cli._poll_healthz", lambda *a, **k: True)

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr("aiod.cli.subprocess.Popen", fake_popen)
    return captured


def test_up_detach_forwards_opt_tokens(up_detach):
    from aiod.cli import app
    result = runner.invoke(
        app, ["up", "org/repo", "--detach", "--opt", "kv-cache-fp8", "--opt", "max-num-seqs=256"]
    )
    assert result.exit_code == 0, result.output
    cmd = up_detach["cmd"]
    # Each token re-emitted as a real --opt flag.
    assert "--opt" in cmd
    assert "kv-cache-fp8" in cmd
    assert "max-num-seqs=256" in cmd
    # exactly two --opt occurrences
    assert cmd.count("--opt") == 2
