import inspect
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aiod import ccr, engine, profiles, proxy
from aiod.cli import app

runner = CliRunner()


@pytest.fixture
def wired(monkeypatch):
    """Stub out the side-effecting pieces of `aiod up` and capture the wiring."""
    calls = {}

    fake_settings = SimpleNamespace(vllm_api_key="sk-aiod-test123")
    monkeypatch.setattr("aiod.cli.Settings.load", staticmethod(lambda: fake_settings))
    monkeypatch.setattr("aiod.cli._require_provider_key", lambda s, provider: None)
    monkeypatch.setattr("aiod.cli.was_token_minted", lambda s: False)
    # Default: no profile resolves (HF-link path). Tests override as needed.
    monkeypatch.setattr(profiles, "get", lambda name: None)

    def fake_write_config(base_url, api_key, model):
        calls["ccr"] = (base_url, api_key, model)

    def fake_run_gateway(s, spin_kwargs, idle_minutes, **kwargs):
        calls["run_gateway"] = {
            "settings": s,
            "spin_kwargs": spin_kwargs,
            "idle_minutes": idle_minutes,
            "kwargs": kwargs,
        }

    monkeypatch.setattr(ccr, "write_config", fake_write_config)
    monkeypatch.setattr(proxy, "run_gateway", fake_run_gateway)

    return calls


def test_up_wires_ccr_local_base_and_parsed_model(wired):
    result = runner.invoke(app, ["up", "meta-llama/Llama-3.1-8B-Instruct", "--port", "4000"])
    assert result.exit_code == 0, result.output
    base, api_key, model = wired["ccr"]
    assert base == "http://127.0.0.1:4000/v1"
    assert api_key == "sk-aiod-test123"
    assert model == "meta-llama/Llama-3.1-8B-Instruct"


def test_up_wires_ccr_with_custom_port(wired):
    result = runner.invoke(app, ["up", "org/repo", "--port", "5005"])
    assert result.exit_code == 0, result.output
    base, _, _ = wired["ccr"]
    assert base == "http://127.0.0.1:5005/v1"


def test_up_full_url_is_parsed(wired):
    result = runner.invoke(app, ["up", "https://huggingface.co/org/repo/tree/main"])
    assert result.exit_code == 0, result.output
    assert wired["ccr"][2] == "org/repo"


def test_up_spin_kwargs_match_engine_launch_signature(wired):
    result = runner.invoke(app, ["up", "org/repo"])
    assert result.exit_code == 0, result.output
    spin_kwargs = wired["run_gateway"]["spin_kwargs"]
    # `startup_grace` is an additive launch-tuning kwarg (used by `aiod tune` for
    # early bad-node abort); like `on_event` it's not part of the spin config the
    # gateway threads through, so it's excluded from this in-sync check.
    launch_params = set(inspect.signature(engine.launch).parameters) - {
        "s", "on_event", "startup_grace"
    }
    assert set(spin_kwargs) == launch_params


def test_up_eager_spin_passed_to_run_gateway(wired):
    result = runner.invoke(app, ["up", "org/repo"])
    assert result.exit_code == 0, result.output
    assert wired["run_gateway"]["kwargs"]["eager_spin"] is True


def test_up_no_spin_disables_eager_spin(wired):
    result = runner.invoke(app, ["up", "org/repo", "--no-spin"])
    assert result.exit_code == 0, result.output
    assert wired["run_gateway"]["kwargs"]["eager_spin"] is False


def test_up_no_ccr_skips_write_config(wired):
    result = runner.invoke(app, ["up", "org/repo", "--no-ccr"])
    assert result.exit_code == 0, result.output
    assert "ccr" not in wired
    # gateway still starts
    assert "run_gateway" in wired


def test_up_profile_name_resolves_model_via_profiles_get(wired, monkeypatch):
    fake_profile = SimpleNamespace(
        name="coder-7b",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        provider="vast",
        quant="bf16",
        max_price=0.6,
        ttl_hours=None,
        context=None,
        concurrency=4,
        tool_call_parser=None,
        extra_vllm_args=[],
    )
    monkeypatch.setattr(
        profiles, "get", lambda name: fake_profile if name == "coder-7b" else None
    )
    result = runner.invoke(app, ["up", "coder-7b"])
    assert result.exit_code == 0, result.output
    assert wired["ccr"][2] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert wired["run_gateway"]["spin_kwargs"]["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert wired["run_gateway"]["spin_kwargs"]["quant"] == "bf16"


def test_up_garbage_link_exits_nonzero(wired):
    result = runner.invoke(app, ["up", "this is not valid"])
    assert result.exit_code != 0
    assert "run_gateway" not in wired
