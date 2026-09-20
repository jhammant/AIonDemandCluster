from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aiod import profiles
from aiod.cli import app
from aiod.sizing import ModelSpec, QuantPlan, SizingResult

runner = CliRunner()


@pytest.fixture
def sized(monkeypatch):
    """Let `spin` reach the engine check with a sizing result we control."""
    fake_settings = SimpleNamespace(
        vllm_api_key="sk-aiod-test123", hf_token=None, ttl_hours=2.0, max_price=1.0,
        vast_api_key="fake", runpod_api_key="",
    )
    monkeypatch.setattr("aiod.cli.Settings.load", staticmethod(lambda: fake_settings))
    monkeypatch.setattr("aiod.cli._require_provider_key", lambda s, provider: None)
    monkeypatch.setattr(profiles, "get", lambda name: None)

    def fake_size_any(engine):
        spec = ModelSpec(
            repo_id="org/model", params=7_000_000_000, dtype="BF16", num_layers=32,
            hidden_size=4096, num_attention_heads=32, num_kv_heads=8, max_context=8192,
            architecture="llama", gated=False, params_source="safetensors",
        )
        plan = QuantPlan(
            quant="bf16", label="BF16", weights_gb=14.0, kv_gb=2.0,
            required_vram_gb=20.0, options=[],
        )
        return SizingResult(model=spec, context_tokens=8192, plans=[plan], engine=engine)

    return fake_size_any


def test_kv_quant_on_a_vllm_model_is_refused(monkeypatch, sized):
    """--cache-type-k/v are llama.cpp flags; vLLM would refuse to start on them."""
    monkeypatch.setattr("aiod.cli.size_any", lambda *a, **k: sized("vllm"))
    r = runner.invoke(app, ["spin", "org/model", "--kv-quant", "q8_0", "--yes"])
    assert r.exit_code == 1
    assert "--kv-quant only applies to GGUF" in r.output
    assert "vllm" in r.output
