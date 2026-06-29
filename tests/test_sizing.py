from aiod.sizing import (
    ModelSpec,
    _params_from_name,
    estimate_vram,
    parse_repo_id,
    plan_gpus,
)


def _spec(params: int) -> ModelSpec:
    # A Llama-70B-ish shape for KV math.
    return ModelSpec(
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


def test_parse_repo_id_url_and_bare():
    assert parse_repo_id("https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct") == (
        "Qwen/Qwen2.5-Coder-32B-Instruct"
    )
    assert parse_repo_id("Qwen/Qwen2.5-Coder-32B-Instruct") == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert parse_repo_id("  org/name  ") == "org/name"


def test_params_from_name():
    assert _params_from_name("meta/Llama-3.1-70B-Instruct") == 70_000_000_000
    assert _params_from_name("Qwen/Qwen2.5-0.5B") == 500_000_000
    assert _params_from_name("org/no-size-here") is None


def test_quant_reduces_vram_monotonically():
    spec = _spec(70_000_000_000)
    _, _, bf16 = estimate_vram(spec, "bf16", context_tokens=8192)
    _, _, fp8 = estimate_vram(spec, "fp8", context_tokens=8192)
    _, _, int4 = estimate_vram(spec, "awq-int4", context_tokens=8192)
    assert bf16 > fp8 > int4


def test_plan_gpus_picks_power_of_two_and_fits():
    # ~140GB requirement should fit on 2x 80GB, not 1x.
    opts = {o.tier.name: o for o in plan_gpus(140.0)}
    a100 = opts["A100 80GB"]
    assert a100.num_gpus == 2
    assert a100.fits
    assert a100.num_gpus in (1, 2, 4, 8)


def test_gguf_quant_from_path():
    from aiod.sizing import _gguf_quant_from_path

    assert _gguf_quant_from_path("UD-IQ1_M/GLM-5.2-UD-IQ1_M-00001-of-00005.gguf") == "UD-IQ1_M"
    assert _gguf_quant_from_path("model.Q4_K_M.gguf") == "Q4_K_M"
    assert _gguf_quant_from_path("foo/model-Q3_K_M.gguf") == "Q3_K_M"


def test_small_model_fits_one_gpu():
    spec = _spec(7_000_000_000)
    _, _, req = estimate_vram(spec, "bf16", context_tokens=8192)
    opts = {o.tier.name: o for o in plan_gpus(req)}
    assert opts["A100 80GB"].num_gpus == 1
