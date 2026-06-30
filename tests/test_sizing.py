from aiod.sizing import (
    ModelSpec,
    _params_from_name,
    auto_context_for_offer,
    context_for_vram,
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


def test_context_for_vram_scales_with_kv_quant():
    # q4_0 KV is ~0.27x f16, so the same VRAM buys ~3.7x the tokens.
    f16 = context_for_vram(16.0, None)
    q4 = context_for_vram(16.0, "q4_0")
    assert f16 == 16384  # 16 GB / 8 GB-per-8k * 8192
    assert q4 > f16 * 3
    assert context_for_vram(0.0) == 0


def test_auto_context_fills_offer_vram():
    # Tight box: weights 342.7 GB on a 392 GB box leaves little for f16 KV,
    # but q4_0 stretches it well past the f16 result.
    f16 = auto_context_for_offer(392.0, 342.7, None, num_gpus=4)
    q4 = auto_context_for_offer(392.0, 342.7, "q4_0", num_gpus=4)
    assert q4 > f16 > 0
    assert f16 % 4096 == 0 and q4 % 4096 == 0  # rounded to multiple


def test_auto_context_headroom_scales_with_gpu_count():
    # The working-VRAM buffer is per-device, so more GPUs reserve more and
    # leave less for context on an otherwise identical budget.
    few = auto_context_for_offer(360.0, 300.0, None, num_gpus=1)
    many = auto_context_for_offer(360.0, 300.0, None, num_gpus=8)
    assert few > many


def test_auto_context_small_model_not_starved_by_buffer():
    # A 7B-ish Q4 (~4.6 GB) on a single 24 GB card should get a real window,
    # not get floored because a fat flat buffer ate the spare VRAM.
    assert auto_context_for_offer(24.0, 4.6, None, num_gpus=1) > 8192


def test_auto_context_clamps_to_cap_and_floor():
    # Huge box -> cap; box that can't even hold weights -> floor.
    assert auto_context_for_offer(2000.0, 342.7, "q4_0", num_gpus=4) == 262144
    assert auto_context_for_offer(350.0, 342.7, None, num_gpus=4) == 8192
