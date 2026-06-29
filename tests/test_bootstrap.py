from aiod.bootstrap import ServerConfig


def _cfg(**kw):
    base = dict(repo_id="org/model", num_gpus=2, quant="bf16", api_key="sk-tok")
    base.update(kw)
    return ServerConfig(**base)


def test_server_args_core_flags():
    args = _cfg().server_args()
    assert "--model" in args and "org/model" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--api-key") + 1] == "sk-tok"
    # Claude Code needs tool calling.
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"


def test_bf16_has_no_quant_flag():
    assert "--quantization" not in _cfg(quant="bf16").server_args()


def test_fp8_and_awq_quant_flags():
    assert _cfg(quant="fp8").server_args()[
        _cfg(quant="fp8").server_args().index("--quantization") + 1
    ] == "fp8"
    awq = _cfg(quant="awq-int4").server_args()
    assert awq[awq.index("--quantization") + 1] == "awq"


def test_hf_token_in_env():
    env = _cfg(hf_token="hf_abc").env()
    assert env["HF_TOKEN"] == "hf_abc"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_abc"
    assert _cfg().env() == {}


def test_max_model_len_optional():
    assert "--max-model-len" not in _cfg().server_args()
    args = _cfg(max_model_len=8192).server_args()
    assert args[args.index("--max-model-len") + 1] == "8192"


def test_default_images_per_engine():
    from aiod.bootstrap import LLAMACPP_IMAGE, VLLM_IMAGE

    assert _cfg().image == VLLM_IMAGE
    assert _cfg(engine="llamacpp", gguf_quant="UD-IQ1_M").image == LLAMACPP_IMAGE


def test_llamacpp_args():
    args = _cfg(engine="llamacpp", gguf_quant="UD-Q3_K_M", repo_id="org/m-GGUF").server_args()
    assert args[args.index("-hf") + 1] == "org/m-GGUF:UD-Q3_K_M"  # repo:quant for shards
    assert args[args.index("-ngl") + 1] == "999"  # all layers across GPUs
    assert "--jinja" in args  # tool calling
    assert args[args.index("--port") + 1] == "8000"  # same published port as vLLM
    # no vLLM-only flags leak in
    assert "--tensor-parallel-size" not in args
    assert "--enable-auto-tool-choice" not in args


def test_llamacpp_args_without_quant_tag():
    args = _cfg(engine="llamacpp", repo_id="org/m-GGUF").server_args()
    assert args[args.index("-hf") + 1] == "org/m-GGUF"  # no ":quant" when unset
