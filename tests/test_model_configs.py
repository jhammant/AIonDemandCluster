from aiod import model_configs


def test_resolve_known_families():
    assert model_configs.resolve("Qwen/Qwen2.5-Coder-7B-Instruct").tool_call_parser == "hermes"
    assert model_configs.resolve("Qwen/Qwen2.5-7B-Instruct").tool_call_parser == "hermes"
    # -coder specials must win over the generic family match (registry order).
    assert model_configs.resolve("Qwen/Qwen3-Coder-30B-A3B").tool_call_parser == "qwen3_coder"
    assert model_configs.resolve("Qwen/Qwen3-8B").tool_call_parser == "hermes"
    assert model_configs.resolve("meta-llama/Llama-3.1-70B-Instruct").tool_call_parser == "llama3_json"
    assert model_configs.resolve("mistralai/Mistral-Nemo-Instruct").tool_call_parser == "mistral"
    assert model_configs.resolve("deepseek-ai/DeepSeek-V3").tool_call_parser == "deepseek_v3"
    assert model_configs.resolve("zai-org/GLM-4.6").tool_call_parser == "glm45"


def test_coder_gets_chat_template():
    # Qwen2.5-Coder gets the hermes scaffolding template (in-image path).
    mc = model_configs.resolve("Qwen/Qwen2.5-Coder-32B-Instruct")
    assert "--chat-template" in mc.vllm_serving_args()
    assert "/vllm-workspace/examples/" in mc.vllm_serving_args()[-1]


def test_resolve_default():
    assert model_configs.resolve("some/unknown-model").tool_call_parser == "hermes"


def test_chat_template_becomes_arg():
    mc = model_configs.ModelConfig(tool_call_parser="hermes", chat_template="/x/tpl.jinja")
    args = mc.vllm_serving_args()
    assert args[args.index("--chat-template") + 1] == "/x/tpl.jinja"


def test_no_chat_template_no_arg():
    assert model_configs.ModelConfig(tool_call_parser="hermes").vllm_serving_args() == []
