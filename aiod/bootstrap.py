"""Builds the serving configuration that runs on the rented box: the Docker
image, the OpenAI-API server command, environment, and the published port. The
provider clients (vast/runpod) consume this when creating the instance.

Two engines:
  * vllm     — safetensors / AWQ / GPTQ / fp8 models (image vllm/vllm-openai)
  * llamacpp — GGUF models (image ghcr.io/ggml-org/llama.cpp:server-cuda); loads
               the GGUF straight from HF with -hf repo:QUANT (multi-part shards
               auto-detected), spreads across all GPUs with -ngl 999, and enables
               tool calling with --jinja. We pass --port 8000 so the published
               port is identical to vLLM's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VLLM_IMAGE = "vllm/vllm-openai:latest"
LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"

# Port the server listens on inside the container (published to a public port).
CONTAINER_PORT = 8000

# Map our quant keys to vLLM's --quantization values. bf16 = no flag.
VLLM_QUANT_FLAG = {
    "bf16": None,
    "fp16": None,
    "fp8": "fp8",
    "awq-int4": "awq",
    "gptq-int4": "gptq",
}


@dataclass
class ServerConfig:
    repo_id: str
    num_gpus: int
    quant: str
    api_key: str
    engine: str = "vllm"  # "vllm" | "llamacpp"
    image: str | None = None  # defaults per engine
    port: int = CONTAINER_PORT
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.92
    tool_call_parser: str = "hermes"
    extra_args: list[str] = field(default_factory=list)
    optimizations: list[str] = field(default_factory=list)  # resolved opt keys
    opt_values: dict[str, str] = field(default_factory=dict)  # {key: value} for value-bearing opts
    hf_token: str | None = None
    gguf_quant: str | None = None  # llamacpp: the GGUF quant tag, e.g. "UD-IQ1_M"

    def __post_init__(self) -> None:
        if self.image is None:
            self.image = LLAMACPP_IMAGE if self.engine == "llamacpp" else VLLM_IMAGE

    def server_args(self) -> list[str]:
        if self.engine == "llamacpp":
            return self._llamacpp_args()
        return self._vllm_args()

    def _vllm_args(self) -> list[str]:
        args = [
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--model", self.repo_id,
            "--served-model-name", self.repo_id,
            "--api-key", self.api_key,
            "--tensor-parallel-size", str(self.num_gpus),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            # Claude Code is extremely tool-call heavy; enable robust tool parsing.
            "--enable-auto-tool-choice",
            "--tool-call-parser", self.tool_call_parser,
        ]
        flag = VLLM_QUANT_FLAG.get(self.quant)
        if flag:
            args += ["--quantization", flag]
        if self.max_model_len:
            args += ["--max-model-len", str(self.max_model_len)]
        # Optimization flags go JUST BEFORE extra_args so extra_args still wins.
        # Empty list => byte-identical argv to before.
        if self.optimizations:
            from . import optimizations
            ctx = optimizations.OptContext(
                engine=self.engine, quant=self.quant, repo_id=self.repo_id
            )
            args += optimizations.vllm_flags(self.optimizations, self.opt_values, ctx)
        args += self.extra_args
        return args

    def _llamacpp_args(self) -> list[str]:
        # -hf repo[:quant]; the tag is a case-insensitive substring match against
        # filenames (works across subfolders) and auto-downloads all shards.
        ref = f"{self.repo_id}:{self.gguf_quant}" if self.gguf_quant else self.repo_id
        args = [
            "-hf", ref,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--api-key", self.api_key,
            "-ngl", "999",          # all layers on GPU; default layer-split spreads them
            "--jinja",              # enable OpenAI tool/function calling
            "-c", str(self.max_model_len or 32768),
        ]
        if self.optimizations:
            from . import optimizations
            ctx = optimizations.OptContext(
                engine=self.engine, quant=self.quant, repo_id=self.repo_id
            )
            args += optimizations.llamacpp_flags(self.optimizations, self.opt_values, ctx)
        args += self.extra_args
        return args

    def env(self) -> dict[str, str]:
        e: dict[str, str] = {}
        if self.hf_token:
            # vLLM/huggingface_hub read both; llama.cpp reads HF_TOKEN.
            e["HF_TOKEN"] = self.hf_token
            e["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        return e
