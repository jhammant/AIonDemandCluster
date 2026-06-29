"""Per-model-family serving config registry.

Different model families need different vLLM tool-calling settings
(`--tool-call-parser`, sometimes `--chat-template`) for the OpenAI `tools` API to
parse into structured `tool_calls`. Claude Code is tool-call-heavy, so getting
this right per model is what makes a model actually usable.

This registry maps a model-name pattern to a `ModelConfig`. It's applied
automatically on spin; a profile or CLI flag can still override. It's a plain
data table on purpose — add a verified config for a new model with a one-line
entry (and a PR), rather than hand-tuning flags each time.

Layering (later wins):  registry default  <  this registry match  <  profile  <  CLI flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    tool_call_parser: str = "hermes"  # vLLM --tool-call-parser
    chat_template: str | None = None  # path (in image) or URL for --chat-template
    extra_args: list[str] = field(default_factory=list)  # extra vLLM flags
    recommended_quant: str | None = None  # nudge (not forced)
    notes: str = ""

    def vllm_serving_args(self) -> list[str]:
        """Tool-calling-related vLLM args this config contributes."""
        args = list(self.extra_args)
        if self.chat_template:
            args += ["--chat-template", self.chat_template]
        return args


# vLLM ships these tool chat templates inside the vllm/vllm-openai image
# (Dockerfile `COPY examples examples` -> /vllm-workspace/examples).
_TPL = "/vllm-workspace/examples/tool_chat_template_{}.jinja"

# Default when nothing matches. Hermes is the most broadly compatible parser.
DEFAULT = ModelConfig(tool_call_parser="hermes", notes="generic default")

# Ordered: first matching pattern wins (so put '-coder' / version specials BEFORE
# the generic family). Patterns are case-insensitive substrings of the HF repo id.
# Parsers/templates verified against vLLM tool-calling docs (2026). NOTE: the very
# newest parsers (glm45, qwen3_xml, deepseek_v31) require a recent image — vLLM
# `:latest` should have them; pin the image if you need determinism.
_REGISTRY: list[tuple[re.Pattern, ModelConfig]] = [
    # --- Qwen ---
    (re.compile(r"qwen-?3.*coder", re.I),
     ModelConfig(tool_call_parser="qwen3_coder", notes="Qwen3-Coder (XML tool format)")),
    (re.compile(r"qwen-?3", re.I),
     ModelConfig(tool_call_parser="hermes", notes="Qwen3 — strong native tool calling")),
    # Qwen2.5-Coder has NO native tool format (vLLM #32926); the hermes scaffolding
    # template is the best in-image attempt but is still unreliable — prefer
    # Qwen2.5-Instruct (non-Coder) or Qwen3 for tool-heavy use like Claude Code.
    (re.compile(r"qwen-?2\.?5.*coder", re.I),
     ModelConfig(tool_call_parser="hermes", chat_template=_TPL.format("hermes"),
                 notes="Qwen2.5-Coder: weak tool calling (best-effort hermes template)")),
    (re.compile(r"qwen-?2\.?5|qwq", re.I),
     ModelConfig(tool_call_parser="hermes", notes="Qwen2.5-Instruct / QwQ — native hermes")),
    # --- Llama ---
    (re.compile(r"llama-?4", re.I),
     ModelConfig(tool_call_parser="llama4_pythonic", notes="Llama 4")),
    (re.compile(r"llama-?3", re.I),
     ModelConfig(tool_call_parser="llama3_json", chat_template=_TPL.format("llama3.1_json"),
                 notes="Llama 3.x (no parallel tool calls)")),
    # --- Mistral ---
    (re.compile(r"mistral|mixtral|ministral|magistral", re.I),
     ModelConfig(tool_call_parser="mistral", chat_template=_TPL.format("mistral_parallel"),
                 notes="Mistral / Nemo")),
    # --- DeepSeek ---
    (re.compile(r"deepseek.*v3\.?1", re.I),
     ModelConfig(tool_call_parser="deepseek_v31", notes="DeepSeek-V3.1")),
    (re.compile(r"deepseek", re.I),
     ModelConfig(tool_call_parser="deepseek_v3", notes="DeepSeek-V3 / R1")),
    # --- GLM ---
    (re.compile(r"glm-?4\.?[567]", re.I),
     ModelConfig(tool_call_parser="glm45", notes="GLM-4.5/4.6/4.7 (native)")),
    (re.compile(r"glm-?4", re.I),
     ModelConfig(tool_call_parser="hermes", notes="GLM-4 (older)")),
    # --- misc ---
    (re.compile(r"kimi", re.I), ModelConfig(tool_call_parser="kimi_k2", notes="Kimi-K2")),
    (re.compile(r"hermes|nous", re.I), ModelConfig(tool_call_parser="hermes")),
]


def resolve(model_id: str) -> ModelConfig:
    """Return the serving config for a model id (or the default)."""
    for pattern, cfg in _REGISTRY:
        if pattern.search(model_id):
            return cfg
    return DEFAULT
