"""Sizing engine.

Given a HuggingFace model (link or `org/name`), figure out:
  1. how big it is (parameters, dtype, KV-cache shape, context length),
  2. how much VRAM it needs under each quantization option, and
  3. what GPU configuration (count x tier) can host it.

The output feeds `vast.py`, which turns the GPU plan into a real offer search
and a live $/hr cost estimate.

This module is deliberately dependency-light (just httpx) and has no vast.ai or
vLLM coupling, so it is easy to unit-test offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/config.json"

# Bytes per parameter for the weights under each quantization scheme.
# (Quantized formats still keep some tensors in higher precision; the small
# fudge is folded into OVERHEAD below rather than here.)
QUANT_BYTES: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "awq-int4": 0.5,
    "gptq-int4": 0.5,
}

# Human-friendly labels for the quant options we surface.
QUANT_LABELS: dict[str, str] = {
    "bf16": "bf16/fp16 (full precision)",
    "fp8": "FP8 (~half the VRAM, near-lossless)",
    "awq-int4": "AWQ int4 (~quarter VRAM, small quality cost)",
    "gptq-int4": "GPTQ int4 (~quarter VRAM, small quality cost)",
}

# CUDA context + activations + framework overhead, as a multiplier on weights,
# plus a fixed per-GPU reservation. Conservative so estimates don't under-shoot.
OVERHEAD_MULT = 1.20
FIXED_GB_PER_GPU = 2.0

# KV-cache cache dtype size in bytes (vLLM default fp16; fp8 kv-cache halves it).
KV_BYTES = 2.0


@dataclass
class GpuTier:
    """A GPU model we know how to plan around. `vram_gb` is per-GPU."""

    name: str
    vram_gb: int
    # Substrings that vast.ai uses in `gpu_name` for matching this tier.
    aliases: list[str] = field(default_factory=list)


# Ordered cheapest-capability first. vast.ai `gpu_name` examples in aliases.
GPU_TIERS: list[GpuTier] = [
    GpuTier("RTX 4090", 24, ["RTX 4090", "RTX_4090"]),
    GpuTier("RTX 5090", 32, ["RTX 5090", "RTX_5090"]),
    GpuTier("RTX 6000 Ada / A6000", 48, ["RTX 6000Ada", "A6000", "RTX A6000"]),
    GpuTier("A100 80GB", 80, ["A100 SXM4 80GB", "A100 PCIE 80GB", "A100_SXM4_80GB", "A100"]),
    GpuTier("H100 80GB", 80, ["H100 SXM", "H100 PCIE", "H100_SXM", "H100"]),
    GpuTier("H100 NVL 94GB", 94, ["H100 NVL", "H100_NVL"]),
    GpuTier("H200 141GB", 141, ["H200"]),
    GpuTier("B200 180GB", 180, ["B200"]),
]

# vLLM tensor-parallel sizes that are safe to assume (power of two).
VALID_TP = [1, 2, 4, 8]

# Param-count fallback parsed from the model id, e.g. "...-32B-...", "7b", "0.5B".
_PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


class ModelNotFound(Exception):
    pass


@dataclass
class ModelSpec:
    repo_id: str
    params: int  # total parameter count
    dtype: str  # native weight dtype reported by HF, e.g. "BF16"
    num_layers: int | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_kv_heads: int | None
    max_context: int | None
    architecture: str | None
    gated: bool
    params_source: str  # "safetensors" | "name-heuristic"

    @property
    def params_b(self) -> float:
        return self.params / 1e9


@dataclass
class GpuOption:
    """One concrete way to host the model: N GPUs of a given tier."""

    tier: GpuTier
    num_gpus: int
    total_vram_gb: int
    fits: bool  # does required VRAM fit in total_vram with headroom?
    headroom_gb: float


@dataclass
class QuantPlan:
    quant: str
    label: str
    weights_gb: float
    kv_gb: float
    required_vram_gb: float
    options: list[GpuOption]

    @property
    def cheapest_fit(self) -> GpuOption | None:
        fits = [o for o in self.options if o.fits]
        if not fits:
            return None
        # Fewest GPUs, then smallest tier.
        return min(fits, key=lambda o: (o.num_gpus, o.tier.vram_gb))


@dataclass
class SizingResult:
    model: ModelSpec
    context_tokens: int  # context length x concurrency used for the KV estimate
    plans: list[QuantPlan]
    engine: str = "vllm"  # "vllm" (safetensors) | "llamacpp" (GGUF)

    def plan(self, quant: str) -> QuantPlan | None:
        return next((p for p in self.plans if p.quant == quant), None)


# --------------------------------------------------------------------------- #
# Parsing & metadata
# --------------------------------------------------------------------------- #

def parse_repo_id(model: str) -> str:
    """Accept a full HF URL or a bare `org/name` and return `org/name`."""
    model = model.strip()
    if model.startswith("http://") or model.startswith("https://"):
        path = urlparse(model).path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        raise ValueError(f"Could not parse a repo id from URL: {model}")
    return model


def _params_from_name(repo_id: str) -> int | None:
    m = _PARAM_RE.search(repo_id)
    if not m:
        return None
    return int(float(m.group(1)) * 1e9)


def fetch_model_spec(model: str, hf_token: str | None = None, timeout: float = 20.0) -> ModelSpec:
    """Look up a model on the HuggingFace Hub and build a ModelSpec."""
    repo_id = parse_repo_id(model)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        r = client.get(f"{HF_API}/{repo_id}")
        if r.status_code == 404:
            raise ModelNotFound(f"Model not found on HuggingFace: {repo_id}")
        if r.status_code == 401 or r.status_code == 403:
            raise ModelNotFound(
                f"Model {repo_id} is gated/private. Set HF_TOKEN (and accept the "
                f"license on the model page)."
            )
        r.raise_for_status()
        info = r.json()

        # Config holds the architecture-level shape used for the KV-cache math.
        config: dict = info.get("config", {}) or {}
        if not config or "num_hidden_layers" not in config:
            cr = client.get(HF_RESOLVE.format(repo=repo_id))
            if cr.status_code == 200:
                try:
                    config = {**config, **cr.json()}
                except Exception:
                    pass

    safetensors = info.get("safetensors") or {}
    params = safetensors.get("total")
    params_source = "safetensors"
    dtype = "BF16"
    if isinstance(safetensors.get("parameters"), dict) and safetensors["parameters"]:
        dtype = max(safetensors["parameters"], key=safetensors["parameters"].get)
    if not params:
        params = _params_from_name(repo_id)
        params_source = "name-heuristic"
    if not params:
        raise ModelNotFound(
            f"Could not determine parameter count for {repo_id} (no safetensors "
            f"metadata and no size in the name). Pass --params manually."
        )

    n_heads = config.get("num_attention_heads")
    return ModelSpec(
        repo_id=repo_id,
        params=int(params),
        dtype=dtype,
        num_layers=config.get("num_hidden_layers"),
        hidden_size=config.get("hidden_size"),
        num_attention_heads=n_heads,
        num_kv_heads=config.get("num_key_value_heads", n_heads),
        max_context=config.get("max_position_embeddings"),
        architecture=(info.get("config", {}).get("model_type") or (info.get("tags") and None)),
        gated=bool(info.get("gated")),
        params_source=params_source,
    )


# --------------------------------------------------------------------------- #
# VRAM math & GPU planning
# --------------------------------------------------------------------------- #

def _kv_bytes_per_token(spec: ModelSpec) -> float | None:
    """KV-cache bytes for a single token across all layers."""
    if not (spec.num_layers and spec.hidden_size and spec.num_attention_heads):
        return None
    head_dim = spec.hidden_size / spec.num_attention_heads
    kv_heads = spec.num_kv_heads or spec.num_attention_heads
    # 2 = one K and one V tensor.
    return 2 * spec.num_layers * kv_heads * head_dim * KV_BYTES


def estimate_vram(
    spec: ModelSpec,
    quant: str,
    context_tokens: int,
) -> tuple[float, float, float]:
    """Return (weights_gb, kv_gb, required_total_gb) for a quant scheme."""
    bytes_per_param = QUANT_BYTES[quant]
    weights_gb = spec.params * bytes_per_param / 1e9

    kv_per_tok = _kv_bytes_per_token(spec)
    if kv_per_tok is not None:
        kv_gb = kv_per_tok * context_tokens / 1e9
    else:
        # No config shape available — assume KV ~ 15% of weights as a rough floor.
        kv_gb = weights_gb * 0.15

    required = weights_gb * OVERHEAD_MULT + kv_gb + FIXED_GB_PER_GPU
    return weights_gb, kv_gb, required


def plan_gpus(required_vram_gb: float, safety: float = 0.90) -> list[GpuOption]:
    """For each known tier, find the smallest valid TP size that fits."""
    options: list[GpuOption] = []
    for tier in GPU_TIERS:
        chosen: GpuOption | None = None
        for n in VALID_TP:
            total = n * tier.vram_gb
            usable = total * safety
            if usable >= required_vram_gb:
                chosen = GpuOption(
                    tier=tier,
                    num_gpus=n,
                    total_vram_gb=total,
                    fits=True,
                    headroom_gb=usable - required_vram_gb,
                )
                break
        if chosen is None:
            # Even 8x doesn't fit — record the max config as a non-fit.
            total = VALID_TP[-1] * tier.vram_gb
            chosen = GpuOption(
                tier=tier,
                num_gpus=VALID_TP[-1],
                total_vram_gb=total,
                fits=False,
                headroom_gb=total * safety - required_vram_gb,
            )
        options.append(chosen)
    return options


def size_model(
    model: str,
    hf_token: str | None = None,
    quants: list[str] | None = None,
    context_len: int | None = None,
    concurrency: int = 4,
    max_context_for_estimate: int = 32768,
) -> SizingResult:
    """End-to-end: HF link -> sizing across quant options."""
    spec = fetch_model_spec(model, hf_token=hf_token)

    ctx = context_len or spec.max_context or 8192
    ctx = min(ctx, max_context_for_estimate)
    context_tokens = ctx * max(1, concurrency)

    quants = quants or ["bf16", "fp8", "awq-int4"]
    plans: list[QuantPlan] = []
    for q in quants:
        weights_gb, kv_gb, required = estimate_vram(spec, q, context_tokens)
        plans.append(
            QuantPlan(
                quant=q,
                label=QUANT_LABELS.get(q, q),
                weights_gb=weights_gb,
                kv_gb=kv_gb,
                required_vram_gb=required,
                options=plan_gpus(required),
            )
        )

    return SizingResult(model=spec, context_tokens=context_tokens, plans=plans)


# --------------------------------------------------------------------------- #
# GGUF (llama.cpp engine) — size by file, not by params
# --------------------------------------------------------------------------- #

HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"

# Matches quant tokens in GGUF paths: Q4_K_M, IQ1_M, UD-IQ2_XXS, BF16, F16, MXFP4…
_GGUF_QUANT_RE = re.compile(
    r"((?:UD-)?(?:IQ\d+[A-Z0-9_]*|Q\d+[A-Z0-9_]*|BF16|F16|F32|MXFP4))", re.IGNORECASE
)


@dataclass
class GgufQuant:
    name: str
    size_gb: float
    n_files: int
    sample_path: str


def _gguf_quant_from_path(path: str) -> str:
    parts = path.split("/")
    # GGUF repos usually put each quant in its own folder (e.g. UD-IQ1_M/...).
    if len(parts) > 1 and _GGUF_QUANT_RE.search(parts[-2]):
        return parts[-2]
    m = _GGUF_QUANT_RE.search(parts[-1])
    if m:
        return m.group(1)
    return parts[-2] if len(parts) > 1 else parts[-1]


def fetch_gguf_quants(repo: str, hf_token: str | None = None, timeout: float = 30.0) -> dict[str, GgufQuant]:
    """List the GGUF quant builds in a repo with their (multi-part) total sizes."""
    repo = parse_repo_id(repo)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        r = client.get(HF_TREE.format(repo=repo))
        if r.status_code == 404:
            raise ModelNotFound(f"Repo not found: {repo}")
        r.raise_for_status()
        tree = r.json()

    acc: dict[str, list] = {}
    for f in tree:
        path = f.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        quant = _gguf_quant_from_path(path)
        entry = acc.setdefault(quant, [0, 0, path])
        entry[0] += size
        entry[1] += 1
    return {
        q: GgufQuant(name=q, size_gb=s / 1e9, n_files=n, sample_path=p)
        for q, (s, n, p) in acc.items()
    }


def detect_format(repo: str, hf_token: str | None = None, timeout: float = 20.0) -> str:
    """Return 'safetensors', 'gguf', or 'unknown' for a repo."""
    repo = parse_repo_id(repo)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        r = client.get(f"{HF_API}/{repo}")
        if r.status_code >= 400:
            return "unknown"
        info = r.json()
    sibs = [s.get("rfilename", "") for s in info.get("siblings", [])]
    if info.get("safetensors") or any(s.endswith(".safetensors") for s in sibs):
        return "safetensors"
    if any(s.lower().endswith(".gguf") for s in sibs):
        return "gguf"
    return "unknown"


def size_gguf_model(
    repo: str,
    hf_token: str | None = None,
    context_len: int | None = None,
) -> SizingResult:
    """Size every GGUF quant in a repo by file size and produce a GPU plan each."""
    repo = parse_repo_id(repo)
    quants = fetch_gguf_quants(repo, hf_token=hf_token)
    if not quants:
        raise ModelNotFound(f"No .gguf files found in {repo}")

    ctx = context_len or 8192
    # llama.cpp KV cache scales with context; rough budget (no per-model config here).
    kv_gb = max(4.0, (ctx / 8192.0) * 8.0)

    plans: list[QuantPlan] = []
    for q in sorted(quants.values(), key=lambda x: x.size_gb):
        # llama.cpp packs weights tightly (no CUDA-graph blow-up like vLLM), so the
        # weight overhead is small and we can use a tighter GPU-packing safety margin.
        required = q.size_gb * 1.02 + kv_gb
        plans.append(
            QuantPlan(
                quant=q.name,
                label=f"GGUF {q.name} ({q.size_gb:.0f}GB, {q.n_files} file{'s' if q.n_files > 1 else ''})",
                weights_gb=q.size_gb,
                kv_gb=kv_gb,
                required_vram_gb=required,
                options=plan_gpus(required, safety=0.97),
            )
        )

    spec = ModelSpec(
        repo_id=repo,
        params=0,
        dtype="GGUF",
        num_layers=None,
        hidden_size=None,
        num_attention_heads=None,
        num_kv_heads=None,
        max_context=None,
        architecture="gguf",
        gated=False,
        params_source="gguf",
    )
    return SizingResult(model=spec, context_tokens=ctx, plans=plans, engine="llamacpp")


def size_any(
    repo: str,
    engine: str = "auto",
    hf_token: str | None = None,
    quants: list[str] | None = None,
    context_len: int | None = None,
    concurrency: int = 4,
) -> SizingResult:
    """Engine-aware entry point: auto-detect GGUF vs safetensors (or force via
    `engine`) and return the matching SizingResult."""
    repo = parse_repo_id(repo)
    eng = engine
    if engine == "auto":
        eng = "llamacpp" if detect_format(repo, hf_token=hf_token) == "gguf" else "vllm"
    if eng == "llamacpp":
        return size_gguf_model(repo, hf_token=hf_token, context_len=context_len)
    return size_model(
        repo, hf_token=hf_token, quants=quants, context_len=context_len, concurrency=concurrency
    )
