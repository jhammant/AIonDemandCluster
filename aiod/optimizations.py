"""Pluggable serving/sizing optimizations.

An *optimization* is a named toggle (optionally value-bearing) that can do up to
three things, each on its own seam:

  (a) append vLLM / llama.cpp **serving flags** (just before ``extra_args``, so
      ``extra_args`` still wins) — see :meth:`Optimization.vllm_args` /
      :meth:`Optimization.llamacpp_args`;
  (b) scale the **VRAM sizing** estimate via multiplicative
      :class:`SizingKnobs` applied inside ``sizing.estimate_vram`` before GPU
      selection — see :meth:`Optimization.sizing`;
  (c) raise a **hardware floor** (compute-capability) via
      :class:`OptRequirements`, stamped onto the ``QuantPlan`` and folded into
      the vast offer search.

ONE source of truth: :func:`resolve` produces a single :class:`ResolvedOpts`
whose ``keys`` drive BOTH the serving argv (``ServerConfig.optimizations``) and
the sizing path (``estimate_vram`` knobs). Applicability is decided by
:meth:`Optimization.applies`, which — by contract — may read ONLY
``engine``/``quant``/``repo_id``/``value``/``draft_model`` from the context and
NEVER ``spec``. ``spec`` is populated only on the sizing path and may be read
only by :meth:`Optimization.sizing`. That contract is what closes the
serving(spec=None) vs sizing(spec=set) split-brain: ``applies`` returns the same
answer on both paths, so the same key set feeds argv and sizing.

Registry layering mirrors ``model_configs``/``profiles`` ("later wins"):
built-in  <  entry-point plugin (group ``aiod.optimizations``). Built-ins all
ship ``default_on=False`` so the no-opt path is byte-identical to before.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .sizing import ModelSpec

# importlib.metadata entry-point group third-party plugins register under.
ENTRYPOINT_GROUP = "aiod.optimizations"


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OptContext:
    """Everything an optimization may inspect.

    CONTRACT: ``applies``/``vllm_args``/``llamacpp_args`` may read
    ``engine``/``quant``/``repo_id``/``value``/``draft_model`` only. ``spec`` is
    populated ONLY on the sizing path and may be read ONLY by ``sizing``. This
    forbids spec-dependent applicability, which is what closes the
    serving(spec=None) vs sizing(spec=set) split-brain.
    """

    engine: str
    quant: str
    repo_id: str
    concurrency: int = 4
    spec: ModelSpec | None = None
    value: str | None = None
    draft_model: str | None = None


@dataclass
class SizingKnobs:
    """Multiplicative VRAM scalers (order-independent; defaults == today)."""

    kv_scale: float = 1.0
    weight_scale: float = 1.0


@dataclass(frozen=True)
class OptRequirements:
    """Hardware/runtime needs an optimization imposes."""

    min_compute_cap: int = 0
    needs_draft_model: bool = False


@runtime_checkable
class Optimization(Protocol):
    key: str
    summary: str
    tradeoff: str
    engines: tuple[str, ...]
    default_on: bool
    takes_value: bool
    conflicts_with: tuple[str, ...]
    provides: tuple[str, ...]

    def applies(self, ctx: OptContext) -> bool: ...
    def requirements(self, ctx: OptContext) -> OptRequirements: ...
    def vllm_args(self, ctx: OptContext) -> list[str]: ...
    def llamacpp_args(self, ctx: OptContext) -> list[str]: ...
    def sizing(self, knobs: SizingKnobs, ctx: OptContext) -> SizingKnobs: ...


@dataclass(frozen=True)
class BaseOpt:
    """No-op defaults so a plugin overrides only what it touches (mirrors
    ``ModelConfig``). Subclass and override the seams you actually use."""

    key: str
    summary: str
    tradeoff: str
    engines: tuple[str, ...] = ("vllm",)
    default_on: bool = False
    takes_value: bool = False
    conflicts_with: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def applies(self, ctx: OptContext) -> bool:
        return ctx.engine in self.engines

    def requirements(self, ctx: OptContext) -> OptRequirements:
        return OptRequirements()

    def vllm_args(self, ctx: OptContext) -> list[str]:
        return []

    def llamacpp_args(self, ctx: OptContext) -> list[str]:
        return []

    def sizing(self, knobs: SizingKnobs, ctx: OptContext) -> SizingKnobs:
        return knobs


@dataclass
class ResolvedOpts:
    """The single source of truth threaded to BOTH serving argv and sizing."""

    keys: list[str]
    knobs: SizingKnobs
    requirements: OptRequirements
    skipped: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Built-in optimizations
# --------------------------------------------------------------------------- #

class _PrefixCaching(BaseOpt):
    def vllm_args(self, ctx: OptContext) -> list[str]:
        return ["--enable-prefix-caching"]


class _ChunkedPrefill(BaseOpt):
    def vllm_args(self, ctx: OptContext) -> list[str]:
        args = ["--enable-chunked-prefill"]
        if ctx.value:
            args += ["--max-num-batched-tokens", str(ctx.value)]
        return args


class _KvCacheFp8(BaseOpt):
    def vllm_args(self, ctx: OptContext) -> list[str]:
        return ["--kv-cache-dtype", "fp8"]

    def requirements(self, ctx: OptContext) -> OptRequirements:
        # fp8 KV is only reliable on Ada+ (cc 8.9), same floor as fp8 weights.
        return OptRequirements(min_compute_cap=890)

    def sizing(self, knobs: SizingKnobs, ctx: OptContext) -> SizingKnobs:
        # Scale the OUTPUT kv_gb (not KV_BYTES) so it also halves the
        # config-less weights*0.15 fallback branch in estimate_vram.
        return replace(knobs, kv_scale=knobs.kv_scale * 0.5)


class _MaxNumSeqs(BaseOpt):
    def vllm_args(self, ctx: OptContext) -> list[str]:
        return ["--max-num-seqs", str(ctx.value or 256)]

    def sizing(self, knobs: SizingKnobs, ctx: OptContext) -> SizingKnobs:
        # Raise the KV estimate to match the higher concurrent-seq cap so the
        # sizer picks a bigger GPU instead of OOMing at runtime (never the
        # OOM direction).
        n = int(ctx.value or 256)
        factor = max(1.0, n / max(1, ctx.concurrency))
        return replace(knobs, kv_scale=knobs.kv_scale * factor)


class _SpeculativeDecoding(BaseOpt):
    def applies(self, ctx: OptContext) -> bool:
        # Inert until a draft model is supplied (real impl is future work).
        return ctx.engine in self.engines and ctx.draft_model is not None

    def requirements(self, ctx: OptContext) -> OptRequirements:
        return OptRequirements(needs_draft_model=True)

    def vllm_args(self, ctx: OptContext) -> list[str]:
        if not ctx.draft_model:
            return []
        return ["--speculative-config", f'{{"model": "{ctx.draft_model}"}}']


_BUILTINS: list[Optimization] = [
    _KvCacheFp8(
        key="kv-cache-fp8",
        summary="FP8 KV cache (~half the KV-cache VRAM)",
        tradeoff="Tiny accuracy cost; needs Ada+ (compute cap 8.9).",
        engines=("vllm",),
        provides=("kv-dtype",),
    ),
    _PrefixCaching(
        key="prefix-caching",
        summary="Reuse shared prompt prefixes across requests (APC)",
        tradeoff="None; recent vLLM already enables it — forces it on pinned images.",
        engines=("vllm",),
    ),
    _ChunkedPrefill(
        key="chunked-prefill",
        summary="Chunk long prefills for steadier latency",
        tradeoff="Marginal throughput trade for smoother TTFT.",
        engines=("vllm",),
        takes_value=True,
    ),
    _MaxNumSeqs(
        key="max-num-seqs",
        summary="Raise the concurrent-sequence cap (default 256)",
        tradeoff="More KV VRAM; the sizer grows the estimate to match.",
        engines=("vllm",),
        takes_value=True,
    ),
    _SpeculativeDecoding(
        key="speculative-decoding",
        summary="Speculative decoding (requires draft model)",
        tradeoff="Faster decode; needs a draft model (inert today).",
        engines=("vllm",),
        takes_value=True,
    ),
]

_BUILTIN_KEYS = {o.key for o in _BUILTINS}


# --------------------------------------------------------------------------- #
# Registry + entry-point loading
# --------------------------------------------------------------------------- #

def _iter_entry_points():
    """List entry points in our group (isolated so tests can monkeypatch it)."""
    from importlib.metadata import entry_points

    return list(entry_points(group=ENTRYPOINT_GROUP))


def _load_entrypoints() -> list[Optimization]:
    """Load plugin optimizations. Fail-SOFT but NOT silent: a broken plugin is
    skipped with a one-line stderr warning so a launch never breaks yet bugs are
    visible."""
    out: list[Optimization] = []
    try:
        eps = _iter_entry_points()
    except Exception as e:  # noqa: BLE001 - never let plugin discovery break a launch
        print(f"aiod: warning: could not enumerate optimization plugins: {e}", file=sys.stderr)
        return out
    for ep in eps:
        try:
            obj = ep.load()
            opt = obj() if isinstance(obj, type) else obj
            if not hasattr(opt, "key"):
                raise TypeError("loaded object is not an Optimization (no .key attribute)")
            out.append(opt)
        except Exception as e:  # noqa: BLE001 - fail-soft per plugin, but warn
            name = getattr(ep, "name", "?")
            print(
                f"aiod: warning: skipping optimization plugin '{name}': {e}",
                file=sys.stderr,
            )
    return out


def registry() -> dict[str, Optimization]:
    """Built-ins then entry-points, keyed by ``opt.key`` (later wins)."""
    reg: dict[str, Optimization] = {o.key: o for o in _BUILTINS}
    for o in _load_entrypoints():
        if o.key in _BUILTIN_KEYS:
            print(
                f"aiod: warning: plugin overrides built-in optimization '{o.key}' "
                f"(core keys are reserved)",
                file=sys.stderr,
            )
        reg[o.key] = o
    return reg


def enumerate_all() -> list[Optimization]:
    return list(registry().values())


def get(key: str) -> Optimization | None:
    return registry().get(key)


def default_selection() -> list[str]:
    return [o.key for o in enumerate_all() if o.default_on]


# --------------------------------------------------------------------------- #
# Selection parsing + resolution
# --------------------------------------------------------------------------- #

def parse_selection(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split ``['max-num-seqs=256', 'prefix-caching']`` into ordered keys plus a
    ``{key: value}`` map. Splits on the first ``=`` only."""
    keys: list[str] = []
    values: dict[str, str] = {}
    for tok in tokens:
        if tok is None:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
        else:
            k, v = tok, None
        k = k.strip()
        if not k:
            continue
        keys.append(k)
        if v is not None:
            values[k] = v
    return keys, values


def _check_conflicts(chosen: list[Optimization]) -> None:
    """Raise on a declared ``conflicts_with`` pair or a duplicate ``provides``."""
    keys = {o.key for o in chosen}
    provided: dict[str, str] = {}
    for o in chosen:
        for c in o.conflicts_with:
            if c in keys:
                raise ValueError(f"optimization '{o.key}' conflicts with '{c}'")
        for p in o.provides:
            if p in provided:
                raise ValueError(
                    f"optimizations '{provided[p]}' and '{o.key}' both provide '{p}'"
                )
            provided[p] = o.key


def resolve(selected: list[str], values: dict[str, str], ctx: OptContext) -> ResolvedOpts:
    """Fold the selected opts (in REGISTRY order) into one ResolvedOpts.

    Validates value usage and conflicts/provides, applies the ``applies`` gate
    (recording skipped (key, reason)), multiplies the sizing knobs, and takes the
    max over ``min_compute_cap``.
    """
    reg = registry()
    values = values or {}
    skipped: list[tuple[str, str]] = []

    # Reject a value on a toggle-only opt (value-form ambiguity guard).
    for k, _v in values.items():
        opt = reg.get(k)
        if opt is not None and not opt.takes_value:
            raise ValueError(f"optimization '{k}' does not take a value")

    selected_set = set(selected)
    for k in selected:
        if k not in reg:
            skipped.append((k, "unknown"))

    # REGISTRY order (not selection order) => reproducible argv & sizing.
    ordered = [reg[k] for k in reg if k in selected_set]
    _check_conflicts(ordered)

    keys: list[str] = []
    knobs = SizingKnobs()
    min_cc = 0
    needs_draft = False
    for opt in ordered:
        octx = replace(ctx, value=values.get(opt.key))
        if not opt.applies(octx):
            skipped.append((opt.key, "not applicable"))
            continue
        keys.append(opt.key)
        knobs = opt.sizing(knobs, octx)
        req = opt.requirements(octx)
        min_cc = max(min_cc, req.min_compute_cap)
        needs_draft = needs_draft or req.needs_draft_model

    return ResolvedOpts(
        keys=keys,
        knobs=knobs,
        requirements=OptRequirements(min_compute_cap=min_cc, needs_draft_model=needs_draft),
        skipped=skipped,
    )


def _engine_flags(
    keys: list[str], values: dict[str, str], ctx: OptContext, *, engine: str
) -> list[str]:
    reg = registry()
    values = values or {}
    selected = set(keys)
    out: list[str] = []
    for k in reg:  # REGISTRY order for reproducible argv
        if k not in selected:
            continue
        opt = reg[k]
        octx = replace(ctx, value=values.get(k))
        if not opt.applies(octx):
            continue
        out += opt.vllm_args(octx) if engine == "vllm" else opt.llamacpp_args(octx)
    return out


def vllm_flags(keys: list[str], values: dict[str, str], ctx: OptContext) -> list[str]:
    """Emit vLLM flags for ``keys`` in REGISTRY order (gated by ``applies``)."""
    return _engine_flags(keys, values, ctx, engine="vllm")


def llamacpp_flags(keys: list[str], values: dict[str, str], ctx: OptContext) -> list[str]:
    """Emit llama.cpp flags for ``keys`` in REGISTRY order (gated by ``applies``)."""
    return _engine_flags(keys, values, ctx, engine="llamacpp")
