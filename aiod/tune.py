"""Pure core for `aiod tune` — the cost-per-million-token optimizer.

This module is deliberately side-effect-free and dependency-light: every
money/IO operation (renting a box, benchmarking it, sizing/pricing) is *injected*
as a callable, so the whole "money path" is unit-testable without Typer, the
network, or a real provider.

Responsibilities (all pure unless a dep is called):

  * :class:`TuneRow` / :class:`Combo` / :class:`Recommendation` — value types.
  * :func:`build_combos` — enumerate the (quant x opt-axis) sweep, routing every
    candidate through :func:`optimizations.resolve` (the SOLE validator) and
    de-duplicating. Always includes the no-opt control.
  * :func:`project_combo` — size + price a combo to a projected $/hr (or None to
    skip it) without renting anything.
  * :func:`rank` / :func:`recommend` — order rows by measured $/1M and pick the
    cheapest *measured* row meeting the latency bar. Projected rows are NEVER
    recommended.
  * :class:`CostGuard` — hard $/minute caps enforced at every boundary.
  * :func:`estimate` — up-front $ / wall-clock estimate from projected $/hr.
  * :func:`run_sweep` — the lifecycle orchestration (launch -> ladder -> teardown)
    with launch/bench/destroy/state INJECTED, so teardown invariants are testable
    with everything mocked.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import optimizations

# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


@dataclass
class TuneRow:
    """One leaderboard row — a measured (combo, concurrency) point, or a modeled
    (``projected=True``) row. Projected rows are display/ordering only and are
    never eligible to be the recommendation."""

    opt_keys: list[str]
    opt_values: dict[str, str]
    opt_tokens: list[str]
    quant: str
    gpu_desc: str
    price_per_hr: float | None
    concurrency: int
    cost_per_million: float | None
    throughput_tok_s: float | None
    ttft_p50: float | None
    ttft_p95: float | None
    avg_decode_tok_s: float | None
    ok: int
    n: int
    projected: bool = False
    error: str | None = None


@dataclass
class Combo:
    """One launch configuration in the sweep (opts are frozen at engine.launch,
    so a combo == one model load). ``key`` is a stable de-dupe id."""

    quant: str
    tokens: list[str]
    opt_keys: list[str]
    opt_values: dict[str, str]
    key: str


@dataclass
class Recommendation:
    row: TuneRow | None
    reason: str
    fallback: bool


# --------------------------------------------------------------------------- #
# Combo enumeration
# --------------------------------------------------------------------------- #


def parse_sweep_axes(specs: list[str] | None) -> list[list[str | None]]:
    """Parse ``--sweep-opt`` specs into opt axes.

    A bare ``KEY`` becomes an on/off toggle axis ``[None, "KEY"]``; a
    ``KEY=v1,v2`` grid becomes a value axis ``["KEY=v1", "KEY=v2"]``.
    """
    axes: list[list[str | None]] = []
    for spec in specs or []:
        if spec is None:
            continue
        if "=" in spec:
            key, grid = spec.split("=", 1)
            key = key.strip()
            vals = [v.strip() for v in grid.split(",") if v.strip()]
            axes.append([f"{key}={v}" for v in vals])
        else:
            key = spec.strip()
            if not key:
                continue
            axes.append([None, key])
    return axes


def _combo_key(quant: str, keys: list[str], values: dict[str, str]) -> str:
    """Stable de-dupe id: quant + the resolved key SET + the value map of the
    value-bearing keys (so distinct max-num-seqs grids don't collapse)."""
    kf = ",".join(sorted(keys))
    vf = ",".join(f"{k}={values[k]}" for k in sorted(values) if k in keys)
    return f"{quant}|{kf}|{vf}"


def build_combos(
    repo: str,
    base_tokens: list[str] | None,
    sweep_axes: list[list[str | None]] | None,
    quants: list[str] | None,
    *,
    engine: str = "vllm",
    draft_model: str | None = None,
    max_conc: int = 16,
) -> list[Combo]:
    """Cartesian product of ``quants`` x opt axes -> de-duplicated combos.

    Every candidate is routed through :func:`optimizations.resolve` (the sole
    validator): a ``conflicts_with`` pair, a duplicate ``provides``, or a value on
    a toggle raises ``ValueError`` and the candidate is dropped. ``resolve`` also
    gates out non-applicable opts (e.g. speculative-decoding with no draft model),
    so those candidates collapse onto the control via de-dup. The no-opt control
    (``opt_keys=[]``) is always present.
    """
    base = list(base_tokens or [])
    axes = sweep_axes or []
    quants = quants or ["bf16"]

    if axes:
        axis_choices: list[tuple] = list(itertools.product(*axes))
    else:
        axis_choices = [()]

    combos: list[Combo] = []
    seen: set[str] = set()
    for quant in quants:
        # Control (base opts only) first, then every axis combination.
        candidates: list[list[str]] = [list(base)]
        for choice in axis_choices:
            candidates.append(base + [t for t in choice if t is not None])

        for tokens in candidates:
            keys, values = optimizations.parse_selection(tokens)
            ctx = optimizations.OptContext(
                engine=engine,
                quant=quant,
                repo_id=repo,
                concurrency=max_conc,
                draft_model=draft_model,
            )
            try:
                resolved = optimizations.resolve(keys, values, ctx)
            except ValueError:
                # conflict / dup-provides / value-on-toggle — resolve is the SOLE
                # validator; drop the candidate (never re-walk conflicts here).
                continue
            rkeys = list(resolved.keys)
            if resolved.requirements.needs_draft_model and not draft_model:
                continue
            rel_values = {k: values[k] for k in rkeys if k in values}
            key = _combo_key(quant, rkeys, rel_values)
            if key in seen:
                continue
            seen.add(key)
            tokens_out = [f"{k}={rel_values[k]}" if k in rel_values else k for k in rkeys]
            combos.append(
                Combo(
                    quant=quant,
                    tokens=tokens_out,
                    opt_keys=rkeys,
                    opt_values=dict(rel_values),
                    key=key,
                )
            )
    return combos


# --------------------------------------------------------------------------- #
# Sizing projection (no rent)
# --------------------------------------------------------------------------- #


def project_combo(
    combo: Combo,
    *,
    repo: str,
    context: int | None,
    max_conc: int,
    size_any: Callable[..., Any],
    price_plan: Callable[..., Any],
    pick_cheapest: Callable[[Any], Any],
    disk: float,
    max_price: float | None,
) -> float | None:
    """Project a combo's $/hr without renting: ``size_any`` -> cheapest GPU fit ->
    ``price_plan`` -> ``pick_cheapest`` -> dph. Returns None (combo skipped) when
    the engine isn't vLLM (opts don't apply to GGUF) or nothing fits/prices.
    """
    sizing = size_any(
        repo,
        quants=[combo.quant],
        context_len=context,
        concurrency=max_conc,
        opts=combo.opt_keys,
        opt_values=combo.opt_values,
    )
    if getattr(sizing, "engine", "vllm") != "vllm":
        return None
    plan = sizing.plan(combo.quant)
    if plan is None or plan.cheapest_fit is None:
        return None
    priced = price_plan(plan, disk, max_price=max_price)
    best = pick_cheapest(priced)
    if best is None or getattr(best, "offer", None) is None:
        return None
    return best.offer.dph_total


# --------------------------------------------------------------------------- #
# Ranking + recommendation
# --------------------------------------------------------------------------- #

_INF = float("inf")


def _p95_key(r: TuneRow) -> float:
    return r.ttft_p95 if r.ttft_p95 is not None else _INF


def rank(
    rows: list[TuneRow],
    *,
    ttft_p95: float | None = None,
    ttft_p50: float | None = None,
    min_decode: float | None = None,
) -> tuple[list[TuneRow], list[TuneRow]]:
    """Return ``(valid_sorted, passing)``.

    ``valid`` = rows with ``ok>0`` and a real ``cost_per_million``, sorted
    ascending by $/1M (tie-break: lower ttft_p95). ``passing`` = ``valid``
    filtered by the (None-guarded) latency bars.
    """
    valid = [r for r in rows if r.ok > 0 and r.cost_per_million is not None]
    valid_sorted = sorted(valid, key=lambda r: (r.cost_per_million, _p95_key(r)))

    def passes(r: TuneRow) -> bool:
        if ttft_p95 is not None and (r.ttft_p95 is None or r.ttft_p95 > ttft_p95):
            return False
        if ttft_p50 is not None and (r.ttft_p50 is None or r.ttft_p50 > ttft_p50):
            return False
        if min_decode is not None and (
            r.avg_decode_tok_s is None or r.avg_decode_tok_s < min_decode
        ):
            return False
        return True

    passing = [r for r in valid_sorted if passes(r)]
    return valid_sorted, passing


def recommend(
    valid: list[TuneRow],
    passing: list[TuneRow],
    *,
    has_bar: bool,
) -> Recommendation:
    """Pick the winner. The winner is ALWAYS a measured row (``projected=False``);
    projected rows are excluded even when cheapest (Judge-3 fix).

    Fallbacks: no bar -> cheapest measured valid row; bar set but nothing passing
    -> the lowest-ttft_p95 measured row (flagged ``fallback=True`` with a reason).
    """
    measured_passing = [r for r in passing if not r.projected]
    if measured_passing:
        reason = (
            "cheapest measured config meeting the latency bar"
            if has_bar
            else "cheapest measured config (no latency bar set)"
        )
        return Recommendation(row=measured_passing[0], reason=reason, fallback=False)

    measured_valid = [r for r in valid if not r.projected]
    if not measured_valid:
        return Recommendation(
            row=None,
            reason="no measured rows to recommend",
            fallback=True,
        )
    if not has_bar:
        return Recommendation(
            row=measured_valid[0],
            reason="cheapest measured config (no latency bar set)",
            fallback=False,
        )
    best = min(measured_valid, key=_p95_key)
    return Recommendation(
        row=best,
        reason="no config met the latency bar; showing the lowest-TTFT measured config "
        "(relax the bar or raise --n)",
        fallback=True,
    )


def projected_row(
    combo: Combo,
    *,
    dph: float | None,
    baseline_throughput: float | None,
    gpu_desc: str,
    concurrency: int,
) -> TuneRow | None:
    """Build a modeled (``projected=True``) leaderboard row for a combo that was
    not measured: assumes tok/s unchanged from the baseline (false for kv-fp8's
    throughput effect — hence ``~`` in the UI and never recommended)."""
    if dph is None or not baseline_throughput:
        return None
    cpm = (dph / 3600.0) / baseline_throughput * 1_000_000
    return TuneRow(
        opt_keys=list(combo.opt_keys),
        opt_values=dict(combo.opt_values),
        opt_tokens=list(combo.tokens),
        quant=combo.quant,
        gpu_desc=gpu_desc,
        price_per_hr=dph,
        concurrency=concurrency,
        cost_per_million=cpm,
        throughput_tok_s=baseline_throughput,
        ttft_p50=None,
        ttft_p95=None,
        avg_decode_tok_s=None,
        ok=0,
        n=0,
        projected=True,
    )


# --------------------------------------------------------------------------- #
# Cost guard
# --------------------------------------------------------------------------- #


class CostGuard:
    """Hard $ / wall-clock caps, enforced at every boundary.

    ``spent`` = finished (torn-down) box cost + the live box's
    ``est_cost_so_far``. ``check`` returns a stop-reason string or None.
    """

    def __init__(
        self,
        max_cost: float | None = None,
        max_minutes: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_cost = max_cost
        self.max_minutes = max_minutes
        self._clock = clock
        self._start = clock()
        self.finished_total = 0.0

    def add_finished(self, cost: float) -> None:
        self.finished_total += max(0.0, cost or 0.0)

    def spent(self, live_inst: Any | None = None, finished_total: float | None = None) -> float:
        ft = self.finished_total if finished_total is None else finished_total
        live = getattr(live_inst, "est_cost_so_far", 0.0) if live_inst is not None else 0.0
        return ft + (live or 0.0)

    def would_exceed(self, next_box_est: float, live_inst: Any | None = None) -> bool:
        if self.max_cost is None:
            return False
        return self.spent(live_inst) + max(0.0, next_box_est or 0.0) > self.max_cost

    def elapsed_minutes(self) -> float:
        return (self._clock() - self._start) / 60.0

    def time_exceeded(self) -> bool:
        if self.max_minutes is None:
            return False
        return self.elapsed_minutes() >= self.max_minutes

    def check(self, live_inst: Any | None = None) -> str | None:
        if self.max_cost is not None and self.spent(live_inst) >= self.max_cost:
            return "max-cost"
        if self.time_exceeded():
            return "max-minutes"
        return None


# --------------------------------------------------------------------------- #
# Up-front estimate
# --------------------------------------------------------------------------- #


def estimate(
    combos: list[Combo],
    projected_dph_by_key: dict[str, float],
    *,
    load_minutes_guess: float,
    per_point_minutes: float,
    n_points: int,
    cold_pull_contingency: float = 1.0,
) -> tuple[float, float]:
    """Up-front (est_cost, est_minutes) from injected projected $/hr.

    Each box costs ``dph/60 * box_minutes`` where ``box_minutes`` = the
    (contingency-widened) load guess + a bench point per ladder rung. Mode-B
    multi-combo sweeps pass ``cold_pull_contingency>1`` to widen the least
    predictable term (cold HF pulls on vast).
    """
    load = max(0.0, load_minutes_guess) * max(1.0, cold_pull_contingency)
    est_cost = 0.0
    est_minutes = 0.0
    for combo in combos:
        dph = projected_dph_by_key.get(combo.key)
        if dph is None:
            continue
        box_minutes = load + per_point_minutes * max(0, n_points)
        est_minutes += box_minutes
        est_cost += dph / 60.0 * box_minutes
    return est_cost, est_minutes


# --------------------------------------------------------------------------- #
# Orchestration (deps injected -> fully mockable)
# --------------------------------------------------------------------------- #


@dataclass
class SweepDeps:
    """Injected money/IO operations. ``destroy`` should be idempotent against a
    box that is already gone (provider 404 == success)."""

    launch: Callable[[Combo], Any]  # (combo) -> Instance | None
    bench: Callable[[Any, int], Any]  # (inst, concurrency) -> BenchResult
    destroy: Callable[[Any], Any]  # (inst) -> None
    state_load: Callable[[], Any]  # () -> Instance | None
    state_clear: Callable[[], None]  # () -> None
    on_event: Callable[[str, str], None] = lambda *_a: None


@dataclass
class SweepPlan:
    combos: list[Combo]
    ladder: list[int]
    guard: CostGuard
    projected_dph: dict[str, float] = field(default_factory=dict)
    ttft_p95: float | None = None
    ttft_p50: float | None = None
    min_decode: float | None = None
    early_stop: bool = True
    load_minutes_guess: float = 6.0


@dataclass
class SweepResult:
    rows: list[TuneRow]
    stop_reason: str | None = None
    interrupted: bool = False


def _failure_row(combo: Combo, inst: Any | None, error: str | None) -> TuneRow:
    return TuneRow(
        opt_keys=list(combo.opt_keys),
        opt_values=dict(combo.opt_values),
        opt_tokens=list(combo.tokens),
        quant=combo.quant,
        gpu_desc=getattr(inst, "gpu_desc", "") if inst is not None else "",
        price_per_hr=getattr(inst, "price_per_hr", None) if inst is not None else None,
        concurrency=0,
        cost_per_million=None,
        throughput_tok_s=None,
        ttft_p50=None,
        ttft_p95=None,
        avg_decode_tok_s=None,
        ok=0,
        n=0,
        error=error or "launch failed",
    )


def _row_from_bench(combo: Combo, inst: Any, concurrency: int, res: Any) -> TuneRow:
    okn = len(res.ok)
    return TuneRow(
        opt_keys=list(combo.opt_keys),
        opt_values=dict(combo.opt_values),
        opt_tokens=list(combo.tokens),
        quant=combo.quant,
        gpu_desc=getattr(inst, "gpu_desc", ""),
        price_per_hr=getattr(inst, "price_per_hr", None),
        concurrency=concurrency,
        cost_per_million=res.cost_per_million,
        throughput_tok_s=res.throughput_tok_s,
        ttft_p50=res.ttft_p50,
        ttft_p95=res.ttft_p95,
        avg_decode_tok_s=res.avg_decode_tok_s,
        ok=okn,
        n=res.n,
        error=None if okn else "all requests failed",
    )


def _box_est(dph: float | None, plan: SweepPlan) -> float:
    """Rough cost of one box for the whole ladder (used at the launch boundary)."""
    if dph is None:
        return 0.0
    minutes = plan.load_minutes_guess + 1.0 * max(1, len(plan.ladder))
    return dph / 60.0 * minutes


def run_sweep(deps: SweepDeps, plan: SweepPlan) -> SweepResult:
    """Sequential lifecycle: for each combo, launch one box, run the concurrency
    ladder, then GUARANTEE teardown — on success, error, or KeyboardInterrupt —
    destroying ``inst`` if bound else ``state_load()`` (covers the boot window
    before ``inst`` is bound). Hard caps are checked at every boundary.
    """
    rows: list[TuneRow] = []
    guard = plan.guard
    stop_reason: str | None = None
    interrupted = False
    incumbent: float | None = None  # best measured $/1M so far
    best_throughput: float | None = None

    def teardown(inst: Any | None) -> None:
        target = inst if inst is not None else deps.state_load()
        if target is not None:
            try:
                deps.destroy(target)
            except Exception:  # noqa: BLE001 - destroy on a gone box is harmless
                pass
        # belt-and-suspenders: the single state slot must be empty afterward.
        try:
            if deps.state_load() is not None:
                deps.state_clear()
        except Exception:  # noqa: BLE001
            pass

    try:
        for combo in plan.combos:
            r = guard.check()
            if r is not None:
                stop_reason = r
                break
            dph = plan.projected_dph.get(combo.key)
            if dph is not None and guard.would_exceed(_box_est(dph, plan)):
                stop_reason = "max-cost"
                break
            # Early-stop: a combo whose projected $/1M floor can't beat the
            # incumbent is never rented.
            if (
                plan.early_stop
                and incumbent is not None
                and dph is not None
                and best_throughput
            ):
                floor = (dph / 3600.0) / best_throughput * 1_000_000
                if floor >= incumbent:
                    deps.on_event(
                        "skip",
                        f"{combo.key}: projected floor ~${floor:.3f}/1M can't beat "
                        f"${incumbent:.3f}/1M",
                    )
                    continue

            inst: Any | None = None
            try:
                try:
                    inst = deps.launch(combo)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001 - record + continue the sweep
                    rows.append(_failure_row(combo, None, str(e)))
                    continue
                if inst is None or getattr(inst, "status", None) != "running":
                    rows.append(_failure_row(combo, inst, getattr(inst, "status", None)))
                    continue

                for c in plan.ladder:
                    r = guard.check(live_inst=inst)
                    if r is not None:
                        stop_reason = r
                        break
                    try:
                        res = deps.bench(inst, c)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:  # noqa: BLE001
                        rows.append(_failure_row(combo, inst, str(e)))
                        break
                    row = _row_from_bench(combo, inst, c, res)
                    rows.append(row)
                    if row.cost_per_million is not None and row.ok > 0:
                        if incumbent is None or row.cost_per_million < incumbent:
                            incumbent = row.cost_per_million
                    if row.throughput_tok_s and (
                        best_throughput is None or row.throughput_tok_s > best_throughput
                    ):
                        best_throughput = row.throughput_tok_s
                    r = guard.check(live_inst=inst)
                    if r is not None:
                        stop_reason = r
                        break
            finally:
                if inst is not None:
                    guard.add_finished(getattr(inst, "est_cost_so_far", 0.0) or 0.0)
                teardown(inst)

            if stop_reason is not None:
                break
    except KeyboardInterrupt:
        interrupted = True
        teardown(deps.state_load())
        stop_reason = stop_reason or "interrupted"

    # Final safety net: the single state slot MUST be empty when we return.
    try:
        if deps.state_load() is not None:
            teardown(deps.state_load())
    except Exception:  # noqa: BLE001
        pass

    return SweepResult(rows=rows, stop_reason=stop_reason, interrupted=interrupted)
