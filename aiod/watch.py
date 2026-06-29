"""Idle auto-shutdown watcher.

Polls the box's vLLM Prometheus `/metrics` to tell whether the model is doing
any work. After `idle_minutes` with no running/waiting requests and no token
growth, it destroys the instance. The TTL acts as a hard backstop.

This runs LOCALLY (it holds the provider API key), so it never puts secrets on
the public box. Caveat: it must stay running — if the machine sleeps, the watcher
pauses; the TTL backstop still applies once it (or `aiod status`) next runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

# Metric names we treat as "activity".
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
_COUNTERS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total")


def spawn_detached(idle_minutes: int, log_path) -> bool:
    """Start `aiod watch` as a detached background process. Used by spin/TUI so
    idle-shutdown keeps running after the launching command returns."""
    import subprocess
    import sys
    from pathlib import Path

    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            subprocess.Popen(
                [sys.executable, "-m", "aiod.cli", "watch", "--idle", str(idle_minutes)],
                stdout=fh,
                stderr=fh,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return True
    except Exception:
        return False


def metrics_url(base_url: str) -> str:
    """vLLM serves /metrics at the server root, not under /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


def parse_metrics(text: str) -> dict[str, float]:
    """Sum Prometheus samples per base metric name (ignoring labels)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `name{labels} value`  or  `name value`
        try:
            left, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        name = left.split("{", 1)[0].strip()
        totals[name] = totals.get(name, 0.0) + val
    return totals


@dataclass
class Activity:
    in_flight: float  # running + waiting requests right now
    tokens: float  # cumulative generated+prompt tokens (a counter)


def sample_activity(base_url: str, api_key: str | None = None, timeout: float = 8.0) -> Activity | None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(metrics_url(base_url), headers=headers, timeout=timeout)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    m = parse_metrics(r.text)
    in_flight = sum(m.get(g, 0.0) for g in _GAUGES)
    tokens = sum(m.get(c, 0.0) for c in _COUNTERS)
    return Activity(in_flight=in_flight, tokens=tokens)


def is_active(now: Activity, prev: Activity | None) -> bool:
    if now.in_flight > 0:
        return True
    if prev is not None and now.tokens > prev.tokens:
        return True
    return False


def watch_loop(
    base_url: str,
    api_key: str | None,
    idle_minutes: float,
    created_at: float,
    ttl_hours: float,
    destroy: callable,
    poll_seconds: float = 30.0,
    on_event=None,
) -> str:
    """Block until the instance is destroyed. Returns the reason: 'idle' | 'ttl'
    | 'gone'. `destroy` is called once before returning (except for 'gone')."""
    last_active = time.time()
    prev: Activity | None = None
    misses = 0

    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    while True:
        if ttl_hours and (time.time() - created_at) / 3600.0 >= ttl_hours:
            emit("TTL reached — destroying")
            destroy()
            return "ttl"

        act = sample_activity(base_url, api_key=api_key)
        if act is None:
            misses += 1
            # If we can't reach /metrics for a long time, the box may be gone.
            if misses >= 10:
                emit("endpoint unreachable for too long — giving up watch")
                return "gone"
            emit("metrics unreachable, retrying")
        else:
            misses = 0
            if is_active(act, prev):
                last_active = time.time()
            prev = act
            idle_for = (time.time() - last_active) / 60.0
            emit(f"in-flight={act.in_flight:.0f} idle={idle_for:.1f}/{idle_minutes:g}m")
            if idle_for >= idle_minutes:
                emit(f"idle for {idle_minutes:g}m — destroying")
                destroy()
                return "idle"

        time.sleep(poll_seconds)
