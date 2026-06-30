"""Tracks the single running instance `aiod` manages, so `status` / `teardown`
work across separate CLI invocations. State lives in the user data dir (not the
repo) and never contains secrets beyond the per-launch endpoint token."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_dir

STATE_DIR = Path(user_data_dir("aiod", appauthor=False))
STATE_FILE = STATE_DIR / "state.json"


@dataclass
class Instance:
    instance_id: int | str  # vast uses an int contract id; runpod uses a string pod id
    repo_id: str
    quant: str
    gpu_desc: str  # e.g. "2x H100 80GB"
    price_per_hr: float
    created_at: float  # unix seconds
    ttl_hours: float
    host: str | None = None  # public IP / hostname
    port: int | None = None  # mapped external port for the vLLM endpoint
    api_key: str | None = None  # bearer token protecting the endpoint
    status: str = "creating"  # creating | loading | running | error
    provider: str = "vast"  # which backend rented this (vast | runpod | ...)
    idle_minutes: int | None = None  # auto-shutdown threshold, if a watcher is running
    weights_gb: float | None = None  # model download size, for download-progress %

    @property
    def base_url(self) -> str | None:
        if self.host and self.port:
            return f"http://{self.host}:{self.port}/v1"
        return None

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0

    @property
    def expires_in_hours(self) -> float:
        return self.ttl_hours - self.age_hours

    @property
    def est_cost_so_far(self) -> float:
        return self.age_hours * self.price_per_hr


def download_progress(
    disk_usage_gb: float | None, weights_gb: float | None, gpu_util: float | None = None
) -> str | None:
    """A human 'Download' line from live telemetry, e.g.
    '135 / 343 GB  (39%) — downloading weights'. Returns None when there's no
    usable figure. `disk_usage_gb` counts the container image too, so the % runs a
    little optimistic; GPU activity is what tells us the load phase has started."""
    if disk_usage_gb is None:
        return None
    if not weights_gb:
        return f"{disk_usage_gb:.0f} GB on disk"
    pct = min(100.0, disk_usage_gb / weights_gb * 100)
    if gpu_util and gpu_util > 0:
        phase = "loading into VRAM"
    elif pct >= 99:
        phase = "download complete — loading"
    else:
        phase = "downloading weights"
    return f"{disk_usage_gb:.0f} / {weights_gb:.0f} GB  ({pct:.0f}%) — {phase}"


def save(inst: Instance) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(inst), indent=2))


def load() -> Instance | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        # Tolerate fields written by a newer version (drop unknowns) so an older
        # binary doesn't lose track of a live instance.
        fields = {f.name for f in dataclasses.fields(Instance)}
        return Instance(**{k: v for k, v in data.items() if k in fields})
    except (json.JSONDecodeError, TypeError):
        return None


def clear() -> None:
    STATE_FILE.unlink(missing_ok=True)
