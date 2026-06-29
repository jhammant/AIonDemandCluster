"""Tracks the single running instance `aiod` manages, so `status` / `teardown`
work across separate CLI invocations. State lives in the user data dir (not the
repo) and never contains secrets beyond the per-launch endpoint token."""

from __future__ import annotations

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


def save(inst: Instance) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(inst), indent=2))


def load() -> Instance | None:
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text())
        return Instance(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def clear() -> None:
    STATE_FILE.unlink(missing_ok=True)
