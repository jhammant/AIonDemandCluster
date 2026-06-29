"""A tiny cross-process progress log so warm-up status is visible everywhere.

The engine appends phase events here as it launches; the CLI (`status`), the TUI
status panel, and the proxy's "warming up" reply all read it. One JSONL line per
event under the state dir.
"""

from __future__ import annotations

import json
import time

from .state import STATE_DIR

EVENTS_FILE = STATE_DIR / "events.jsonl"

# Ordered phases a launch moves through (for nice display).
PHASES = ["sizing", "searching", "renting", "booting", "loading", "ready", "error"]


def clear() -> None:
    EVENTS_FILE.unlink(missing_ok=True)


def append(phase: str, msg: str = "") -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps({"ts": time.time(), "phase": phase, "msg": msg}) + "\n")


def read(limit: int = 20) -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    out: list[dict] = []
    for line in EVENTS_FILE.read_text().splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def latest() -> dict | None:
    rows = read(1)
    return rows[-1] if rows else None
