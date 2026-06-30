"""Pure helpers behind `aiod init` / `aiod doctor`: reading and updating .env
(preserving comments and order) and validating credentials against the live
APIs. Kept free of prompts/printing so it is easy to unit-test."""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
from platformdirs import user_config_dir

ENV_FILE = Path(".env")  # project-local
ENV_EXAMPLE = Path(".env.example")
GLOBAL_ENV = Path(user_config_dir("aiod", appauthor=False)) / ".env"


def env_path() -> Path:
    """Where `init`/`doctor` read & write keys: the project .env if you're inside a
    project, otherwise the global ~/.config/aiod/.env (used by a global install)."""
    return ENV_FILE if ENV_FILE.exists() else GLOBAL_ENV


def read_env(path: Path | None = None) -> dict[str, str]:
    path = path or env_path()
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        values[k.strip()] = v.strip()
    return values


def set_env_values(updates: dict[str, str], path: Path | None = None) -> None:
    """Update keys in-place, preserving existing comments/order; append new keys.
    Seeds from .env.example the first time so the file keeps its documentation."""
    path = path or env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and ENV_EXAMPLE.exists():
        path.write_text(ENV_EXAMPLE.read_text())

    lines = path.read_text().splitlines() if path.exists() else []
    written: set[str] = set()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                written.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in written:
            out.append(f"{k}={v}")
    path.write_text("\n".join(out) + "\n")


def validate_vast_key(key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Authenticated probe. Returns (ok, message)."""
    if not key:
        return False, "no key provided"
    try:
        r = httpx.get(
            "https://console.vast.ai/api/v1/instances/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return False, f"could not reach vast.ai: {e}"
    if r.status_code == 200:
        try:
            n = len(r.json().get("instances", []) or [])
            return True, f"valid (you have {n} instance(s))"
        except ValueError:
            return True, "valid"
    # vast.ai signals a bad/revoked key with an auth_error body — and not always a
    # 401 (a wrong key currently comes back as HTTP 404 with this body).
    try:
        if r.json().get("error") == "auth_error":
            return False, f"rejected (HTTP {r.status_code}) — wrong or revoked key"
    except ValueError:
        pass
    return False, f"unexpected HTTP {r.status_code}"


def validate_runpod_key(key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Authenticated probe against RunPod's REST API."""
    if not key:
        return True, "not set (only needed for --provider runpod)"
    try:
        r = httpx.get(
            "https://rest.runpod.io/v1/pods",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return False, f"could not reach RunPod: {e}"
    if r.status_code == 200:
        return True, "valid"
    if r.status_code in (401, 403):
        return False, "rejected — wrong or revoked key"
    return False, f"unexpected HTTP {r.status_code}"


def validate_hf_token(token: str, timeout: float = 15.0) -> tuple[bool, str]:
    if not token:
        return True, "not set (fine — only gated models need it)"
    try:
        r = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return False, f"could not reach HuggingFace: {e}"
    if r.status_code == 200:
        try:
            return True, f"valid (user: {r.json().get('name', '?')})"
        except ValueError:
            return True, "valid"
    return False, f"rejected (HTTP {r.status_code})"


def ccr_installed() -> str | None:
    """Path to the `ccr` binary, or None if not on PATH."""
    return shutil.which("ccr")
