"""Loads configuration from the environment / a local .env file."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_dir

# Project-local .env wins (override=False keeps the first-loaded values); a global
# ~/.config/aiod/.env is the fallback so a globally-installed `aiod` finds your keys
# from any directory. OS environment variables take precedence over both.
GLOBAL_ENV = Path(user_config_dir("aiod", appauthor=False)) / ".env"

load_dotenv()  # CWD .env (project)
load_dotenv(GLOBAL_ENV)  # global fallback


@dataclass
class Settings:
    vast_api_key: str
    hf_token: str | None
    vllm_api_key: str  # bearer token protecting the endpoint (generated if unset)
    ttl_hours: float
    max_price: float
    runpod_api_key: str = ""

    @classmethod
    def load(cls) -> Settings:
        return cls(
            vast_api_key=os.getenv("VAST_API_KEY", "").strip(),
            hf_token=(os.getenv("HF_TOKEN") or "").strip() or None,
            vllm_api_key=(os.getenv("VLLM_API_KEY") or "").strip() or f"sk-aiod-{secrets.token_hex(16)}",
            ttl_hours=float(os.getenv("AIOD_TTL_HOURS", "4") or 4),
            max_price=float(os.getenv("AIOD_MAX_PRICE", "6.0") or 6.0),
            runpod_api_key=os.getenv("RUNPOD_API_KEY", "").strip(),
        )


def was_token_minted(s: Settings) -> bool:
    """True when the bearer token was freshly minted this process (VLLM_API_KEY
    unset in the environment), so callers know to persist + warn exactly once."""
    return s.vllm_api_key.startswith("sk-aiod-") and not os.getenv("VLLM_API_KEY")


def persist_vllm_api_key(token: str, env_path: Path = GLOBAL_ENV) -> bool:
    """Persist a minted bearer token to the global ~/.config/aiod/.env so CCR, the
    gateway bearer, the upstream vLLM bearer and the chat page share one
    deterministic token across restarts.

    No-op (returns False) when VLLM_API_KEY is already set in the OS environment or
    already present in the global .env. Returns True when it appends the token."""
    if os.getenv("VLLM_API_KEY"):
        return False
    existing = env_path.read_text() if env_path.exists() else ""
    if "VLLM_API_KEY=" in existing:
        return False
    env_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    with open(env_path, "a") as f:
        f.write(f"{sep}VLLM_API_KEY={token}\n")
    return True
