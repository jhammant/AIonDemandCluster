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
