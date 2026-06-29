"""Generate / merge the Claude Code Router config so `ccr code` routes Claude
Code to our remote vLLM box.

Verified schema notes (musistudio/claude-code-router, classic CLI):
  * config path: ~/.claude-code-router/config.json
  * Providers[].api_base_url must be the FULL URL ending in /chat/completions
    (NOT just the base) for a custom OpenAI-compatible endpoint like vLLM.
  * Providers[].api_key is sent upstream as `Authorization: Bearer`.
  * Providers[].models lists names exactly as /v1/models reports them.
  * transformer is omitted for a vanilla OpenAI-compatible endpoint.
  * Router values are the string "providerName,modelName".

We MERGE: existing providers and top-level settings are preserved; we only
replace our own provider (matched by name) and repoint the Router.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

CONFIG_DIR = Path.home() / ".claude-code-router"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROVIDER_NAME = "aiod-vllm"


def build_provider(base_url: str, api_key: str, model: str) -> dict:
    """base_url ends in /v1 (e.g. http://1.2.3.4:33526/v1)."""
    return {
        "name": PROVIDER_NAME,
        "api_base_url": f"{base_url.rstrip('/')}/chat/completions",
        "api_key": api_key or "dummy",
        "models": [model],
    }


def build_router(model: str) -> dict:
    ref = f"{PROVIDER_NAME},{model}"
    return {
        "default": ref,
        "background": ref,
        "think": ref,
        "longContext": ref,
        "longContextThreshold": 60000,
        "webSearch": ref,
    }


def _load_existing() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def write_config(base_url: str, api_key: str, model: str) -> Path:
    """Merge our provider+router into the CCR config, preserving everything else.
    Backs up an existing config to config.json.bak. Returns the config path."""
    cfg = _load_existing()

    if CONFIG_FILE.exists():
        shutil.copy2(CONFIG_FILE, CONFIG_FILE.with_suffix(".json.bak"))

    cfg.setdefault("LOG", False)
    cfg.setdefault("HOST", "127.0.0.1")
    cfg.setdefault("PORT", 3456)

    providers = cfg.get("Providers")
    if not isinstance(providers, list):
        providers = []
    providers = [p for p in providers if p.get("name") != PROVIDER_NAME]
    providers.append(build_provider(base_url, api_key, model))
    cfg["Providers"] = providers

    cfg["Router"] = build_router(model)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    return CONFIG_FILE


def config_snippet(base_url: str, api_key: str, model: str) -> str:
    """The exact provider+router block, for printing / dry-run."""
    return json.dumps(
        {"Providers": [build_provider(base_url, api_key, model)], "Router": build_router(model)},
        indent=2,
    )
