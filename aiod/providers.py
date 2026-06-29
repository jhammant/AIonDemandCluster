"""Backend factory. Each provider exposes the same surface the CLI/TUI use
(search_offers / price_plan / create_instance / get_instance / endpoint_of /
status_of / destroy_instance), so they're drop-in. vast.ai is the first; RunPod
is next (its API returns a TCP ip:publicPort we'll use instead of the proxy URL,
to avoid the 100s streaming timeout).
"""

from __future__ import annotations

from .config import Settings
from .runpod import RunpodClient, RunpodError
from .vast import VastClient, VastError

SUPPORTED = ["vast", "runpod"]


class ProviderError(Exception):
    pass


# Any of these means "a provider call failed" — flows catch this tuple.
PROVIDER_ERRORS = (ProviderError, VastError, RunpodError)


def get_client(provider: str, settings: Settings):
    provider = (provider or "vast").lower()
    if provider == "vast":
        return VastClient(settings.vast_api_key)
    if provider == "runpod":
        return RunpodClient(settings.runpod_api_key)
    raise ProviderError(f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED)}")


def api_key_for(provider: str, settings: Settings) -> str:
    provider = (provider or "vast").lower()
    if provider == "vast":
        return settings.vast_api_key
    if provider == "runpod":
        return settings.runpod_api_key
    return ""


__all__ = [
    "get_client", "api_key_for", "ProviderError", "PROVIDER_ERRORS",
    "VastError", "RunpodError", "SUPPORTED",
]
