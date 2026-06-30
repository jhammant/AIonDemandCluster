"""Live smoke probes — hit the real vendor endpoints aiod depends on, to catch
upstream API changes (deprecations / response-shape changes) BEFORE they break
users. This is the one failure class the mocked unit suite can't catch: PR #1
fixed exactly such a break (vast.ai retired its v0 instance-list endpoint → 410),
and no mocked test would ever have flagged it.

These are EXCLUDED from the default run via pyproject `addopts = -m "not live"`,
so `pytest tests/` stays hermetic and needs no network or secrets. Run them
explicitly:

    pytest -m live                          # auth-free probes (+ any with keys in env)
    VAST_API_KEY=... RUNPOD_API_KEY=... pytest -m live   # also the authenticated paths

The weekly GitHub Action (.github/workflows/smoke.yml) runs these and opens an
issue if a depended-on endpoint changes. The auth-free probes (RunPod pricing,
HuggingFace sizing) run there with no secrets; the vast/runpod authenticated
probes only run if you add those API keys as repo Actions secrets.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.live

VAST_KEY = os.environ.get("VAST_API_KEY", "").strip()
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
TIMEOUT = 30.0


# --- auth-free: the pricing / sizing paths (no secret required) --------------


def test_runpod_gpu_types_graphql():
    """RunPod pricing + VRAM come from the gpuTypes GraphQL query. aiod/runpod.py
    reads memoryInGb / securePrice / secureCloud / displayName off each entry."""
    r = httpx.post(
        "https://api.runpod.io/graphql",
        json={"query": "query{ gpuTypes{ id displayName memoryInGb securePrice secureCloud } }"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"runpod GraphQL → HTTP {r.status_code}"
    types = r.json().get("data", {}).get("gpuTypes")
    assert isinstance(types, list) and types, "gpuTypes empty/missing — query shape changed?"
    needed = {"memoryInGb", "securePrice", "secureCloud", "displayName"}
    missing = needed - set(types[0])
    assert not missing, f"runpod gpuType fields changed; missing {missing}"


def test_hf_model_api():
    """Sizing reads the HuggingFace model API / file tree (aiod/sizing.py)."""
    r = httpx.get(
        "https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"HF model API → HTTP {r.status_code}"
    assert "siblings" in r.json(), "HF model payload shape changed (no 'siblings')"


# --- authenticated: the endpoints that actually broke (need a key) -----------


@pytest.mark.skipif(not VAST_KEY, reason="VAST_API_KEY not set")
def test_vast_v1_instances():
    """The list endpoint PR #1 fixed: v0 was retired (410). Confirm v1 still
    answers 200 with the {'instances': [...]} shape validate_vast_key relies on."""
    from aiod.onboard import validate_vast_key

    ok, msg = validate_vast_key(VAST_KEY)
    assert ok, f"vast v1 instances probe failed: {msg}"


@pytest.mark.skipif(not VAST_KEY, reason="VAST_API_KEY not set")
def test_vast_bundles_search():
    """Offer search (POST /api/v0/bundles/) is still on v0 — make sure that
    namespace is alive and the call returns a list (the market may be thin)."""
    from aiod.vast import VastClient

    offers = VastClient(VAST_KEY).search_offers(
        num_gpus=1, min_gpu_ram_gb=24, min_disk_gb=20, limit=1
    )
    assert isinstance(offers, list)


@pytest.mark.skipif(not RUNPOD_KEY, reason="RUNPOD_API_KEY not set")
def test_runpod_rest_pods():
    """RunPod REST list endpoint (aiod/runpod.py uses GET /v1/pods)."""
    from aiod.onboard import validate_runpod_key

    ok, msg = validate_runpod_key(RUNPOD_KEY)
    assert ok, f"runpod REST probe failed: {msg}"
