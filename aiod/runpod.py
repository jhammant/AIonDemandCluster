"""RunPod backend.

Implements the same duck-typed surface as VastClient (price_plan / create_instance
/ get_instance / endpoint_of / status_of / destroy_instance / build_create_body),
so it's a drop-in provider.

Verified API shapes (June 2026):
  * Auth: Authorization: Bearer <RUNPOD_API_KEY>
  * Pricing/VRAM are NOT in REST -> one GraphQL call to gpuTypes.
  * Create: POST https://rest.runpod.io/v1/pods  (gpuTypeIds[], gpuCount,
    containerDiskInGb, ports:["8000/tcp"], env{}, dockerStartCmd[], cloudType SECURE).
    Returns {"id": "<podId>"}.
  * Endpoint: GET /v1/pods/{id} -> publicIp + portMappings{"8000": <ext>}; only
    populated once the pod is placed. base = http://{publicIp}:{ext}/v1.
    We expose 8000/**tcp** (direct public IP), NOT the *.proxy.runpod.net URL,
    whose ~100s Cloudflare timeout would break long streamed responses.
  * Destroy: DELETE /v1/pods/{id} -> 204.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .bootstrap import ServerConfig
from .sizing import QuantPlan
from .vast import PricedOption, gpu_name_matches

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"


class RunpodError(Exception):
    pass


@dataclass
class RunpodOffer:
    id: str  # gpuTypeId, e.g. "NVIDIA A100 80GB PCIe"
    gpu_name: str
    num_gpus: int
    dph_total: float  # estimated $/hr for the whole pod (secure on-demand)
    total_vram_gb: float
    reliability: float = 1.0
    geolocation: str | None = None

    @property
    def desc(self) -> str:
        return f"{self.num_gpus}x {self.gpu_name}"


class RunpodClient:
    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise RunpodError("RUNPOD_API_KEY is not set. Add it to .env or run `aiod init`.")
        self._c = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        self._gpu_types: list[dict] | None = None

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> RunpodClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Pricing / VRAM (GraphQL — not available over REST)
    # ------------------------------------------------------------------ #

    def _fetch_gpu_types(self) -> list[dict]:
        if self._gpu_types is not None:
            return self._gpu_types
        query = (
            "query { gpuTypes { id displayName memoryInGb secureCloud "
            "communityCloud securePrice communityPrice } }"
        )
        try:
            r = self._c.post(GRAPHQL, json={"query": query})
        except httpx.HTTPError as e:
            raise RunpodError(f"RunPod GraphQL unreachable: {e}") from e
        if r.status_code == 401:
            raise RunpodError("RunPod rejected the API key (401). Check RUNPOD_API_KEY.")
        if r.status_code >= 400:
            raise RunpodError(f"RunPod GraphQL HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if data.get("errors"):
            raise RunpodError(f"RunPod GraphQL error: {data['errors']}")
        self._gpu_types = data.get("data", {}).get("gpuTypes", []) or []
        return self._gpu_types

    def _cheapest_gpu_for(
        self, vram_gb: float, gpu_match: list[str] | None = None
    ) -> dict | None:
        """Cheapest SECURE gpu type in the same VRAM class as `vram_gb`."""
        queries = [g for g in (gpu_match or []) if g.strip()]
        cands: list[tuple[float, dict]] = []
        for g in self._fetch_gpu_types():
            mem, price = g.get("memoryInGb"), g.get("securePrice")
            if not mem or not price or not g.get("secureCloud"):
                continue
            if queries:
                name = str(g.get("displayName", g.get("id", "")))
                if not gpu_name_matches(name, queries):
                    continue
            if vram_gb * 0.95 <= mem <= vram_gb * 1.6:
                cands.append((price, g))
        if not cands:
            return None
        return min(cands, key=lambda x: x[0])[1]

    def price_plan(
        self, plan: QuantPlan, min_disk_gb: float, max_price: float | None = None,
        max_candidates: int = 3, gpu_match: list[str] | None = None,
    ) -> list[PricedOption]:
        """Price the few least-wasteful configs that fit (mirrors the vast backend;
        all pricing comes from one cached GraphQL call, so this is cheap)."""
        opts = sorted((o for o in plan.options if o.fits), key=lambda o: o.total_vram_gb)
        priced: list[PricedOption] = []
        for opt in opts[:max_candidates]:
            g = self._cheapest_gpu_for(opt.tier.vram_gb, gpu_match=gpu_match)
            offer = None
            if g:
                total = g["securePrice"] * opt.num_gpus
                if max_price is None or total <= max_price:
                    offer = RunpodOffer(
                        id=g["id"],
                        gpu_name=g.get("displayName", g["id"]),
                        num_gpus=opt.num_gpus,
                        dph_total=total,
                        total_vram_gb=g["memoryInGb"] * opt.num_gpus,
                    )
            priced.append(PricedOption(option=opt, offer=offer))
        return priced

    # ------------------------------------------------------------------ #
    # Create / poll / destroy (REST)
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_create_body(
        cfg: ServerConfig, disk_gb: int, max_price: float, label: str = "aiod-vllm",
        gpu_type_id: str = "<gpu-type>",
    ) -> dict:
        return {
            "name": label,
            "imageName": cfg.image,
            "cloudType": "SECURE",
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": cfg.num_gpus,
            "containerDiskInGb": int(disk_gb),
            "volumeInGb": 0,
            "ports": [f"{cfg.port}/tcp"],  # public TCP — not the proxy URL
            "supportPublicIp": True,
            "env": cfg.env(),
            "dockerStartCmd": cfg.server_args(),  # vLLM flags override the image CMD
        }

    def create_instance(
        self, offer_id, cfg: ServerConfig, disk_gb: int, max_price: float, label: str = "aiod-vllm"
    ) -> str:
        body = self.build_create_body(cfg, disk_gb, max_price, label, gpu_type_id=str(offer_id))
        try:
            r = self._c.post(f"{REST}/pods", json=body)
        except httpx.HTTPError as e:
            raise RunpodError(f"RunPod create failed: {e}") from e
        if r.status_code >= 400:
            raise RunpodError(f"RunPod create HTTP {r.status_code}: {r.text[:300]}")
        pod = r.json()
        pid = pod.get("id")
        if not pid:
            raise RunpodError(f"RunPod returned no pod id: {pod}")
        return str(pid)

    def get_instance(self, instance_id) -> dict:
        try:
            r = self._c.get(f"{REST}/pods/{instance_id}")
        except httpx.HTTPError as e:
            raise RunpodError(f"RunPod get failed: {e}") from e
        if r.status_code >= 400:
            raise RunpodError(f"RunPod get HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def destroy_instance(self, instance_id) -> None:
        try:
            r = self._c.delete(f"{REST}/pods/{instance_id}")
        except httpx.HTTPError as e:
            raise RunpodError(f"RunPod destroy failed: {e}") from e
        if r.status_code >= 400 and r.status_code != 404:
            raise RunpodError(f"RunPod destroy HTTP {r.status_code}: {r.text[:200]}")

    @staticmethod
    def endpoint_of(pod: dict, container_port: int = 8000) -> tuple[str, int] | None:
        """Public TCP host:port — populated only once the pod is placed/running."""
        ip = pod.get("publicIp")
        mappings = pod.get("portMappings") or {}
        ext = mappings.get(str(container_port))
        if ip and ext:
            return str(ip), int(ext)
        return None

    @staticmethod
    def status_of(pod: dict) -> str:
        # desiredStatus is the *desired* state; real readiness is endpoint_of() != None.
        return str(pod.get("desiredStatus") or "unknown")
