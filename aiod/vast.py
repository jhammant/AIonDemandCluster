"""vast.ai REST client.

We talk to the REST API directly (not the `vastai` SDK) because the SDK is a thin
auto-generated wrapper over the CLI with two input dialects (query-string vs JSON
operators, env-string vs env-dict) that are easy to get subtly wrong. Direct REST
gives clean JSON and exact control over the `env` dict and `ports` parsing.

Key facts encoded here (verified against the vast-cli source):
  * Auth:      Authorization: Bearer <key>
  * Search:    POST /api/v0/bundles/   with JSON operator filters; gpu_ram is
               PER-GPU VRAM in MB; needs direct_port_count >= 1 to be reachable.
  * Create:    PUT  /api/v0/asks/{offer_id}/   ; env is a DICT, port maps are
               keys like "-p 8000:8000": "1"; returns {"new_contract": <id>}.
  * Endpoint:  GET  /api/v0/instances/{id}/    ; host = public_ipaddr,
               port = ports["8000/tcp"][0]["HostPort"].
  * Destroy:   DELETE /api/v0/instances/{id}/
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import httpx

from .bootstrap import ServerConfig
from .sizing import GpuOption, QuantPlan

DEFAULT_BASE = "https://console.vast.ai"


class VastError(Exception):
    pass


def gpu_name_matches(name: str, queries: list[str]) -> bool:
    """True if `name` matches any --gpu query. A query matches when ALL its
    whitespace-separated tokens appear (case-insensitively) in the GPU name, so
    `rtx 6000` matches "RTX PRO 6000 WS" and `a6000` matches "RTX A6000". A single
    glued token like `rtx6000` only matches a contiguous run, so prefer spaces."""
    n = name.lower()
    for q in queries:
        toks = q.lower().split()
        if toks and all(t in n for t in toks):
            return True
    return False


@dataclass
class Offer:
    id: int
    gpu_name: str
    gpu_ram_mb: int  # per-GPU
    num_gpus: int
    dph_total: float  # $/hr for the whole machine
    reliability: float
    disk_space: float
    direct_port_count: int
    geolocation: str | None
    cuda_max_good: float | None

    @property
    def per_gpu_vram_gb(self) -> float:
        return self.gpu_ram_mb / 1000.0

    @property
    def total_vram_gb(self) -> float:
        return self.per_gpu_vram_gb * self.num_gpus

    @property
    def desc(self) -> str:
        return f"{self.num_gpus}x {self.gpu_name} ({self.per_gpu_vram_gb:.0f}GB)"


@dataclass
class PricedOption:
    """A GpuOption from the sizing engine paired with the cheapest real offer."""

    option: GpuOption
    offer: Offer | None

    @property
    def price_per_hr(self) -> float | None:
        return self.offer.dph_total if self.offer else None


class VastClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE, timeout: float = 30.0):
        if not api_key:
            raise VastError("VAST_API_KEY is not set. Add it to .env or your environment.")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        # vast.ai rate-limits the bundles API (~5 rapid calls). Throttle + back off.
        self._min_interval = 0.3
        self._last_req = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VastClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Offers
    # ------------------------------------------------------------------ #

    def search_offers(
        self,
        *,
        num_gpus: int,
        min_gpu_ram_gb: float,
        min_disk_gb: float,
        max_price: float | None = None,
        min_reliability: float = 0.95,
        min_compute_cap: int = 800,
        min_inet_mbps: int = 400,
        gpu_names: list[str] | None = None,
        gpu_match: list[str] | None = None,
        limit: int = 30,
    ) -> list[Offer]:
        query: dict = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "external": {"eq": False},
            "num_gpus": {"eq": num_gpus},
            # gpu_ram is per-GPU in MB; 0.95 margin so 80GB cards (~81920) match an 80 tier.
            "gpu_ram": {"gte": int(min_gpu_ram_gb * 1000 * 0.95)},
            "disk_space": {"gte": int(min_disk_gb)},
            "reliability2": {"gte": min_reliability},
            "direct_port_count": {"gte": 1},
            # Compute-capability floor. 800 (Ampere) by default — recent vLLM dropped
            # pre-Ampere (V100/T4/P100). Raised to 890 (Ada) for fp8, which is
            # numerically broken on Ampere (produces garbage); see price_plan.
            "compute_cap": {"gte": min_compute_cap},
            # Avoid slow-network hosts: the image + weights download there. The floor
            # scales with model size in price_plan (a 343GB GGUF on a slow node would
            # cost a fortune in idle GPU time during the download).
            "inet_down": {"gte": min_inet_mbps},
            "type": "ondemand",
            "order": [["dph_total", "asc"]],
            # When substring-filtering by GPU name we fetch a wider pool and trim
            # client-side, since vast's gpu_name filter only does exact matches.
            "limit": max(limit, 50) if gpu_match else limit,
        }
        if max_price is not None:
            query["dph_total"] = {"lte": float(max_price)}
        if gpu_names:
            query["gpu_name"] = {"in": gpu_names}

        data = self._post("/api/v0/bundles/", query)
        offers = [self._parse_offer(o) for o in (data.get("offers", []) or [])]
        if gpu_match:
            queries = [g for g in gpu_match if g.strip()]
            offers = [o for o in offers if gpu_name_matches(o.gpu_name, queries)]
            offers = offers[:limit]
        return offers

    @staticmethod
    def _parse_offer(o: dict) -> Offer:
        return Offer(
            id=int(o["id"]),
            gpu_name=str(o.get("gpu_name", "?")),
            gpu_ram_mb=int(o.get("gpu_ram", 0)),
            num_gpus=int(o.get("num_gpus", 1)),
            dph_total=float(o.get("dph_total", 0.0)),
            reliability=float(o.get("reliability2", o.get("reliability", 0.0)) or 0.0),
            disk_space=float(o.get("disk_space", 0.0)),
            direct_port_count=int(o.get("direct_port_count", 0)),
            geolocation=o.get("geolocation"),
            cuda_max_good=o.get("cuda_max_good"),
        )

    # ------------------------------------------------------------------ #
    # Instances
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_create_body(
        cfg: ServerConfig, disk_gb: int, max_price: float, label: str = "aiod-vllm"
    ) -> dict:
        """The exact body sent to PUT /asks/{id}/. Exposed so `--dry-run` can show it."""
        env: dict[str, str] = {f"-p {cfg.port}:{cfg.port}": "1"}
        env.update(cfg.env())
        return {
            "client_id": "me",
            "image": cfg.image,
            "disk": disk_gb,
            "runtype": "args",  # append our args to the image's vLLM entrypoint
            "price": float(max_price),
            "label": label,
            "env": env,
            "args": cfg.server_args(),
            "target_state": "running",
        }

    def create_instance(
        self,
        offer_id: int,
        cfg: ServerConfig,
        disk_gb: int,
        max_price: float,
        label: str = "aiod-vllm",
    ) -> int:
        """Rent `offer_id` and start the vLLM server. Returns the instance id."""
        body = self.build_create_body(cfg, disk_gb, max_price, label)
        data = self._put(f"/api/v0/asks/{offer_id}/", body)
        if not data.get("success"):
            raise VastError(f"vast.ai refused to create the instance: {data}")
        new_contract = data.get("new_contract")
        if not new_contract:
            raise VastError(f"vast.ai did not return an instance id: {data}")
        return int(new_contract)

    def get_instance(self, instance_id: int) -> dict:
        data = self._get(f"/api/v0/instances/{instance_id}/")
        inst = data.get("instances")
        if inst is None:
            raise VastError(f"Instance {instance_id} not found.")
        # Some responses wrap a single instance in a list.
        if isinstance(inst, list):
            if not inst:
                raise VastError(f"Instance {instance_id} not found.")
            return inst[0]
        return inst

    def list_instances(self) -> list[dict]:
        data = self._get("/api/v0/instances/")
        inst = data.get("instances", [])
        return inst if isinstance(inst, list) else [inst]

    def destroy_instance(self, instance_id: int) -> None:
        self._delete(f"/api/v0/instances/{instance_id}/")

    def fetch_logs(self, instance_id, tail: str = "200") -> str:
        """Container stdout/stderr (for live download progress). Two-step: request a
        log dump, then poll the presigned URL (public, no auth) until it's ready."""
        try:
            data = self._put(f"/api/v0/instances/request_logs/{instance_id}/", {"tail": tail})
        except VastError:
            return ""
        url = data.get("result_url")
        if not url:
            return ""
        for _ in range(15):
            time.sleep(0.3)
            try:
                r = httpx.get(url, timeout=15)
            except httpx.HTTPError:
                continue
            if r.status_code == 200:
                return r.text
        return ""

    @staticmethod
    def endpoint_of(inst: dict, container_port: int = 8000) -> tuple[str, int] | None:
        """Extract (public_ip, external_port) once the port is mapped, else None."""
        ip = inst.get("public_ipaddr")
        ports = inst.get("ports") or {}
        mapping = ports.get(f"{container_port}/tcp")
        if ip and mapping:
            host_port = mapping[0].get("HostPort")
            if host_port:
                return str(ip).strip(), int(host_port)
        return None

    @staticmethod
    def status_of(inst: dict) -> str:
        return str(inst.get("actual_status") or inst.get("cur_state") or "unknown")

    # ------------------------------------------------------------------ #
    # Cost estimation: price each fitting GPU option from the sizing plan
    # ------------------------------------------------------------------ #

    def price_plan(
        self,
        plan: QuantPlan,
        min_disk_gb: float,
        max_price: float | None = None,
        max_candidates: int = 3,
        gpu_match: list[str] | None = None,
    ) -> list[PricedOption]:
        """Price the few least-wasteful GPU configs that fit and return them all
        (the caller picks the cheapest by actual price). We sort fitting configs by
        total VRAM ascending — least waste tends to be cheapest, and a couple of
        smaller GPUs is often cheaper than one big one — then price the top few.
        Bounded to `max_candidates` searches to stay under vast.ai's rate limit
        (the client also throttles + backs off on 429)."""
        # fp8 is only reliable on Ada+ (cc 8.9); on Ampere it yields garbage output.
        min_cc = 890 if plan.quant == "fp8" else 800
        # Scale the network-speed floor to the download size so big GGUF models don't
        # land on a slow node (343GB @ 200Mbps = ~3.8h of idle GPU billing).
        if plan.weights_gb > 150:
            min_inet = 2000
        elif plan.weights_gb > 40:
            min_inet = 1000
        else:
            min_inet = 400
        opts = sorted((o for o in plan.options if o.fits), key=lambda o: o.total_vram_gb)
        priced: list[PricedOption] = []
        for opt in opts[:max_candidates]:
            offers = self.search_offers(
                num_gpus=opt.num_gpus,
                min_gpu_ram_gb=opt.tier.vram_gb,
                min_disk_gb=min_disk_gb,
                max_price=max_price,
                min_compute_cap=min_cc,
                min_inet_mbps=min_inet,
                gpu_match=gpu_match,
                limit=10,
            )
            priced.append(PricedOption(option=opt, offer=offers[0] if offers else None))
        return priced

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        for attempt in range(4):
            gap = time.time() - self._last_req
            if gap < self._min_interval:
                time.sleep(self._min_interval - gap)
            try:
                r = self._client.request(method, path, json=json)
            except httpx.HTTPError as e:
                raise VastError(f"vast.ai request failed ({method} {path}): {e}") from e
            self._last_req = time.time()

            if r.status_code == 429 and attempt < 3:
                retry_after = 5.0
                try:
                    retry_after = float(r.json().get("retry_after", 5))
                except (ValueError, TypeError):
                    pass
                time.sleep(min(retry_after, 12))
                continue
            if r.status_code == 401:
                raise VastError("vast.ai rejected the API key (401). Check VAST_API_KEY.")
            if r.status_code >= 400:
                raise VastError(f"vast.ai {method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
            if not r.content:
                return {}
            try:
                return r.json()
            except ValueError:
                return {}
        raise VastError("vast.ai rate-limited (429) after retries — try again shortly.")

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, json: dict) -> dict:
        return self._request("POST", path, json=json)

    def _put(self, path: str, json: dict) -> dict:
        return self._request("PUT", path, json=json)

    def _delete(self, path: str) -> dict:
        return self._request("DELETE", path)


def recommend_disk_gb(weights_gb: float, floor: int = 40) -> int:
    """Disk needs to hold the weights download plus the image and scratch."""
    return max(floor, int(math.ceil(weights_gb * 1.3 + 25)))


# A download/progress line in vLLM or llama.cpp/HF stdout, e.g.
#   "model-00001-of-00005.gguf:  47%|####6     | 3.2G/6.8G [00:12<00:14, 257MB/s]"
_SIZE_HINT = ("b/s", "mb", "gb", "gib", "mib", "/s", "b/")


def extract_download_progress(log_text: str) -> str | None:
    """Pull the most recent download-progress line out of container logs.
    tqdm/HF bars overwrite with \\r, so the last segment is the freshest."""
    if not log_text:
        return None
    best: str | None = None
    for line in log_text.replace("\r", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        has_pct = "%" in line
        if (has_pct and any(h in low for h in _SIZE_HINT)) or ("download" in low and has_pct):
            best = line
    if best is None:
        return None
    best = re.sub(r"\s+", " ", best)
    return best[:110]
