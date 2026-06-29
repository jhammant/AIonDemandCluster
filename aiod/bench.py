"""A small benchmark for a running OpenAI-compatible endpoint.

Measures the numbers you actually want to report:
  * TTFT  — time to first token (streaming), p50/p95
  * decode speed — output tokens/sec per request
  * throughput   — aggregate output tokens/sec under concurrency
  * $/1M tokens  — derived from the instance price and throughput

Works against vLLM or llama.cpp (both stream OpenAI-style SSE).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

DEFAULT_PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that merges overlapping "
    "intervals and returns the merged list, with a short docstring."
)


@dataclass
class ReqResult:
    ok: bool
    ttft: float | None = None  # seconds to first token
    total: float | None = None  # total wall time
    completion_tokens: int = 0
    error: str | None = None

    @property
    def decode_tok_s(self) -> float | None:
        if self.ok and self.ttft is not None and self.total and self.completion_tokens:
            gen_time = max(self.total - self.ttft, 1e-6)
            return self.completion_tokens / gen_time
        return None


@dataclass
class BenchResult:
    n: int
    concurrency: int
    wall_time: float
    results: list[ReqResult] = field(default_factory=list)
    price_per_hr: float | None = None

    @property
    def ok(self) -> list[ReqResult]:
        return [r for r in self.results if r.ok]

    @property
    def ttft_p50(self) -> float | None:
        return _pct([r.ttft for r in self.ok if r.ttft is not None], 50)

    @property
    def ttft_p95(self) -> float | None:
        return _pct([r.ttft for r in self.ok if r.ttft is not None], 95)

    @property
    def avg_decode_tok_s(self) -> float | None:
        vals = [r.decode_tok_s for r in self.ok if r.decode_tok_s]
        return sum(vals) / len(vals) if vals else None

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.ok)

    @property
    def throughput_tok_s(self) -> float | None:
        if self.wall_time > 0 and self.total_completion_tokens:
            return self.total_completion_tokens / self.wall_time
        return None

    @property
    def cost_per_million(self) -> float | None:
        """$/1M output tokens at this throughput."""
        tps = self.throughput_tok_s
        if tps and self.price_per_hr:
            return (self.price_per_hr / 3600.0) / tps * 1_000_000
        return None


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100.0) * (len(s) - 1))))
    return s[k]


def _one_request(
    base_url: str, model: str, api_key: str | None, prompt: str, max_tokens: int, timeout: float
) -> ReqResult:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    start = time.time()
    ttft: float | None = None
    chunk_tokens = 0
    usage_tokens = 0
    try:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                r.read()
                return ReqResult(ok=False, error=f"HTTP {r.status_code}: {r.text[:160]}")
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    # reasoning models (GLM-5, DeepSeek-R1, ...) stream into
                    # reasoning_content; count it as generated work too.
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.time() - start
                        chunk_tokens += 1
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens", 0)
    except httpx.HTTPError as e:
        return ReqResult(ok=False, error=str(e))

    total = time.time() - start
    return ReqResult(
        ok=True, ttft=ttft, total=total, completion_tokens=usage_tokens or chunk_tokens
    )


def run_benchmark(
    base_url: str,
    model: str,
    api_key: str | None = None,
    n: int = 5,
    max_tokens: int = 256,
    concurrency: int = 1,
    prompt: str = DEFAULT_PROMPT,
    price_per_hr: float | None = None,
    timeout: float = 180.0,
) -> BenchResult:
    """Fire `n` requests `concurrency`-at-a-time and aggregate the stats."""
    results: list[ReqResult] = []
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, base_url, model, api_key, prompt, max_tokens, timeout)
            for _ in range(n)
        ]
        for f in futures:
            results.append(f.result())
    wall = time.time() - wall_start
    return BenchResult(
        n=n, concurrency=concurrency, wall_time=wall, results=results, price_per_hr=price_per_hr
    )
