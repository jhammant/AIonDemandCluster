"""Poll a vLLM endpoint until the model is loaded and serving.

vLLM exposes an OpenAI-compatible API. We treat the box as healthy once
`GET /v1/models` returns 200 with the model listed. Before the container has
even bound the port we'll get connection errors; while weights download we may
get connection-refused or 503 — both are 'keep waiting', not 'failed'.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class HealthState:
    reachable: bool  # TCP/HTTP responded at all
    ready: bool  # /v1/models returned 200 with a model
    detail: str


def check_once(base_url: str, api_key: str | None = None, timeout: float = 8.0) -> HealthState:
    """One health probe. `base_url` ends in /v1."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{base_url.rstrip('/')}/models"
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        return HealthState(reachable=False, ready=False, detail="not reachable yet")
    except httpx.HTTPError as e:
        return HealthState(reachable=False, ready=False, detail=f"http error: {e}")

    if r.status_code == 200:
        try:
            models = [m.get("id") for m in r.json().get("data", [])]
        except Exception:
            models = []
        return HealthState(reachable=True, ready=True, detail=f"serving: {', '.join(models) or '?'}")
    if r.status_code in (401, 403):
        return HealthState(reachable=True, ready=False, detail="auth rejected — check endpoint token")
    if r.status_code == 503:
        return HealthState(reachable=True, ready=False, detail="loading model (503)")
    return HealthState(reachable=True, ready=False, detail=f"HTTP {r.status_code}")


def sample_completion(
    base_url: str,
    model: str,
    api_key: str | None = None,
    with_tool: bool = False,
    timeout: float = 60.0,
) -> dict:
    """Send one chat completion to confirm the model actually serves. Returns
    {'ok', 'text', 'tool_call', 'latency_s', 'error'}."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    if with_tool:
        payload["messages"] = [
            {"role": "user", "content": "What's the weather in Paris? Use the tool."}
        ]
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        payload["max_tokens"] = 128

    url = f"{base_url.rstrip('/')}/chat/completions"
    start = time.time()
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        return {"ok": False, "error": str(e), "latency_s": time.time() - start}
    latency = time.time() - start
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}", "latency_s": latency}

    try:
        msg = r.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as e:
        return {"ok": False, "error": f"bad response shape: {e}", "latency_s": latency}

    tool_call = None
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        tool_call = f"{fn.get('name')}({fn.get('arguments')})"
        break

    return {
        "ok": True,
        "text": (msg.get("content") or "").strip(),
        "tool_call": tool_call,
        "latency_s": latency,
        "error": None,
    }


def wait_until_ready(
    base_url: str,
    api_key: str | None = None,
    timeout_s: float = 1800.0,
    interval_s: float = 10.0,
    on_progress=None,
) -> bool:
    """Block until ready or timeout. Calls on_progress(HealthState, elapsed) each
    poll if provided. Returns True if the model became ready."""
    start = time.time()
    while True:
        st = check_once(base_url, api_key=api_key)
        elapsed = time.time() - start
        if on_progress:
            on_progress(st, elapsed)
        if st.ready:
            return True
        if elapsed >= timeout_s:
            return False
        time.sleep(interval_s)
