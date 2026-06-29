import asyncio

from starlette.testclient import TestClient

from aiod import proxy, state
from aiod.config import Settings


def _manager(monkeypatch):
    # No instance + already 'spinning' so ensure() never triggers a real launch.
    monkeypatch.setattr(state, "load", lambda: None)
    s = Settings(
        vast_api_key="k", hf_token=None, vllm_api_key="sk-x", ttl_hours=4, max_price=6.0
    )
    m = proxy.Manager(s, {"model": "org/model", "quant": "fp8"}, idle_minutes=20)
    m.spinning = True
    return m


def test_warming_json_on_cold_nonstream(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch))
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json={"model": "org/model", "messages": []})
    assert r.status_code == 200
    obj = r.json()
    assert obj["object"] == "chat.completion"
    assert "warming up" in obj["choices"][0]["message"]["content"].lower()


def test_warming_response_stream_format():
    # The streaming warm-up SSE format (finite: role+content, stop, [DONE]).
    resp = proxy._warming_response("hold on, warming up", True, "org/model")

    async def drain():
        return [c async for c in resp.body_iterator]

    text = "".join(
        c if isinstance(c, str) else c.decode() for c in asyncio.run(drain())
    )
    assert "chat.completion.chunk" in text
    assert "warming up" in text.lower()
    assert "[DONE]" in text


def test_chunk_and_sse_helpers():
    ch = proxy._chunk("id1", 123, "m", content="hi", role="assistant")
    assert ch["choices"][0]["delta"] == {"role": "assistant", "content": "hi"}
    assert proxy._sse({"a": 1}) == 'data: {"a": 1}\n\n'


def test_aiod_status_endpoint(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch))
    with TestClient(app) as client:
        r = client.get("/aiod/status")
    assert r.status_code == 200
    body = r.json()
    assert body["spinning"] is True
    assert body["idle_minutes"] == 20
    assert body["instance"] is None
