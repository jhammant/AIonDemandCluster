import asyncio

import httpx
from starlette.responses import JSONResponse, StreamingResponse
from starlette.testclient import TestClient

from aiod import proxy, state
from aiod.config import Settings


def _settings():
    return Settings(
        vast_api_key="k", hf_token=None, vllm_api_key="sk-x", ttl_hours=4, max_price=6.0
    )


def _manager(monkeypatch, *, enable_anthropic=False, require_auth=False, spinning=True):
    # No instance + already 'spinning' so ensure() never triggers a real launch.
    monkeypatch.setattr(state, "load", lambda: None)
    m = proxy.Manager(
        _settings(),
        {"model": "org/model", "quant": "fp8"},
        idle_minutes=20,
        enable_anthropic=enable_anthropic,
        require_auth=require_auth,
    )
    m.spinning = spinning
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


# --------------------------------------------------------------------------- #
# /healthz
# --------------------------------------------------------------------------- #

def test_healthz_reports_not_ready_when_no_instance(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch))
    with TestClient(app) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "up", "ready": False}


# --------------------------------------------------------------------------- #
# Synthesized /v1/models cold path
# --------------------------------------------------------------------------- #

def test_models_synthesized_without_spin(monkeypatch):
    m = _manager(monkeypatch, spinning=False)
    app = proxy.build_app(m)
    with TestClient(app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "org/model"
    assert body["data"][0]["owned_by"] == "aiod"
    # synthesized cold path must NOT kick a spin
    assert m.spinning is False


# --------------------------------------------------------------------------- #
# Auth gate
# --------------------------------------------------------------------------- #

def test_auth_loopback_open_allows_without_bearer(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, require_auth=False))
    with TestClient(app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200


def test_auth_required_rejects_missing_bearer_openai_shape(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, require_auth=True))
    with TestClient(app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_required_accepts_correct_bearer(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, require_auth=True))
    with TestClient(app) as client:
        r = client.get("/v1/models", headers={"Authorization": "Bearer sk-x"})
    assert r.status_code == 200


def test_auth_required_messages_anthropic_shape_and_x_api_key(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, enable_anthropic=True, require_auth=True))
    with TestClient(app) as client:
        missing = client.post("/v1/messages", json={"model": "org/model", "messages": []})
        assert missing.status_code == 401
        assert missing.json()["type"] == "error"
        assert missing.json()["error"]["type"] == "authentication_error"

        ok = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-x"},
            json={"model": "org/model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert ok.status_code == 200
        assert ok.json()["type"] == "message"


# --------------------------------------------------------------------------- #
# /v1/messages cold-start (opt-in)
# --------------------------------------------------------------------------- #

def test_messages_route_absent_without_anthropic(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, enable_anthropic=False))
    with TestClient(app) as client:
        r = client.post("/v1/messages", json={"model": "org/model", "messages": []})
    # /v1/messages falls through to the /v1/{path} catch-all, which only allows
    # GET/POST passthrough — POST to a cold backend returns the OpenAI warming shape.
    assert r.json().get("object") == "chat.completion"


def test_messages_cold_nonstream_returns_anthropic_message(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, enable_anthropic=True))
    with TestClient(app) as client:
        r = client.post(
            "/v1/messages",
            json={"model": "org/model", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"


# --------------------------------------------------------------------------- #
# _forward pre-first-byte re-resolve + retry
# --------------------------------------------------------------------------- #

class _FakeUpstream:
    def __init__(self):
        self.status_code = 200
        self.headers = {}

    async def aiter_raw(self):
        yield b"ok"

    async def aclose(self):
        pass


class _FakeClient:
    """build_request returns a sentinel; send raises on the first call then
    succeeds, so we can exercise _forward's single pre-first-byte retry."""

    def __init__(self, *, raises, succeeds_after=1):
        self.calls = 0
        self.raises = raises
        self.succeeds_after = succeeds_after

    def build_request(self, method, url, headers=None, content=None):
        return ("req", url)

    async def send(self, req, stream=True):
        self.calls += 1
        if self.calls <= self.succeeds_after and self.raises:
            raise httpx.ConnectError("boom")
        return _FakeUpstream()


def _inst(host, port):
    return state.Instance(
        instance_id=1, repo_id="org/model", quant="fp8", gpu_desc="1x H100",
        price_per_hr=1.0, created_at=0.0, ttl_hours=4, host=host, port=port, status="running",
    )


def test_forward_retries_on_changed_base_url(monkeypatch):
    m = _manager(monkeypatch)
    new = _inst("2.2.2.2", 2222)
    monkeypatch.setattr(m, "ready_instance", lambda: new)
    client = _FakeClient(raises=True, succeeds_after=1)
    old = _inst("1.1.1.1", 1111)
    resp = asyncio.run(_forward(m, client, old))
    assert isinstance(resp, StreamingResponse)
    assert client.calls == 2  # first raised, retry against the new base_url succeeded


def test_forward_502_when_base_url_unchanged(monkeypatch):
    m = _manager(monkeypatch)
    old = _inst("1.1.1.1", 1111)
    monkeypatch.setattr(m, "ready_instance", lambda: old)
    client = _FakeClient(raises=True, succeeds_after=99)
    resp = asyncio.run(_forward(m, client, old))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 502


def _forward(manager, client, inst):
    return proxy._forward(
        manager, client, inst, "POST", "chat/completions", {}, b"{}"
    )
