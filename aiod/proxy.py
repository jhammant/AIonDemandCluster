"""Auto-spin-up reverse proxy (non-blocking warm-up).

Point CCR at this instead of the box. Flow:
  * request arrives, an instance is running  -> forward (streaming) to the box,
    reset the idle timer.
  * request arrives, nothing running          -> kick off a background spin and
    immediately return a valid "warming up" chat completion that reports the
    current phase (so Claude Code shows progress instead of hanging/timing out).
  * no requests for `idle_minutes`            -> destroy the box.

Progress is also written to the shared events log, so `aiod status` / the TUI /
`GET /aiod/status` all show what's happening during the cold start.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import engine, events, state, translate
from .config import Settings

_DROP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection", "host"}

GATEWAY_FILE = state.STATE_DIR / "gateway.json"


def write_gateway_file(port: int, token: str) -> None:
    """Record the live gateway so `aiod up`/`aiod chat` (PR2/PR3) can discover and
    reuse a running gateway + its token without env plumbing."""
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    GATEWAY_FILE.write_text(json.dumps({"pid": os.getpid(), "port": port, "token": token}))


def read_gateway_file() -> dict | None:
    if not GATEWAY_FILE.exists():
        return None
    try:
        return json.loads(GATEWAY_FILE.read_text())
    except (ValueError, OSError):
        return None


def clear_gateway_file() -> None:
    GATEWAY_FILE.unlink(missing_ok=True)


def _wants_stream(body: bytes) -> bool:
    try:
        return bool(json.loads(body or b"{}").get("stream"))
    except (ValueError, TypeError):
        return False


def _model_of(body: bytes, default: str) -> str:
    try:
        return json.loads(body or b"{}").get("model") or default
    except (ValueError, TypeError):
        return default


def _filter_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_HEADERS}


class Manager:
    def __init__(
        self,
        settings: Settings,
        spin_kwargs: dict,
        idle_minutes: int | None,
        on_event=None,
        *,
        enable_anthropic: bool = False,
        require_auth: bool = False,
        token: str | None = None,
    ):
        self.s = settings
        self.spin_kwargs = spin_kwargs
        self.idle_minutes = idle_minutes
        self.on_event = on_event or (lambda *a: None)
        self.last_activity = time.time()
        self.spinning = False
        self._lock = asyncio.Lock()
        self.enable_anthropic = enable_anthropic
        self.require_auth = require_auth
        self.token = token if token is not None else settings.vllm_api_key

    def ready_instance(self) -> state.Instance | None:
        inst = state.load()
        if inst and inst.base_url and inst.status == "running":
            return inst
        return None

    def _should_spin(self) -> bool:
        """Spin when there's no running instance AND nothing is being created
        (state is None) OR the saved instance is wedged in 'error' — but never
        while a spin is already in flight or during a legitimate creating/loading
        window."""
        if self.spinning:
            return False
        if self.ready_instance() is not None:
            return False
        inst = state.load()
        return inst is None or inst.status == "error"

    async def ensure(self) -> state.Instance | None:
        inst = self.ready_instance()
        if inst:
            return inst
        async with self._lock:
            if self._should_spin():
                self.spinning = True
                asyncio.create_task(self._spin())
        return None

    async def _spin(self) -> None:
        try:
            await asyncio.to_thread(engine.launch, self.s, on_event=self.on_event, **self.spin_kwargs)
            self.last_activity = time.time()
        finally:
            self.spinning = False

    async def idle_monitor(self) -> None:
        while True:
            await asyncio.sleep(20)
            if not self.idle_minutes or self.spinning:
                continue
            inst = state.load()
            if inst and (time.time() - self.last_activity) / 60.0 >= self.idle_minutes:
                self.on_event("idle-destroy", f"idle {self.idle_minutes}m — destroying")
                await asyncio.to_thread(engine.destroy, self.s, inst)

    def warming_text(self) -> str:
        ev = events.latest()
        phase = ev["phase"] if ev else "starting"
        detail = ev["msg"] if ev else ""
        model = self.spin_kwargs.get("model", "the model")
        return (
            f"🔄 aiod is warming up **{model}** — current phase: **{phase}** {detail}\n\n"
            f"This can take a few minutes (renting a GPU + downloading weights). "
            f"Re-send your message shortly. Live progress: `aiod status`, the TUI, "
            f"or GET /aiod/status."
        )


def _warming_response(text: str, stream: bool, model: str) -> Response:
    created = int(time.time())
    cid = "chatcmpl-aiod-warming"
    if stream:
        async def gen():
            first = {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                             "finish_reason": None}],
            }
            done = {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    obj = {
        "id": cid, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(obj)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(cid: str, created: int, model: str, *, content=None, role=None, finish=None) -> dict:
    delta: dict = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _forward(
    manager: Manager, client: httpx.AsyncClient, inst, method, path, headers, body
) -> Response:
    base = inst.base_url
    try:
        req = client.build_request(
            method, f"{base.rstrip('/')}/{path}",
            headers=_filter_headers(headers), content=body or None,
        )
        up = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        # Endpoint churn: the box may have been replaced between resolve and send.
        # Re-resolve ONCE and retry — but only here, before any upstream byte has
        # been flushed; never inside the aiter loop (that would corrupt the stream).
        new_inst = manager.ready_instance()
        if new_inst is None or new_inst.base_url == base:
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
        try:
            req = client.build_request(
                method, f"{new_inst.base_url.rstrip('/')}/{path}",
                headers=_filter_headers(headers), content=body or None,
            )
            up = await client.send(req, stream=True)
        except httpx.HTTPError as e2:
            return JSONResponse({"error": f"upstream unreachable: {e2}"}, status_code=502)
    return StreamingResponse(
        up.aiter_raw(),
        status_code=up.status_code,
        headers=_filter_headers(up.headers),
        background=BackgroundTask(up.aclose),
    )


async def _warm_then_stream(manager: Manager, client: httpx.AsyncClient, method, path, headers, body, model):
    """Hold the stream open, stream warm-up progress into Claude Code, then relay
    the real upstream completion once the box is ready — all in one message."""
    cid, created = "chatcmpl-aiod-warmup", int(time.time())
    start = time.time()
    yield _sse(_chunk(cid, created, model, role="assistant", content=f"🔄 Warming up {model}…\n"))

    last_phase, last_beat = None, start
    while True:
        inst = manager.ready_instance()
        if inst:
            break
        ev = events.latest()
        phase = ev["phase"] if ev else "starting"
        now = time.time()
        if phase == "error":
            detail = ev["msg"] if ev else ""
            yield _sse(_chunk(cid, created, model, content=f"\n❌ Warm-up failed: {detail}\n"))
            yield _sse(_chunk(cid, created, model, finish="stop"))
            yield "data: [DONE]\n\n"
            return
        if phase != last_phase:
            yield _sse(_chunk(cid, created, model, content=f"  • {phase}: {ev['msg'] if ev else ''}\n"))
            last_phase, last_beat = phase, now
        elif now - last_beat >= 15:
            yield _sse(_chunk(cid, created, model, content=f"  …still {phase} ({int(now - start)}s)\n"))
            last_beat = now
        else:
            yield ": keepalive\n\n"  # keeps the connection warm without adding text
        await asyncio.sleep(2)

    manager.last_activity = time.time()
    yield _sse(_chunk(cid, created, model, content="  • ready ✓\n\n---\n\n"))
    inst = manager.ready_instance()
    try:
        req = client.build_request(
            method, f"{inst.base_url.rstrip('/')}/{path}",
            headers=_filter_headers(headers), content=body or None,
        )
        up = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        yield _sse(_chunk(cid, created, model, content=f"❌ upstream error: {e}"))
        yield _sse(_chunk(cid, created, model, finish="stop"))
        yield "data: [DONE]\n\n"
        return
    async for raw in up.aiter_raw():  # upstream carries its own deltas + [DONE]
        yield raw
    await up.aclose()


# --------------------------------------------------------------------------- #
# Auth + synthesized OpenAI /v1/models
# --------------------------------------------------------------------------- #

def _check_auth(request: Request, token: str, require: bool) -> JSONResponse | None:
    """Bearer gate. Loopback-default-OPEN: when ``require`` is False every request
    is allowed (preserving tokenless local proxy/tui flows). When enforced, accept
    ``Authorization: Bearer <token>`` or ``x-api-key: <token>``; otherwise 401 with
    an OpenAI-shaped body for /v1/* and an Anthropic-shaped body for /v1/messages."""
    if not require:
        return None
    provided = None
    authz = request.headers.get("authorization")
    if authz and authz.lower().startswith("bearer "):
        provided = authz[len("bearer "):].strip()
    if provided is None:
        provided = request.headers.get("x-api-key")
    if provided == token:
        return None
    if request.url.path.endswith("/v1/messages"):
        return JSONResponse(
            {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
            status_code=401,
        )
    return JSONResponse(
        {"error": {"message": "invalid api key", "type": "invalid_request_error", "code": "invalid_api_key"}},
        status_code=401,
    )


def _models_list(model: str) -> JSONResponse:
    return JSONResponse(
        {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "aiod"}]}
    )


# --------------------------------------------------------------------------- #
# Native Anthropic /v1/messages (opt-in)
# --------------------------------------------------------------------------- #

def _anthropic_warming(text: str, model: str, message_id: str) -> JSONResponse:
    """Non-streaming cold-start reply, shaped as a native Anthropic message."""
    return JSONResponse(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )


def _anth_message_start(model: str, message_id: str, input_tokens: int = 0) -> str:
    return translate._sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )


def _anth_text_block_start(index: int) -> str:
    return translate._sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
    )


def _anth_text_delta(index: int, text: str) -> str:
    return translate._sse_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
    )


def _anth_block_stop(index: int) -> str:
    return translate._sse_event("content_block_stop", {"type": "content_block_stop", "index": index})


def _anth_message_close(stop_reason: str = "end_turn", output_tokens: int = 0) -> list[str]:
    return [
        translate._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        translate._sse_event("message_stop", {"type": "message_stop"}),
    ]


async def _messages_nonstream_live(
    client: httpx.AsyncClient, token: str, inst, body_openai: dict, model: str, message_id: str
) -> Response:
    body_openai = dict(body_openai)
    body_openai["stream"] = False
    body_openai.pop("stream_options", None)
    url = f"{inst.base_url.rstrip('/')}/chat/completions"
    try:
        up = await client.post(url, json=body_openai, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": f"upstream unreachable: {e}"}},
            status_code=502,
        )
    # Surface an upstream failure as a real Anthropic error instead of letting an
    # error body (no "choices") translate into a blank, successful-looking 200.
    if up.status_code >= 400:
        return JSONResponse(
            {"type": "error",
             "error": {"type": "api_error", "message": f"upstream returned HTTP {up.status_code}"}},
            status_code=up.status_code,
        )
    data = up.json()
    return JSONResponse(translate.openai_to_anthropic(data, model=model, message_id=message_id))


async def _messages_stream_live(
    client: httpx.AsyncClient, token: str, inst, body_openai: dict, model: str, message_id: str
):
    body_openai = dict(body_openai)
    body_openai["stream"] = True
    body_openai.setdefault("stream_options", {"include_usage": True})
    url = f"{inst.base_url.rstrip('/')}/chat/completions"
    req = client.build_request(
        "POST", url, json=body_openai, headers={"Authorization": f"Bearer {token}"}
    )
    up = await client.send(req, stream=True)
    chunks = translate.iter_openai_chunks(up.aiter_lines())
    try:
        async for ev in translate.translate_stream(chunks, model=model, message_id=message_id):
            yield ev
    finally:
        await up.aclose()


async def _warm_then_messages(manager: Manager, client: httpx.AsyncClient, body_openai, model, message_id):
    """Anthropic cold-start: emit exactly ONE message_start + a warming text block
    (index 0), poll events like _warm_then_stream, then translate the real upstream
    stream with emit_message_start=False / start_index=1 so a single envelope wraps
    the whole response."""
    start = time.time()
    yield _anth_message_start(model, message_id)
    yield _anth_text_block_start(0)
    yield _anth_text_delta(0, f"🔄 Warming up {model}…\n")

    last_phase, last_beat = None, start
    while True:
        if manager.ready_instance():
            break
        ev = events.latest()
        phase = ev["phase"] if ev else "starting"
        now = time.time()
        if phase == "error":
            detail = ev["msg"] if ev else ""
            yield _anth_text_delta(0, f"\n❌ Warm-up failed: {detail}\n")
            yield _anth_block_stop(0)
            for ev_str in _anth_message_close("end_turn", 0):
                yield ev_str
            return
        if phase != last_phase:
            yield _anth_text_delta(0, f"  • {phase}: {ev['msg'] if ev else ''}\n")
            last_phase, last_beat = phase, now
        elif now - last_beat >= 15:
            yield _anth_text_delta(0, f"  …still {phase} ({int(now - start)}s)\n")
            last_beat = now
        else:
            yield ": keepalive\n\n"
        await asyncio.sleep(2)

    manager.last_activity = time.time()
    yield _anth_text_delta(0, "  • ready ✓\n")
    yield _anth_block_stop(0)

    inst = manager.ready_instance()
    body_openai = dict(body_openai)
    body_openai["stream"] = True
    body_openai.setdefault("stream_options", {"include_usage": True})
    url = f"{inst.base_url.rstrip('/')}/chat/completions"
    req = client.build_request(
        "POST", url, json=body_openai, headers={"Authorization": f"Bearer {manager.token}"}
    )
    up = await client.send(req, stream=True)
    chunks = translate.iter_openai_chunks(up.aiter_lines())
    try:
        async for ev_str in translate.translate_stream(
            chunks, model=model, message_id=message_id,
            emit_message_start=False, emit_message_stop=True, start_index=1,
        ):
            yield ev_str
    finally:
        await up.aclose()


def build_app(manager: Manager) -> Starlette:
    async def proxy_v1(request: Request) -> Response:
        auth = _check_auth(request, manager.token, manager.require_auth)
        if auth is not None:
            return auth
        path = request.path_params["path"]
        client: httpx.AsyncClient = request.app.state.http

        # Synthesized model list on the cold path so clients that probe /v1/models
        # before a box is live (e.g. OpenWebUI) succeed without kicking a spin.
        if request.method == "GET" and path == "models" and manager.ready_instance() is None:
            return _models_list(manager.spin_kwargs.get("model", "model"))

        body = await request.body()
        inst = manager.ready_instance()
        if inst is not None:
            manager.last_activity = time.time()
            return await _forward(
                manager, client, inst, request.method, path, dict(request.headers), body
            )

        # Cold: kick off the background spin.
        await manager.ensure()
        model = _model_of(body, manager.spin_kwargs.get("model", "model"))
        if _wants_stream(body):
            # Block-with-progress: stream warm-up details, then the real answer.
            return StreamingResponse(
                _warm_then_stream(
                    manager, client, request.method, path, dict(request.headers), body, model
                ),
                media_type="text/event-stream",
            )
        # Non-streaming requests can't show progress — return a 'warming' message.
        return _warming_response(manager.warming_text(), False, model)

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "up", "ready": manager.ready_instance() is not None})

    async def messages(request: Request) -> Response:
        auth = _check_auth(request, manager.token, manager.require_auth)
        if auth is not None:
            return auth
        raw = await request.body()
        try:
            body = json.loads(raw or b"{}")
        except (ValueError, TypeError):
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}},
                status_code=400,
            )
        client: httpx.AsyncClient = request.app.state.http
        body_openai = translate.anthropic_to_openai(body)
        model = body.get("model") or manager.spin_kwargs.get("model", "model")
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        stream = bool(body.get("stream"))

        inst = manager.ready_instance()
        if inst is not None:
            manager.last_activity = time.time()
            if stream:
                return StreamingResponse(
                    _messages_stream_live(client, manager.token, inst, body_openai, model, message_id),
                    media_type="text/event-stream",
                )
            return await _messages_nonstream_live(
                client, manager.token, inst, body_openai, model, message_id
            )

        # Cold: kick off the background spin.
        await manager.ensure()
        if stream:
            return StreamingResponse(
                _warm_then_messages(manager, client, body_openai, model, message_id),
                media_type="text/event-stream",
            )
        return _anthropic_warming(manager.warming_text(), model, message_id)

    async def aiod_status(request: Request) -> Response:
        inst = state.load()
        return JSONResponse(
            {
                "spinning": manager.spinning,
                "idle_minutes": manager.idle_minutes,
                "idle_seconds": round(time.time() - manager.last_activity),
                "instance": asdict(inst) if inst else None,
                "events": events.read(15),
            }
        )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(None))
        idle_task = asyncio.create_task(manager.idle_monitor())
        try:
            yield
        finally:
            idle_task.cancel()
            await app.state.http.aclose()

    routes = [
        Route("/aiod/status", aiod_status, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
    ]
    # /v1/messages must be registered BEFORE the /v1/{path:path} catch-all so it
    # isn't swallowed by the OpenAI passthrough. Only mounted when opted in.
    if manager.enable_anthropic:
        routes.append(Route("/v1/messages", messages, methods=["POST"]))
    routes.append(Route("/v1/{path:path}", proxy_v1, methods=["GET", "POST"]))

    app = Starlette(routes=routes, lifespan=lifespan)
    return app


def run_gateway(
    settings: Settings,
    spin_kwargs: dict,
    idle_minutes: int | None,
    *,
    host: str = "127.0.0.1",
    port: int = 4000,
    enable_anthropic: bool = False,
    require_auth: bool = False,
    on_event=None,
) -> None:
    """Run the always-on local gateway. The canonical entrypoint; `aiod proxy`
    delegates here with the Anthropic endpoint + auth disabled for back-compat."""
    import uvicorn

    manager = Manager(
        settings, spin_kwargs, idle_minutes, on_event=on_event,
        enable_anthropic=enable_anthropic, require_auth=require_auth,
    )
    write_gateway_file(port, manager.token)
    try:
        uvicorn.run(build_app(manager), host=host, port=port, log_level="warning")
    finally:
        clear_gateway_file()


def run_proxy(
    settings: Settings,
    spin_kwargs: dict,
    idle_minutes: int | None,
    host: str = "127.0.0.1",
    port: int = 4000,
    on_event=None,
) -> None:
    """Back-compat thin wrapper: delegates to run_gateway with the Anthropic
    endpoint and auth gate disabled (byte-for-byte the old proxy behavior)."""
    run_gateway(
        settings, spin_kwargs, idle_minutes,
        host=host, port=port, enable_anthropic=False, require_auth=False, on_event=on_event,
    )
