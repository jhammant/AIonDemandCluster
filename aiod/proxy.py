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
import time
from dataclasses import asdict

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import engine, events, state
from .config import Settings

_DROP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection", "host"}


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
    def __init__(self, settings: Settings, spin_kwargs: dict, idle_minutes: int | None, on_event=None):
        self.s = settings
        self.spin_kwargs = spin_kwargs
        self.idle_minutes = idle_minutes
        self.on_event = on_event or (lambda *a: None)
        self.last_activity = time.time()
        self.spinning = False
        self._lock = asyncio.Lock()

    def ready_instance(self) -> state.Instance | None:
        inst = state.load()
        if inst and inst.base_url and inst.status == "running":
            return inst
        return None

    async def ensure(self) -> state.Instance | None:
        inst = self.ready_instance()
        if inst:
            return inst
        async with self._lock:
            if not self.spinning and state.load() is None:
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


async def _forward(client: httpx.AsyncClient, inst, method, path, headers, body) -> Response:
    url = f"{inst.base_url.rstrip('/')}/{path}"
    try:
        req = client.build_request(method, url, headers=_filter_headers(headers), content=body or None)
        up = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
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


def build_app(manager: Manager) -> Starlette:
    async def proxy_v1(request: Request) -> Response:
        path = request.path_params["path"]
        body = await request.body()
        client: httpx.AsyncClient = request.app.state.http

        inst = manager.ready_instance()
        if inst is not None:
            manager.last_activity = time.time()
            return await _forward(client, inst, request.method, path, dict(request.headers), body)

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

    app = Starlette(
        routes=[
            Route("/aiod/status", aiod_status, methods=["GET"]),
            Route("/v1/{path:path}", proxy_v1, methods=["GET", "POST"]),
        ],
        lifespan=lifespan,
    )
    return app


def run_proxy(
    settings: Settings,
    spin_kwargs: dict,
    idle_minutes: int | None,
    host: str = "127.0.0.1",
    port: int = 4000,
    on_event=None,
) -> None:
    import uvicorn

    manager = Manager(settings, spin_kwargs, idle_minutes, on_event=on_event)
    uvicorn.run(build_app(manager), host=host, port=port, log_level="warning")
