"""Pure-function translation between the Anthropic and OpenAI chat dialects.

This module is deliberately I/O-free and side-effect-free so the riskiest part of
the gateway (the stateful Anthropic streaming translator) can be unit-tested in
isolation. It is consumed by the opt-in ``/v1/messages`` endpoint in proxy.py.

Two directions:
  * ``anthropic_to_openai`` — request body translation (Anthropic -> OpenAI).
  * ``openai_to_anthropic`` / ``translate_stream`` — response translation
    (OpenAI -> Anthropic), non-streaming and streaming respectively.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

# OpenAI finish_reason -> Anthropic stop_reason.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# --------------------------------------------------------------------------- #
# Request translation: Anthropic -> OpenAI
# --------------------------------------------------------------------------- #

def _image_data_uri(source: dict) -> str:
    """Anthropic image source -> an OpenAI ``image_url`` data URI (or passthrough)."""
    if source.get("type") == "base64":
        media = source.get("media_type", "image/png")
        return f"data:{media};base64,{source.get('data', '')}"
    # url source (Anthropic also supports {type:url,url:...})
    return source.get("url", "")


def _flatten_tool_result(block: dict) -> str:
    """tool_result content (string | list of text blocks) -> a flat string."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text":
                parts.append(sub.get("text", ""))
            elif isinstance(sub, str):
                parts.append(sub)
        return "".join(parts)
    return json.dumps(content)


def _system_to_message(system) -> dict | None:
    """system (string | list of text blocks) -> a single OpenAI system message."""
    if not system:
        return None
    if isinstance(system, str):
        text = system
    else:
        text = "".join(
            b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
        )
    return {"role": "system", "content": text}


def _convert_user_content(content) -> tuple[object, list[dict]]:
    """Returns (openai_content, tool_messages) for a user-role Anthropic message.

    ``openai_content`` is a plain string when the message is text-only, otherwise a
    list of OpenAI content parts. ``tool_messages`` are emitted as separate
    ``role:tool`` messages (OpenAI carries tool results as their own messages).
    """
    if isinstance(content, str):
        return content, []
    parts: list[dict] = []
    tool_messages: list[dict] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            parts.append(
                {"type": "image_url", "image_url": {"url": _image_data_uri(block.get("source", {}))}}
            )
        elif btype == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _flatten_tool_result(block),
                }
            )
    if not parts:
        return None, tool_messages
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"], tool_messages
    return parts, tool_messages


def _convert_assistant_content(content) -> dict:
    """Returns a single OpenAI assistant message (text + optional tool_calls)."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    msg: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
        )
    return out


def _convert_tool_choice(choice):
    """auto -> auto, any -> required, {type:tool,name} -> forced function."""
    if not isinstance(choice, dict):
        return None
    ctype = choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "tool":
        return {"type": "function", "function": {"name": choice.get("name", "")}}
    return None


def anthropic_to_openai(body: dict) -> dict:
    """Translate a native Anthropic /v1/messages request into an OpenAI request."""
    messages: list[dict] = []

    sys_msg = _system_to_message(body.get("system"))
    if sys_msg is not None:
        messages.append(sys_msg)

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append(_convert_assistant_content(content))
        else:  # user (and any other role) carries text/image/tool_result
            oai_content, tool_messages = _convert_user_content(content)
            # tool results come first so they directly follow the assistant tool_calls
            messages.extend(tool_messages)
            if oai_content is not None:
                messages.append({"role": "user", "content": oai_content})

    out: dict = {"messages": messages}
    if body.get("model"):
        out["model"] = body["model"]
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("tools"):
        out["tools"] = _convert_tools(body["tools"])
    tc = _convert_tool_choice(body.get("tool_choice"))
    if tc is not None:
        out["tool_choice"] = tc
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# --------------------------------------------------------------------------- #
# Response translation: OpenAI -> Anthropic (non-streaming)
# --------------------------------------------------------------------------- #

def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def openai_to_anthropic(resp: dict, *, model: str, message_id: str | None = None) -> dict:
    """Translate a non-streaming OpenAI chat completion into an Anthropic message."""
    choices = resp.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message", {}) or {}

    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        content_blocks.append(
            {"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": args}
        )

    finish = choice.get("finish_reason")
    stop_reason = STOP_REASON_MAP.get(finish, "end_turn") if finish else "end_turn"

    usage = resp.get("usage") or {}
    return {
        "id": message_id or resp.get("id") or _new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
        },
    }


# --------------------------------------------------------------------------- #
# Response translation: OpenAI -> Anthropic (streaming)
# --------------------------------------------------------------------------- #

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def iter_openai_chunks(line_aiter) -> AsyncIterator[dict]:
    """Parse an httpx ``aiter_lines()`` stream into OpenAI chunk dicts.

    Line-buffered (so SSE events are never split across byte boundaries), tolerant
    of blank lines and comment lines, and terminates on the ``[DONE]`` sentinel.
    """
    async for line in line_aiter:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except (ValueError, TypeError):
            continue


async def translate_stream(
    chunks,
    *,
    model: str,
    message_id: str,
    input_tokens: int = 0,
    emit_message_start: bool = True,
    emit_message_stop: bool = True,
    start_index: int = 0,
) -> AsyncIterator[str]:
    """Stateful OpenAI-chunk -> Anthropic-SSE event machine.

    Emits at most one ``message_start`` (suppressible for the cold-start seam) and
    one terminating ``message_delta``/``message_stop``. Tracks a single open block
    (text or tool_use), accumulating ``tool_calls[].function.arguments`` fragments
    keyed by their OpenAI ``index`` and forwarding them as ``input_json_delta``
    without parsing. ``output_tokens`` is read from the trailing usage chunk.
    """
    if emit_message_start:
        yield _sse_event(
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

    next_index = start_index
    open_index: int | None = None
    open_kind: str | None = None  # "text" | "tool_use"
    tool_index_map: dict[int, int] = {}  # OpenAI tool_calls index -> Anthropic block index
    output_tokens = 0
    stop_reason = "end_turn"

    def _close_open() -> str | None:
        nonlocal open_index, open_kind
        if open_index is None:
            return None
        ev = _sse_event("content_block_stop", {"type": "content_block_stop", "index": open_index})
        open_index = None
        open_kind = None
        return ev

    async for chunk in chunks:
        usage = chunk.get("usage")
        if usage:
            output_tokens = usage.get("completion_tokens", output_tokens) or output_tokens

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            if open_kind != "text":
                closed = _close_open()
                if closed:
                    yield closed
                open_index = next_index
                open_kind = "text"
                next_index += 1
                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": open_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": open_index,
                    "delta": {"type": "text_delta", "text": content},
                },
            )

        for tc in delta.get("tool_calls") or []:
            oai_idx = tc.get("index", 0)
            fn = tc.get("function", {}) or {}
            if oai_idx not in tool_index_map:
                closed = _close_open()
                if closed:
                    yield closed
                open_index = next_index
                open_kind = "tool_use"
                tool_index_map[oai_idx] = open_index
                next_index += 1
                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": open_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": {},
                        },
                    },
                )
            args = fn.get("arguments")
            if args:
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_index_map[oai_idx],
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    },
                )

        finish = choice.get("finish_reason")
        if finish:
            stop_reason = STOP_REASON_MAP.get(finish, "end_turn")

    closed = _close_open()
    if closed:
        yield closed

    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    if emit_message_stop:
        yield _sse_event("message_stop", {"type": "message_stop"})
