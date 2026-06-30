"""Unit tests for the pure Anthropic<->OpenAI translation module."""

import asyncio
import json

from aiod import translate

# --------------------------------------------------------------------------- #
# Async helpers
# --------------------------------------------------------------------------- #

async def _aiter(items):
    for it in items:
        yield it


def _collect(agen):
    async def run():
        return [x async for x in agen]

    return asyncio.run(run())


def _events(sse_strings):
    """Parse a list of SSE event strings into (event_type, data_dict) tuples,
    skipping comment/keepalive lines."""
    out = []
    for s in sse_strings:
        if not s.startswith("event:"):
            continue
        lines = s.splitlines()
        etype = lines[0].split(":", 1)[1].strip()
        data = json.loads(lines[1].split(":", 1)[1].strip())
        out.append((etype, data))
    return out


# --------------------------------------------------------------------------- #
# anthropic_to_openai
# --------------------------------------------------------------------------- #

def test_system_string_prepends_one_system_message():
    out = translate.anthropic_to_openai(
        {"system": "be terse", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert out["messages"][0] == {"role": "system", "content": "be terse"}
    assert sum(1 for m in out["messages"] if m["role"] == "system") == 1


def test_system_blocks_prepend_one_system_message():
    out = translate.anthropic_to_openai(
        {
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    systems = [m for m in out["messages"] if m["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"] == "ab"


def test_tool_use_assistant_maps_to_tool_calls():
    out = translate.anthropic_to_openai(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                }
            ]
        }
    )
    msg = out["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "let me check"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "tu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}


def test_tool_result_user_maps_to_role_tool():
    out = translate.anthropic_to_openai(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "72F"},
                        {"type": "text", "text": "thanks"},
                    ],
                }
            ]
        }
    )
    tool_msg = out["messages"][0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_1"
    assert tool_msg["content"] == "72F"
    # remaining text becomes a user message after the tool message
    assert out["messages"][1] == {"role": "user", "content": "thanks"}


def test_tool_result_list_content_flattened():
    out = translate.anthropic_to_openai(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_9",
                            "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
                        }
                    ],
                }
            ]
        }
    )
    assert out["messages"][0]["content"] == "part1part2"


def test_tools_and_tool_choice_mapping():
    out = translate.anthropic_to_openai(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
    )
    tool = out["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["description"] == "Get weather"
    assert tool["function"]["parameters"]["type"] == "object"
    assert out["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_tool_choice_auto_and_any():
    auto = translate.anthropic_to_openai(
        {"messages": [], "tool_choice": {"type": "auto"}}
    )
    assert auto["tool_choice"] == "auto"
    any_ = translate.anthropic_to_openai(
        {"messages": [], "tool_choice": {"type": "any"}}
    )
    assert any_["tool_choice"] == "required"


def test_image_block_to_data_uri():
    out = translate.anthropic_to_openai(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                        },
                    ],
                }
            ]
        }
    )
    parts = out["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_max_tokens_and_stop_sequences_mapping():
    out = translate.anthropic_to_openai(
        {"messages": [], "max_tokens": 128, "stop_sequences": ["STOP"]}
    )
    assert out["max_tokens"] == 128
    assert out["stop"] == ["STOP"]


def test_stream_options_injected_only_when_stream_true():
    streamed = translate.anthropic_to_openai({"messages": [], "stream": True})
    assert streamed["stream"] is True
    assert streamed["stream_options"] == {"include_usage": True}

    non_streamed = translate.anthropic_to_openai({"messages": [], "stream": False})
    assert "stream" not in non_streamed
    assert "stream_options" not in non_streamed


# --------------------------------------------------------------------------- #
# openai_to_anthropic (non-stream)
# --------------------------------------------------------------------------- #

def test_openai_to_anthropic_text_and_usage():
    resp = {
        "id": "cmpl-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
    }
    out = translate.openai_to_anthropic(resp, model="m", message_id="msg_x")
    assert out["id"] == "msg_x"
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "m"
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 11, "output_tokens": 5}


def test_openai_to_anthropic_tool_calls_parses_arguments():
    resp = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7},
    }
    out = translate.openai_to_anthropic(resp, model="m")
    assert out["stop_reason"] == "tool_use"
    block = out["content"][0]
    assert block == {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}}


def test_openai_to_anthropic_finish_reason_length():
    resp = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]}
    out = translate.openai_to_anthropic(resp, model="m")
    assert out["stop_reason"] == "max_tokens"


# --------------------------------------------------------------------------- #
# iter_openai_chunks
# --------------------------------------------------------------------------- #

def test_iter_openai_chunks_tolerates_blanks_and_done():
    lines = [
        "",
        ": comment",
        'data: {"a": 1}',
        "",
        'data: {"b": 2}',
        "data: [DONE]",
        'data: {"c": 3}',  # after DONE -> must be ignored
    ]
    chunks = _collect(translate.iter_openai_chunks(_aiter(lines)))
    assert chunks == [{"a": 1}, {"b": 2}]


def test_iter_openai_chunks_handles_bytes():
    lines = [b'data: {"a": 1}', b"data: [DONE]"]
    chunks = _collect(translate.iter_openai_chunks(_aiter(lines)))
    assert chunks == [{"a": 1}]


# --------------------------------------------------------------------------- #
# translate_stream
# --------------------------------------------------------------------------- #

def _chunk(content=None, finish=None, usage=None, role=None, tool_calls=None):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    ch = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        ch["usage"] = usage
        ch["choices"] = []
    return ch


def test_translate_stream_text_single_envelope_and_usage():
    chunks = [
        _chunk(role="assistant", content=""),
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish="stop"),
        _chunk(usage={"prompt_tokens": 4, "completion_tokens": 9}),
    ]
    evs = _events(
        _collect(translate.translate_stream(_aiter(chunks), model="m", message_id="msg_1"))
    )
    types = [e[0] for e in evs]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert types.count("message_start") == 1
    assert types.count("message_stop") == 1
    # text deltas
    deltas = [e[1]["delta"]["text"] for e in evs if e[0] == "content_block_delta"]
    assert deltas == ["Hel", "lo"]
    # stop_reason + output_tokens from trailing usage chunk
    md = next(e[1] for e in evs if e[0] == "message_delta")
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["output_tokens"] == 9
    # text block opened at index 0
    cbs = next(e[1] for e in evs if e[0] == "content_block_start")
    assert cbs["index"] == 0
    assert cbs["content_block"]["type"] == "text"


def test_translate_stream_tool_calls_input_json_delta():
    chunks = [
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": ""},
                }
            ]
        ),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"q":'}}]),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": ' "x"}'}}]),
        _chunk(finish="tool_calls"),
        _chunk(usage={"completion_tokens": 3}),
    ]
    evs = _events(
        _collect(translate.translate_stream(_aiter(chunks), model="m", message_id="msg_2"))
    )
    types = [e[0] for e in evs]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    start = next(e[1] for e in evs if e[0] == "content_block_start")
    assert start["content_block"] == {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}
    json_deltas = [
        e[1]["delta"]["partial_json"]
        for e in evs
        if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "input_json_delta"
    ]
    assert json_deltas == ['{"q":', ' "x"}']
    md = next(e[1] for e in evs if e[0] == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use"
    assert md["usage"]["output_tokens"] == 3


def test_translate_stream_text_then_tool_closes_first_block():
    chunks = [
        _chunk(content="thinking"),
        _chunk(
            tool_calls=[
                {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]
        ),
        _chunk(finish="tool_calls"),
    ]
    evs = _events(
        _collect(translate.translate_stream(_aiter(chunks), model="m", message_id="msg_3"))
    )
    # text block index 0, tool_use block index 1
    starts = [e[1] for e in evs if e[0] == "content_block_start"]
    assert starts[0]["index"] == 0 and starts[0]["content_block"]["type"] == "text"
    assert starts[1]["index"] == 1 and starts[1]["content_block"]["type"] == "tool_use"
    stops = [e[1]["index"] for e in evs if e[0] == "content_block_stop"]
    assert stops == [0, 1]


def test_translate_stream_cold_envelope_no_message_start_start_index_1():
    chunks = [
        _chunk(content="hi"),
        _chunk(finish="stop"),
    ]
    evs = _events(
        _collect(
            translate.translate_stream(
                _aiter(chunks),
                model="m",
                message_id="msg_4",
                emit_message_start=False,
                emit_message_stop=True,
                start_index=1,
            )
        )
    )
    types = [e[0] for e in evs]
    assert "message_start" not in types
    assert types[-1] == "message_stop"
    cbs = next(e[1] for e in evs if e[0] == "content_block_start")
    assert cbs["index"] == 1
