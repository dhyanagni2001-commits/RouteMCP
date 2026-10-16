import json

import httpx
import pytest

from routemcp.domain import ChatRequest, Message, ModelConfig
from routemcp.providers import AnthropicProvider


@pytest.mark.asyncio
async def test_anthropic_tool_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"q": "x"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = ModelConfig(
        id="anthropic",
        provider="anthropic",
        model="test-model",
        api_key_env="ANTHROPIC_API_KEY",
    )
    provider = AnthropicProvider(config, client)
    result = await provider.complete(
        ChatRequest(
            messages=[Message(role="user", content="look it up")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        )
    )
    assert captured["tools"][0]["input_schema"]["type"] == "object"
    choice = result.payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    await client.aclose()



def anthropic_sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


async def collect_stream(body: str, request: ChatRequest) -> list:
    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert payload["stream"] is True
        assert http_request.headers["x-api-key"] == "test-key"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = ModelConfig(
        id="anthropic", provider="anthropic", model="test-model", api_key_env="ANTHROPIC_API_KEY"
    )
    stream = await AnthropicProvider(config, client).start_stream(request)
    raw = b"".join([chunk async for chunk in stream.iterator]).decode()
    await stream.close()
    await client.aclose()
    assert raw.endswith("data: [DONE]\n\n")
    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
async def test_anthropic_stream_translates_text_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    body = anthropic_sse(
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": '}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '"x"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    )
    chunks = await collect_stream(body, ChatRequest(messages=[Message(role="user", content="hi")]))
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert {chunk["id"] for chunk in chunks} == {"msg_1"}
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert "".join(d.get("content", "") for d in deltas) == "Hello"
    tool_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert tool_deltas[0]["id"] == "toolu_1" and tool_deltas[0]["function"]["name"] == "lookup"
    assert "".join(t["function"]["arguments"] for t in tool_deltas) == '{"q": "x"}'
    assert {t["index"] for t in tool_deltas} == {0}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_anthropic_stream_error_event_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from routemcp.providers import ProviderError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    body = anthropic_sse(
        {"type": "message_start", "message": {"id": "msg_2"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    with pytest.raises(ProviderError, match="Overloaded") as raised:
        await collect_stream(body, ChatRequest(messages=[Message(role="user", content="hi")]))
    assert raised.value.error_type == "overloaded_error"


@pytest.mark.asyncio
async def test_anthropic_stream_http_error_is_typed_before_first_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from routemcp.providers import ProviderError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(529, json={"error": "overloaded"}))
    )
    config = ModelConfig(
        id="anthropic", provider="anthropic", model="m", api_key_env="ANTHROPIC_API_KEY"
    )
    with pytest.raises(ProviderError) as raised:
        await AnthropicProvider(config, client).start_stream(
            ChatRequest(messages=[Message(role="user", content="hi")])
        )
    assert raised.value.error_type == "provider_unavailable" and raised.value.retryable
    await client.aclose()
