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

