import httpx
import pytest

from routemcp.mcp_server import BearerAuthMiddleware, mcp_api_key


async def ok_app(scope, receive, send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
async def test_bearer_middleware_rejects_missing_and_wrong_keys() -> None:
    app = BearerAuthMiddleware(ok_app, "secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/mcp")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"].startswith("Bearer")
        wrong = await client.post("/mcp", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        allowed = await client.post("/mcp", headers={"Authorization": "Bearer secret"})
        assert allowed.status_code == 200 and allowed.text == "ok"


def test_mcp_key_prefers_dedicated_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEMCP_MCP_API_KEY", raising=False)
    monkeypatch.delenv("ROUTEMCP_API_KEY", raising=False)
    assert mcp_api_key() is None
    monkeypatch.setenv("ROUTEMCP_API_KEY", "gateway")
    assert mcp_api_key() == "gateway"
    monkeypatch.setenv("ROUTEMCP_MCP_API_KEY", "mcp")
    assert mcp_api_key() == "mcp"
