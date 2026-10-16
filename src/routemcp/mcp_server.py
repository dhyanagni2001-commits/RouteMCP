from __future__ import annotations

import hmac
import json
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .api import Container, learned
from .domain import ChatRequest, Message, RouteOptions

ASGIApp = Callable[..., Awaitable[None]]


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <key>`` on every HTTP request to the MCP app."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.api_key = api_key.encode()

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            authorization = headers.get(b"authorization", b"")
            supplied = authorization[7:] if authorization.startswith(b"Bearer ") else b""
            if not hmac.compare_digest(supplied, self.api_key):
                body = json.dumps({"error": "invalid or missing RouteMCP API key"}).encode()
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"www-authenticate", b'Bearer realm="routemcp"'),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def mcp_api_key() -> Optional[str]:
    return os.getenv("ROUTEMCP_MCP_API_KEY") or os.getenv("ROUTEMCP_API_KEY") or None


def create_mcp_server(container: Container) -> Any:
    """Expose RouteMCP routing and completion as genuine MCP tools."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "MCP support requires Python 3.11+ and `pip install 'routemcp[mcp]'`"
        ) from exc

    server = MCPServer(
        "RouteMCP",
        instructions=(
            "Inspect and execute capability-aware LLM routing across the models "
            "registered with RouteMCP. Use route_decide before route_complete when "
            "the caller needs a transparent explanation of model selection."
        ),
    )

    @server.tool()
    async def route_decide(
        messages: List[Dict[str, Any]],
        profile: str = "balanced",
        max_cost_usd: Optional[float] = None,
        privacy: str = "standard",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze a request and return the selected model, scores, and rejection reasons."""
        request = ChatRequest(
            messages=[Message.model_validate(message) for message in messages],
            tools=tools,
            route=RouteOptions(
                profile=profile,
                max_cost_usd=max_cost_usd,
                privacy=privacy,
            ),
        )
        return container.router.decision(request).model_dump(mode="json")

    @server.tool()
    async def route_complete(
        messages: List[Dict[str, Any]],
        profile: str = "balanced",
        max_cost_usd: Optional[float] = None,
        privacy: str = "standard",
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        cascade: bool = False,
    ) -> Dict[str, Any]:
        """Route and execute a chat completion with validation and automatic failover.

        With ``cascade``, cheaper models answer first and their answer is kept when
        it is confident; otherwise the request escalates to the routed model.
        """
        request = ChatRequest(
            messages=[Message.model_validate(message) for message in messages],
            tools=tools,
            response_format=response_format,
            max_tokens=max_tokens,
            route=RouteOptions(
                profile=profile,
                max_cost_usd=max_cost_usd,
                privacy=privacy,
                cascade=cascade,
            ),
        )
        return await container.router.execute(request)

    @server.tool()
    async def list_routed_models() -> List[Dict[str, Any]]:
        """List registered models with capabilities, prices, and runtime health."""
        return [
            {
                **model.model_dump(mode="json", by_alias=True, exclude={"headers"}),
                "health": container.registry.health(model.id),
                "learned": learned(container, model.id),
            }
            for model in container.registry.list()
        ]

    return server


def create_mcp_app(
    container: Container,
    api_key: Optional[str] = None,
    host: str = "127.0.0.1",
    path: str = "/mcp",
) -> ASGIApp:
    """Build the Streamable HTTP ASGI app, wrapped in bearer auth when a key is set."""
    app = create_mcp_server(container).streamable_http_app(streamable_http_path=path, host=host)
    return BearerAuthMiddleware(app, api_key) if api_key else app


def serve_mcp(container: Container, host: str, port: int, api_key: Optional[str] = None) -> None:
    import uvicorn

    uvicorn.run(create_mcp_app(container, api_key, host), host=host, port=port)
