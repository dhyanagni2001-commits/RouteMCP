from __future__ import annotations

from typing import Any, Dict, List, Optional

from .api import Container
from .domain import ChatRequest, Message, RouteOptions


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
    ) -> Dict[str, Any]:
        """Route and execute a chat completion with validation and automatic failover."""
        request = ChatRequest(
            messages=[Message.model_validate(message) for message in messages],
            tools=tools,
            response_format=response_format,
            max_tokens=max_tokens,
            route=RouteOptions(
                profile=profile,
                max_cost_usd=max_cost_usd,
                privacy=privacy,
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
            }
            for model in container.registry.list()
        ]

    return server
