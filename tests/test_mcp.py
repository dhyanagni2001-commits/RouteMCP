from pathlib import Path

import pytest

from routemcp.api import Container
from routemcp.domain import ModelConfig
from routemcp.mcp_server import create_mcp_server
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.telemetry import TelemetryStore

mcp = pytest.importorskip("mcp", reason="MCP extra requires Python 3.11+")


@pytest.mark.asyncio
async def test_mcp_server_exposes_router_tools(tmp_path: Path) -> None:
    model = ModelConfig(id="mock", provider="mock", model="mock")
    providers = ProviderFactory()
    telemetry = TelemetryStore(str(tmp_path / "mcp.db"))
    container = Container(ModelRegistry([model]), providers, telemetry)
    server = create_mcp_server(container)
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "route_decide",
        "route_complete",
        "list_routed_models",
    }
    await providers.close()
    telemetry.close()


@pytest.mark.asyncio
async def test_mcp_tools_route_and_report_learned_stats(tmp_path: Path) -> None:
    from mcp import Client

    from routemcp.mcp_server import BearerAuthMiddleware, create_mcp_app
    from routemcp.stats import StatsStore

    model = ModelConfig(id="mock", provider="mock", model="mock")
    providers = ProviderFactory()
    telemetry = TelemetryStore(str(tmp_path / "mcp.db"))
    container = Container(ModelRegistry([model]), providers, telemetry, StatsStore())
    messages = [{"role": "user", "content": "hello"}]
    async with Client(create_mcp_server(container)) as client:
        completed = await client.call_tool("route_complete", {"messages": messages})
        assert not completed.is_error
        assert completed.structured_content["result"]["_routemcp"]["selected_model"] == "mock"
        listed = await client.call_tool("list_routed_models", {})
        assert listed.structured_content["result"][0]["learned"]["samples"] == 1
    assert isinstance(create_mcp_app(container, api_key="secret"), BearerAuthMiddleware)
    assert not isinstance(create_mcp_app(container), BearerAuthMiddleware)
    await providers.close()
    telemetry.close()
