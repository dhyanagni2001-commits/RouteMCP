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
