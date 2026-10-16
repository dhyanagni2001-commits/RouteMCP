from pathlib import Path

import httpx
import pytest

from routemcp.api import Container, create_app
from routemcp.domain import ModelConfig
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.telemetry import TelemetryStore


def mock_model() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "id": "mock",
            "provider": "mock",
            "model": "mock",
            "local": True,
            "capabilities": {"tools": True, "vision": True},
            "reliability": {"quality": 0.9, "median_latency_ms": 5},
        }
    )


@pytest.mark.asyncio
async def test_api_end_to_end(tmp_path: Path) -> None:
    providers = ProviderFactory()
    telemetry = TelemetryStore(str(tmp_path / "api.db"))
    container = Container(ModelRegistry([mock_model()]), providers, telemetry)
    transport = httpx.ASGITransport(app=create_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/healthz")).json() == {"status": "ok", "models": 1}
        decision = await client.post(
            "/v1/router/decide", json={"messages": [{"role": "user", "content": "hello"}]}
        )
        assert decision.status_code == 200
        assert decision.json()["selected_model"] == "mock"
        completion = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}
        )
        assert completion.status_code == 200
        assert completion.json()["_routemcp"]["selected_model"] == "mock"
    await providers.close()
    telemetry.close()


@pytest.mark.asyncio
async def test_update_requires_existing_model(tmp_path: Path) -> None:
    providers = ProviderFactory()
    telemetry = TelemetryStore(str(tmp_path / "api.db"))
    container = Container(ModelRegistry(), providers, telemetry)
    transport = httpx.ASGITransport(app=create_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            "/v1/router/models/missing", json={"id": "missing", "provider": "mock", "model": "mock"}
        )
        assert response.status_code == 404
    await providers.close()
    telemetry.close()


@pytest.mark.asyncio
async def test_probe_and_workflow(tmp_path: Path) -> None:
    providers = ProviderFactory()
    telemetry = TelemetryStore(str(tmp_path / "api.db"))
    container = Container(ModelRegistry([mock_model()]), providers, telemetry)
    transport = httpx.ASGITransport(app=create_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        probe = await client.post("/v1/router/models/mock/probe")
        assert probe.status_code == 200
        assert probe.json()["healthy"] is True
        workflow = await client.post(
            "/v1/router/workflows",
            json={
                "steps": [
                    {"id": "plan", "request": {"messages": [{"role": "user", "content": "plan"}]}},
                    {"id": "format", "request": {"messages": [{"role": "user", "content": "format"}]}},
                ]
            },
        )
        assert workflow.status_code == 200
        assert [item["step_id"] for item in workflow.json()["steps"]] == ["plan", "format"]
    await providers.close()
    telemetry.close()
