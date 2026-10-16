from pathlib import Path

import pytest
from pydantic import ValidationError

from routemcp.analyzer import analyze
from routemcp.domain import ChatRequest, Message, ModelConfig, ProviderKind, RouteOptions
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.router import Router, RoutingError
from routemcp.telemetry import TelemetryStore


def model(model_id: str, *, quality: float, latency: float, tools: bool = False) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "id": model_id,
            "provider": "mock",
            "model": model_id,
            "local": True,
            "cost": {"input_per_million": quality * 10, "output_per_million": quality * 20},
            "capabilities": {"tools": tools, "context_window": 32000, "max_output_tokens": 4096},
            "reliability": {"quality": quality, "median_latency_ms": latency},
        }
    )


def test_provider_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="base_url is required"):
        ModelConfig(id="remote", model="m", provider=ProviderKind.OPENAI_COMPATIBLE)
    with pytest.raises(ValidationError, match="api_key_env is required"):
        ModelConfig(id="openai", model="m", provider=ProviderKind.OPENAI)


def test_analyzer_detects_tools_vision_and_json() -> None:
    request = ChatRequest(
        messages=[Message(role="user", content=[{"type": "image_url", "image_url": {"url": "x"}}])],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        response_format={"type": "json_object"},
    )
    features = analyze(request)
    assert features.requires_tools
    assert features.requires_vision
    assert features.requires_json


def test_routing_profiles_and_constraints(tmp_path: Path) -> None:
    registry = ModelRegistry(
        [model("fast", quality=0.55, latency=20), model("quality", quality=0.95, latency=900, tools=True)]
    )
    telemetry = TelemetryStore(str(tmp_path / "traces.db"))
    router = Router(registry, ProviderFactory(), telemetry)
    request = ChatRequest(messages=[Message(role="user", content="hello")], route=RouteOptions(profile="fastest"))
    assert router.decision(request).selected_model == "fast"

    tool_request = ChatRequest(
        messages=[Message(role="user", content="look this up")],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        route=RouteOptions(profile="fastest"),
    )
    decision = router.decision(tool_request)
    assert decision.selected_model == "quality"
    assert not next(candidate for candidate in decision.candidates if candidate.model_id == "fast").eligible


@pytest.mark.asyncio
async def test_execute_records_trace_without_prompt(tmp_path: Path) -> None:
    telemetry = TelemetryStore(str(tmp_path / "traces.db"))
    providers = ProviderFactory()
    router = Router(ModelRegistry([model("quality", quality=0.95, latency=10)]), providers, telemetry)
    result = await router.execute(ChatRequest(messages=[Message(role="user", content="a private prompt")]))
    route_data = result["_routemcp"]
    trace = telemetry.get(route_data["trace_id"])
    assert trace is not None
    assert trace["status"] == "success"
    assert trace["request"] == {"model": "auto", "message_count": 1, "tools_count": 0, "prompts_stored": False}
    await providers.close()
    telemetry.close()


@pytest.mark.asyncio
async def test_streaming_is_explicitly_rejected(tmp_path: Path) -> None:
    telemetry = TelemetryStore(str(tmp_path / "traces.db"))
    providers = ProviderFactory()
    router = Router(ModelRegistry([model("fast", quality=0.5, latency=10)]), providers, telemetry)
    with pytest.raises(RoutingError, match="streaming is not available") as error:
        await router.execute(ChatRequest(messages=[Message(role="user", content="hello")], stream=True))
    assert error.value.status_code == 422
    await providers.close()
    telemetry.close()


def test_unhealthy_model_is_filtered(tmp_path: Path) -> None:
    registry = ModelRegistry(
        [model("fast", quality=0.6, latency=10), model("backup", quality=0.7, latency=100)]
    )
    registry.record_health("fast", {"healthy": False})
    telemetry = TelemetryStore(str(tmp_path / "traces.db"))
    providers = ProviderFactory()
    router = Router(registry, providers, telemetry)
    request = ChatRequest(messages=[Message(role="user", content="hello")])
    decision = router.decision(request)
    assert decision.selected_model == "backup"
    rejected = next(item for item in decision.candidates if item.model_id == "fast")
    assert "provider unhealthy at last probe" in rejected.reasons


@pytest.mark.asyncio
async def test_invalid_json_escalates_to_stronger_model(tmp_path: Path) -> None:
    cheap = model("cheap", quality=0.5, latency=10)
    strong = model("strong", quality=0.95, latency=100)
    telemetry = TelemetryStore(str(tmp_path / "traces.db"))
    providers = ProviderFactory()
    router = Router(ModelRegistry([cheap, strong]), providers, telemetry)
    request = ChatRequest(
        messages=[Message(role="user", content="FAIL_JSON")],
        response_format={"type": "json_object"},
        route=RouteOptions(profile="cheapest"),
    )
    result = await router.execute(request)
    assert result["_routemcp"]["selected_model"] == "strong"
    assert result["_routemcp"]["attempts"] == 2
    trace = telemetry.get(result["_routemcp"]["trace_id"])
    assert trace is not None
    assert trace["attempts"][0]["error_type"] == "invalid_json"
    await providers.close()
    telemetry.close()
