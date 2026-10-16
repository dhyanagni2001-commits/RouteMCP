import json
from pathlib import Path
from typing import Dict

import httpx
import pytest

from routemcp.api import Container, create_app
from routemcp.domain import ChatRequest, Message, ModelConfig, RouteOptions
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.router import Router, RoutingError
from routemcp.stats import StatsStore
from routemcp.telemetry import TelemetryStore


def remote(model_id: str, quality: float, **extra: object) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "id": model_id,
            "provider": "openai-compatible",
            "base_url": f"http://{model_id}.test/v1",
            "model": model_id,
            "reliability": {"quality": quality, "median_latency_ms": 10},
            **extra,
        }
    )


def completion(content: str) -> Dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


class Upstreams:
    """Fake OpenAI-compatible hosts; each host maps to an HTTP status."""

    def __init__(self, statuses: Dict[str, int]) -> None:
        self.statuses = statuses
        self.calls: Dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host.split(".")[0]
        self.calls[host] = self.calls.get(host, 0) + 1
        status = self.statuses.get(host, 200)
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        if json.loads(request.content).get("stream"):
            body = (
                'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completion(f"from {host}"))


def make_router(tmp_path: Path, upstreams: Upstreams, threshold: int = 2) -> Router:
    registry = ModelRegistry(
        [remote("primary", 0.95), remote("backup", 0.9)], failure_threshold=threshold
    )
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    return Router(registry, providers, TelemetryStore(str(tmp_path / "t.db")), StatsStore())


def request(**route: object) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        route=RouteOptions(profile="quality", **route),
    )


@pytest.mark.asyncio
async def test_failover_then_circuit_skips_failed_provider(tmp_path: Path) -> None:
    upstreams = Upstreams({"primary": 503})
    router = make_router(tmp_path, upstreams, threshold=2)
    for _ in range(2):
        result = await router.execute(request())
        assert result["_routemcp"]["selected_model"] == "backup"
        assert result["_routemcp"]["attempts"] == 2
    health = router.registry.health("primary")
    assert health is not None and health["circuit_open"] is True
    assert health["consecutive_failures"] == 2

    result = await router.execute(request())
    assert result["_routemcp"]["attempts"] == 1
    assert upstreams.calls["primary"] == 2
    trace = router.telemetry.get(result["_routemcp"]["trace_id"])
    assert trace is not None
    rejected = next(c for c in trace["decision"]["candidates"] if c["model_id"] == "primary")
    assert "circuit breaker open after repeated failures" in rejected["reasons"]

    learned = router.stats.get("primary")
    assert learned is not None and learned.success_rate == 0.0
    await router.providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_client_errors_are_typed_and_respect_max_attempts(tmp_path: Path) -> None:
    router = make_router(tmp_path, Upstreams({"primary": 400, "backup": 429}))
    with pytest.raises(RoutingError) as raised:
        await router.execute(request())
    assert raised.value.error_type == "rate_limit"
    with pytest.raises(RoutingError) as limited:
        await router.execute(request(max_attempts=1))
    assert limited.value.error_type == "provider_error"
    trace = router.telemetry.get(limited.value.trace_id)
    assert trace is not None and len(trace["attempts"]) == 1
    await router.providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_missing_credentials_fall_through_to_next_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROUTEMCP_TEST_MISSING_KEY", raising=False)
    upstreams = Upstreams({})
    registry = ModelRegistry(
        [remote("primary", 0.95, api_key_env="ROUTEMCP_TEST_MISSING_KEY"), remote("backup", 0.9)]
    )
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    router = Router(registry, providers, TelemetryStore(str(tmp_path / "t.db")))
    result = await router.execute(request())
    assert result["_routemcp"]["selected_model"] == "backup"
    assert "primary" not in upstreams.calls
    await providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_stream_fails_over_before_first_byte(tmp_path: Path) -> None:
    router = make_router(tmp_path, Upstreams({"primary": 500}))
    plan = await router.prepare_stream(request().model_copy(update={"stream": True}))
    body = b"".join([chunk async for chunk in plan.iterator])
    assert plan.model_id == "backup"
    assert plan.attempts == 2
    assert body.endswith(b"data: [DONE]\n\n")
    trace = router.telemetry.get(plan.trace_id)
    assert trace is not None and trace["status"] == "success"
    assert [a["status"] for a in trace["attempts"]] == ["failed", "success"]
    await router.providers.close()
    router.telemetry.close()


def test_failed_probe_recovers_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr("routemcp.registry.time.time", lambda: clock[0])
    registry = ModelRegistry([remote("primary", 0.9)], cooldown_seconds=30)
    registry.record_health("primary", {"healthy": False, "detail": "HTTP 503"})
    health = registry.health("primary")
    assert health is not None and health["circuit_open"] is True

    clock[0] += 31
    half_open = registry.health("primary")
    assert half_open is not None
    assert half_open["circuit_open"] is False and half_open["half_open"] is True

    # A live success after the cooldown clears the stale failed probe.
    registry.record_success("primary")
    recovered = registry.health("primary")
    assert recovered is not None
    assert recovered["healthy"] is True and "half_open" not in recovered


@pytest.mark.asyncio
async def test_api_failover_records_learned_stats_and_resets_on_delete(tmp_path: Path) -> None:
    upstreams = Upstreams({"primary": 502})
    registry = ModelRegistry([remote("primary", 0.95), remote("backup", 0.9)])
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    container = Container(registry, providers, TelemetryStore(str(tmp_path / "t.db")), StatsStore())
    transport = httpx.ASGITransport(app=create_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "route": {"profile": "quality"}},
        )
        assert response.status_code == 200
        assert response.json()["_routemcp"]["selected_model"] == "backup"
        models = {m["id"]: m for m in (await client.get("/v1/router/models")).json()["data"]}
        assert models["primary"]["learned"]["success_rate"] == 0.0
        assert models["backup"]["learned"]["samples"] == 1
        assert (await client.delete("/v1/router/models/backup")).status_code == 204
        assert container.stats is not None and container.stats.get("backup") is None
        failed = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert failed.status_code == 503
        assert failed.json()["error"]["type"] == "routing_error"
    await providers.close()
    container.telemetry.close()
