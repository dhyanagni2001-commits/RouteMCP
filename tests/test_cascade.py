import json
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pytest

from routemcp.domain import ChatRequest, Message, ModelConfig, RouteOptions
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.router import Router, answer_confidence
from routemcp.stats import StatsStore
from routemcp.telemetry import TelemetryStore


def remote(model_id: str, quality: float, price: float, logprobs: bool = True) -> ModelConfig:
    return ModelConfig.model_validate({
        "id": model_id, "provider": "openai-compatible", "base_url": f"http://{model_id}.test/v1",
        "model": model_id, "cost": {"input_per_million": price, "output_per_million": price * 4},
        "capabilities": {"logprobs": logprobs},
        "reliability": {"quality": quality, "median_latency_ms": 10},
    })


class Upstreams:
    """Each host answers with a fixed token log-probability, or fails with a status."""

    def __init__(self, logprobs: Dict[str, float], statuses: Optional[Dict[str, int]] = None) -> None:
        self.logprobs = logprobs
        self.statuses = statuses or {}
        self.bodies: Dict[str, List[dict]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host.split(".")[0]
        body = json.loads(request.content)
        self.bodies.setdefault(host, []).append(body)
        if host in self.statuses:
            return httpx.Response(self.statuses[host], json={"error": "boom"})
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": f"answer from {host}"},
            "finish_reason": "stop",
        }
        if body.get("logprobs"):
            choice["logprobs"] = {"content": [
                {"token": "answer", "logprob": -0.0001},
                {"token": host, "logprob": self.logprobs.get(host, 0.0)},
            ]}
        return httpx.Response(200, json={
            "choices": [choice], "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        })


def make_router(tmp_path: Path, upstreams: Upstreams, models: List[ModelConfig]) -> Router:
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(upstreams)))
    return Router(ModelRegistry(models), providers, TelemetryStore(str(tmp_path / "t.db")), StatsStore())


def request(**route: object) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        route=RouteOptions(profile="quality", cascade=True, **route),
    )


def test_answer_confidence() -> None:
    def payload(message: dict, logprobs: object = None) -> dict:
        choice = {"message": message}
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {"choices": [choice]}

    tokens = {"content": [{"logprob": -0.2}, {"logprob": -0.001}]}
    assert answer_confidence(payload({"content": "x"}, tokens)) == -0.2
    assert answer_confidence(payload({"content": "x"})) is None
    call = {"function": {"name": "f", "arguments": '{"a": 1}'}}
    assert answer_confidence(payload({"content": None, "tool_calls": [call]})) == 0.0
    broken = {"function": {"name": "f", "arguments": "{"}}
    assert answer_confidence(payload({"content": None, "tool_calls": [broken]})) is None


@pytest.mark.asyncio
async def test_confident_cheap_answer_is_kept(tmp_path: Path) -> None:
    upstreams = Upstreams({"cheap": -0.001})
    router = make_router(tmp_path, upstreams, [remote("cheap", 0.6, 0.1), remote("strong", 0.9, 5.0)])
    assert router.decision(request()).selected_model == "strong"

    result = await router.execute(request())
    route = result["_routemcp"]
    assert route["selected_model"] == "cheap"
    assert route["attempts"] == 1
    assert route["cascade"]["steps"] == [{"model_id": "cheap", "confidence": -0.001, "accepted": True}]
    assert route["actual_cost_usd"] == pytest.approx((1000 * 0.1 + 100 * 0.4) / 1e6)
    assert "strong" not in upstreams.bodies
    assert upstreams.bodies["cheap"][0]["logprobs"] is True
    # The caller did not ask for logprobs, so they are not returned.
    assert "logprobs" not in result["choices"][0]
    await router.providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_low_confidence_escalates_and_bills_every_attempt(tmp_path: Path) -> None:
    upstreams = Upstreams({"cheap": -2.0})
    router = make_router(tmp_path, upstreams, [remote("cheap", 0.6, 0.1), remote("strong", 0.9, 5.0)])
    result = await router.execute(request())
    route = result["_routemcp"]
    assert route["selected_model"] == "strong"
    assert route["attempts"] == 2
    assert route["reason"] == "cascade: escalated after low-confidence answers"
    cheap_cost = (1000 * 0.1 + 100 * 0.4) / 1e6
    strong_cost = (1000 * 5.0 + 100 * 20.0) / 1e6
    assert route["actual_cost_usd"] == pytest.approx(cheap_cost + strong_cost)
    # The routed model is called exactly as the caller asked, without logprobs.
    assert "logprobs" not in upstreams.bodies["strong"][0]
    trace = router.telemetry.get(route["trace_id"])
    assert trace is not None
    assert [a["status"] for a in trace["attempts"]] == ["low_confidence", "success"]
    assert trace["total_cost_usd"] == pytest.approx(cheap_cost + strong_cost)

    # A stricter threshold escalates; a looser one keeps the cheap answer.
    loose = await router.execute(request(cascade_min_logprob=-3.0))
    assert loose["_routemcp"]["selected_model"] == "cheap"
    await router.providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_failed_escalation_returns_the_low_confidence_answer(tmp_path: Path) -> None:
    upstreams = Upstreams({"cheap": -2.0}, statuses={"strong": 503})
    router = make_router(tmp_path, upstreams, [remote("cheap", 0.6, 0.1), remote("strong", 0.9, 5.0)])
    result = await router.execute(request())
    route = result["_routemcp"]
    assert route["selected_model"] == "cheap"
    assert route["reason"].startswith("cascade: stronger models failed")
    # The cheap model is not called a second time as a failover candidate.
    assert len(upstreams.bodies["cheap"]) == 1
    await router.providers.close()
    router.telemetry.close()


@pytest.mark.asyncio
async def test_cascade_skips_models_without_logprobs_and_is_opt_in(tmp_path: Path) -> None:
    upstreams = Upstreams({"cheap": -0.001, "mid": -0.001})
    router = make_router(tmp_path, upstreams, [
        remote("cheap", 0.6, 0.1, logprobs=False), remote("mid", 0.7, 1.0), remote("strong", 0.9, 5.0),
    ])
    result = await router.execute(request())
    assert result["_routemcp"]["selected_model"] == "mid"
    assert "cheap" not in upstreams.bodies

    plain = await router.execute(request().model_copy(update={"route": RouteOptions(profile="quality")}))
    assert plain["_routemcp"]["selected_model"] == "strong"
    assert "cascade" not in plain["_routemcp"]
    await router.providers.close()
    router.telemetry.close()
