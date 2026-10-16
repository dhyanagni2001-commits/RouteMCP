from pathlib import Path

import pytest

from routemcp.analyzer import analyze
from routemcp.domain import ChatRequest, Message, ModelConfig, RouteOptions
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.router import Router
from routemcp.scoring import effective_profile
from routemcp.stats import ModelStats, QualityEvidence, StatsStore
from routemcp.telemetry import TelemetryStore


def model(model_id: str, *, quality: float, latency: float, price: float) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "id": model_id,
            "provider": "mock",
            "model": model_id,
            "cost": {"input_per_million": price, "output_per_million": price * 4},
            "reliability": {"quality": quality, "median_latency_ms": latency},
        }
    )


def test_ewma_observations_persist_across_restarts(tmp_path: Path) -> None:
    database = tmp_path / "stats.db"
    stats = StatsStore(database, alpha=0.5)
    stats.observe("m", success=True, latency_ms=100)
    stats.observe("m", success=False, latency_ms=900)  # failed latency is ignored
    stats.observe("m", success=True, latency_ms=300, tool_call=True)
    learned = stats.get("m")
    assert learned is not None
    assert learned.samples == 3
    assert learned.latency_samples == 2
    assert learned.latency_ms == pytest.approx(200)
    assert learned.success_rate == pytest.approx(0.75)
    assert learned.tool_samples == 1 and learned.tool_success_rate == 1.0
    stats.record_quality("m", "math", trials=10, successes=4)
    stats.close()

    reopened = StatsStore(database)
    restored = reopened.get("m")
    assert restored is not None and restored.samples == 3
    assert restored.quality["math"] == QualityEvidence(trials=10, successes=4)
    reopened.reset("m")
    assert reopened.get("m") is None
    reopened.close()


def test_record_quality_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        StatsStore().record_quality("m", "math", trials=3, successes=4)


def test_effective_profile_shrinks_toward_prior() -> None:
    config = model("m", quality=0.9, latency=1000, price=1)
    assert effective_profile(config, "math").quality == 0.9
    few = ModelStats(quality={"math": QualityEvidence(trials=2, successes=0)})
    many = ModelStats(quality={"math": QualityEvidence(trials=200, successes=40)})
    assert effective_profile(config, "math", few, quality_prior_strength=10).quality == pytest.approx(
        0.75
    )
    assert effective_profile(config, "math", many, quality_prior_strength=10).quality == pytest.approx(
        (9 + 40) / 210
    )
    # By default measured pass rates outweigh the configured guess quickly.
    assert effective_profile(config, "math", few).quality == pytest.approx(4.5 / 7)
    # Calibration for another task type does not leak.
    assert effective_profile(config, "coding", many).quality == 0.9
    latency = ModelStats(samples=10, latency_samples=10, latency_ms=100)
    blended = effective_profile(config, "math", latency, prior_strength=10)
    assert blended.median_latency_ms == pytest.approx(550)
    assert blended.evidence["source"] == "learned"


def test_calibrated_quality_changes_routing_per_task_type(tmp_path: Path) -> None:
    registry = ModelRegistry(
        [model("cheap", quality=0.6, latency=50, price=0.1), model("strong", quality=0.9, latency=500, price=5)]
    )
    stats = StatsStore()
    telemetry = TelemetryStore(str(tmp_path / "t.db"))
    router = Router(registry, ProviderFactory(), telemetry, stats)
    classify = ChatRequest(
        messages=[Message(role="user", content="Classify the sentiment: I love it.")]
    )
    math = ChatRequest(
        messages=[Message(role="user", content="Compute the total of 17 and 25 boxes.")]
    )
    assert analyze(classify).task_type == "classification"
    assert analyze(math).task_type == "math"
    # Priors alone: the 0.3 prior quality gap exceeds the balanced tolerance.
    assert router.decision(classify).selected_model == "strong"
    assert router.decision(math).selected_model == "strong"

    stats.record_quality("cheap", "classification", 50, 50)
    stats.record_quality("strong", "classification", 50, 50)
    stats.record_quality("cheap", "math", 50, 10)
    stats.record_quality("strong", "math", 50, 49)
    assert router.decision(classify).selected_model == "cheap"
    # A per-request tolerance can still accept the cheap model's measured gap.
    loose = math.model_copy(update={"route": RouteOptions(max_quality_gap=1.0)})
    assert router.decision(loose).selected_model == "cheap"
    decision = router.decision(math)
    assert decision.selected_model == "strong"
    strong = next(c for c in decision.candidates if c.model_id == "strong")
    assert strong.evidence["quality_trials"] == 50
    telemetry.close()


@pytest.mark.asyncio
async def test_live_latency_feeds_fastest_profile(tmp_path: Path) -> None:
    # Configured priors claim "a" is faster, but the mock reports its real latency.
    slow = model("a", quality=0.7, latency=10, price=1)
    fast = model("b", quality=0.7, latency=40, price=1)
    registry = ModelRegistry([slow, fast])
    stats = StatsStore()
    telemetry = TelemetryStore(str(tmp_path / "t.db"))
    providers = ProviderFactory()
    router = Router(registry, providers, telemetry, stats)
    request = ChatRequest(
        messages=[Message(role="user", content="hello")], route=RouteOptions(profile="fastest")
    )
    assert router.decision(request).selected_model == "a"
    for _ in range(40):
        stats.observe("a", success=True, latency_ms=400)
        stats.observe("b", success=True, latency_ms=40)
    assert router.decision(request).selected_model == "b"
    await router.execute(request)
    learned = stats.get("b")
    assert learned is not None and learned.samples == 41
    await providers.close()
    telemetry.close()


def test_measured_ties_go_to_the_cheaper_model(tmp_path: Path) -> None:
    # Configured priors rank "strong" far ahead, but both models aced calibration;
    # even the quality profile should stop paying 50x for the same measured result.
    registry = ModelRegistry(
        [model("cheap", quality=0.6, latency=50, price=0.1), model("strong", quality=0.85, latency=50, price=5)]
    )
    stats = StatsStore()
    telemetry = TelemetryStore(str(tmp_path / "t.db"))
    router = Router(registry, ProviderFactory(), telemetry, stats)
    request = ChatRequest(
        messages=[Message(role="user", content="Classify the sentiment: I love it.")],
        route=RouteOptions(profile="quality"),
    )
    assert router.decision(request).selected_model == "strong"
    stats.record_quality("cheap", "classification", 40, 40)
    stats.record_quality("strong", "classification", 40, 40)
    assert router.decision(request).selected_model == "cheap"
    telemetry.close()
