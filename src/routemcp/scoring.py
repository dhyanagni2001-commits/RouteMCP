from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .domain import CandidateScore, ModelConfig, RequestFeatures, RouteOptions, RoutingDecision

PROFILES: Dict[str, Dict[str, float]] = {
    "cheapest": {"quality": 0.25, "cost": 0.60, "latency": 0.10, "reliability": 0.05},
    "fastest": {"quality": 0.25, "cost": 0.10, "latency": 0.55, "reliability": 0.10},
    "balanced": {"quality": 0.45, "cost": 0.25, "latency": 0.15, "reliability": 0.15},
    "quality": {"quality": 0.70, "cost": 0.05, "latency": 0.05, "reliability": 0.20},
}


def estimate_cost(model: ModelConfig, features: RequestFeatures) -> float:
    return (
        features.estimated_input_tokens * model.cost.input_per_million
        + features.expected_output_tokens * model.cost.output_per_million
    ) / 1_000_000


def _reject(
    model: ModelConfig,
    features: RequestFeatures,
    options: RouteOptions,
    cost: float,
    health: Optional[Dict[str, Any]] = None,
) -> List[str]:
    reasons: List[str] = []
    if not model.enabled:
        reasons.append("model disabled")
    if health is not None and health.get("healthy") is False:
        reasons.append("provider unhealthy at last probe")
    if health is not None and health.get("circuit_open") is True:
        reasons.append("circuit breaker open after repeated failures")
    if features.requires_tools and not model.capabilities.tools:
        reasons.append("tool calling unsupported")
    if features.requires_json and not model.capabilities.json_output:
        reasons.append("JSON output unsupported")
    if features.requires_vision and not model.capabilities.vision:
        reasons.append("vision unsupported")
    if features.required_tokens > model.capabilities.context_window:
        reasons.append(f"requires {features.required_tokens} tokens; limit is {model.capabilities.context_window}")
    if features.expected_output_tokens > model.capabilities.max_output_tokens:
        reasons.append("requested output exceeds model limit")
    if options.max_cost_usd is not None and cost > options.max_cost_usd:
        reasons.append(f"estimated cost ${cost:.6f} exceeds request budget")
    if options.max_latency_ms is not None and model.reliability.median_latency_ms > options.max_latency_ms:
        reasons.append("historical latency exceeds request limit")
    if options.privacy == "local-only" and not model.local:
        reasons.append("local-only privacy policy")
    return reasons


def decide(
    models: Iterable[ModelConfig],
    features: RequestFeatures,
    options: RouteOptions,
    health: Optional[Dict[str, Dict[str, Any]]] = None,
) -> RoutingDecision:
    model_list = list(models)
    costs = {model.id: estimate_cost(model, features) for model in model_list}
    max_cost = max(costs.values(), default=1) or 1
    max_latency = max((m.reliability.median_latency_ms for m in model_list), default=1) or 1
    weights = PROFILES[options.profile]
    candidates: List[CandidateScore] = []

    for model in model_list:
        rejection = _reject(model, features, options, costs[model.id], (health or {}).get(model.id))
        if rejection:
            candidates.append(CandidateScore(
                model_id=model.id, eligible=False, estimated_cost_usd=costs[model.id], reasons=rejection
            ))
            continue
        quality = model.reliability.quality
        # Complex work magnifies quality differences; simple work compresses them.
        quality_fit = quality * (0.65 + 0.35 * features.complexity)
        reliability = (
            model.reliability.tool_success_rate if features.requires_tools else model.reliability.success_rate
        )
        cost_score = 1 - costs[model.id] / max_cost
        latency_score = 1 - model.reliability.median_latency_ms / max_latency
        score = (
            weights["quality"] * quality_fit
            + weights["cost"] * cost_score
            + weights["latency"] * latency_score
            + weights["reliability"] * reliability
        )
        components = {
            "quality": round(quality_fit, 4),
            "cost": round(cost_score, 4),
            "latency": round(latency_score, 4),
            "reliability": round(reliability, 4),
        }
        candidates.append(CandidateScore(
            model_id=model.id,
            eligible=True,
            score=round(score, 6),
            estimated_cost_usd=round(costs[model.id], 8),
            components=components,
            reasons=[f"{options.profile} utility score {score:.3f}"],
        ))

    candidates.sort(key=lambda item: (item.eligible, item.score or -1), reverse=True)
    winner = next((candidate for candidate in candidates if candidate.eligible), None)
    explanation = "No model satisfies all hard constraints."
    if winner:
        explanation = (
            f"Selected {winner.model_id}: highest {options.profile} utility ({winner.score:.3f}) "
            f"after capability, context, budget, latency, and privacy filters."
        )
    return RoutingDecision(
        selected_model=winner.model_id if winner else None,
        profile=options.profile,
        features=features,
        candidates=candidates,
        explanation=explanation,
    )
