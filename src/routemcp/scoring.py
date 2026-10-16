from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .domain import CandidateScore, ModelConfig, RequestFeatures, RouteOptions, RoutingDecision
from .stats import ModelStats

PROFILES: Dict[str, Dict[str, float]] = {
    "cheapest": {"quality": 0.25, "cost": 0.60, "latency": 0.10, "reliability": 0.05},
    "fastest": {"quality": 0.25, "cost": 0.10, "latency": 0.55, "reliability": 0.10},
    "balanced": {"quality": 0.45, "cost": 0.25, "latency": 0.15, "reliability": 0.15},
    "quality": {"quality": 0.70, "cost": 0.05, "latency": 0.05, "reliability": 0.20},
}

# Largest expected-accuracy loss, versus the best eligible model, that a profile
# accepts in exchange for lower cost or latency. Models outside the tolerance stay
# eligible for failover but rank after every model inside it. Overridable per
# request with RouteOptions.max_quality_gap.
QUALITY_TOLERANCE: Dict[str, float] = {
    "cheapest": 0.30,
    "fastest": 0.30,
    "balanced": 0.10,
    "quality": 0.03,
}


# Pseudo-observations given to the YAML prior. With the default of 10, ten live
# samples (or ten calibrated tasks) carry as much weight as the configured value.
DEFAULT_PRIOR_STRENGTH = 10.0

# Pseudo-trials given to the YAML quality prior once a model has calibrated
# results for a task type. Configured quality is a guess, so measured pass rates
# should outweigh it quickly; it still separates models that tie on a small
# calibration set. Chosen with `routemcp replay` over the published OpenAI runs:
# 10 kept the quality profile on the most expensive model even where every model
# scored 100%, while 1-2 downgraded on calibration noise.
DEFAULT_QUALITY_PRIOR_STRENGTH = 5.0


@dataclass
class EffectiveProfile:
    """A model's reliability profile after blending YAML priors with evidence."""

    quality: float
    success_rate: float
    tool_success_rate: float
    median_latency_ms: float
    evidence: Dict[str, Any]


def _blend(prior: float, observed: Optional[float], samples: int, strength: float) -> float:
    if observed is None or samples <= 0:
        return prior
    weight = samples / (samples + strength)
    return prior + weight * (observed - prior)


def effective_profile(
    model: ModelConfig,
    task_type: str,
    stats: Optional[ModelStats] = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    quality_prior_strength: float = DEFAULT_QUALITY_PRIOR_STRENGTH,
) -> EffectiveProfile:
    prior = model.reliability
    if stats is None:
        return EffectiveProfile(
            prior.quality, prior.success_rate, prior.tool_success_rate, prior.median_latency_ms,
            {"source": "prior"},
        )
    quality = prior.quality
    quality_trials = 0
    calibrated = stats.quality.get(task_type)
    if calibrated is not None and calibrated.trials:
        # Beta-binomial posterior mean with the prior as pseudo-trials.
        quality = (prior.quality * quality_prior_strength + calibrated.successes) / (
            quality_prior_strength + calibrated.trials
        )
        quality_trials = calibrated.trials
    return EffectiveProfile(
        quality=quality,
        success_rate=_blend(prior.success_rate, stats.success_rate, stats.samples, prior_strength),
        tool_success_rate=_blend(
            prior.tool_success_rate, stats.tool_success_rate, stats.tool_samples, prior_strength
        ),
        median_latency_ms=_blend(
            prior.median_latency_ms, stats.latency_ms, stats.latency_samples, prior_strength
        ),
        evidence={
            "source": "learned" if stats.samples or quality_trials else "prior",
            "live_samples": stats.samples,
            "quality_trials": quality_trials,
            "quality_task_type": task_type if quality_trials else None,
        },
    )


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
    profile: EffectiveProfile,
    health: Optional[Dict[str, Any]] = None,
) -> List[str]:
    reasons: List[str] = []
    if not model.enabled:
        reasons.append("model disabled")
    if health is not None and health.get("circuit_open") is True:
        if health.get("healthy") is False:
            reasons.append("provider unhealthy at last probe")
        else:
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
    if options.max_latency_ms is not None and profile.median_latency_ms > options.max_latency_ms:
        reasons.append("historical latency exceeds request limit")
    if options.privacy == "local-only" and not model.local:
        reasons.append("local-only privacy policy")
    return reasons


def decide(
    models: Iterable[ModelConfig],
    features: RequestFeatures,
    options: RouteOptions,
    health: Optional[Dict[str, Dict[str, Any]]] = None,
    stats: Optional[Dict[str, ModelStats]] = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    quality_prior_strength: float = DEFAULT_QUALITY_PRIOR_STRENGTH,
) -> RoutingDecision:
    model_list = list(models)
    costs = {model.id: estimate_cost(model, features) for model in model_list}
    profiles = {
        model.id: effective_profile(
            model, features.task_type, (stats or {}).get(model.id), prior_strength,
            quality_prior_strength,
        )
        for model in model_list
    }
    max_cost = max(costs.values(), default=1) or 1
    max_latency = max((p.median_latency_ms for p in profiles.values()), default=1) or 1
    weights = PROFILES[options.profile]
    candidates: List[CandidateScore] = []

    for model in model_list:
        profile = profiles[model.id]
        rejection = _reject(
            model, features, options, costs[model.id], profile, (health or {}).get(model.id)
        )
        if rejection:
            candidates.append(CandidateScore(
                model_id=model.id, eligible=False, estimated_cost_usd=costs[model.id],
                reasons=rejection, evidence=profile.evidence,
            ))
            continue
        # Complex work magnifies quality differences; simple work compresses them.
        quality_fit = profile.quality * (0.65 + 0.35 * features.complexity)
        reliability = profile.tool_success_rate if features.requires_tools else profile.success_rate
        cost_score = 1 - costs[model.id] / max_cost
        latency_score = 1 - profile.median_latency_ms / max_latency
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
            evidence=profile.evidence,
        ))

    eligible = [c for c in candidates if c.eligible]
    tolerance = (
        options.max_quality_gap
        if options.max_quality_gap is not None
        else QUALITY_TOLERANCE[options.profile]
    )
    best_quality = max((profiles[c.model_id].quality for c in eligible), default=0.0)
    within: Dict[str, bool] = {}
    for candidate in eligible:
        gap = best_quality - profiles[candidate.model_id].quality
        candidate.components["quality_gap"] = round(gap, 4)
        within[candidate.model_id] = gap <= tolerance + 1e-9
        if not within[candidate.model_id]:
            candidate.reasons.append(
                f"expected quality {gap:.3f} below best exceeds {options.profile} tolerance "
                f"{tolerance:.2f}; kept as fallback"
            )
    candidates.sort(
        key=lambda item: (item.eligible, within.get(item.model_id, False), item.score or -1),
        reverse=True,
    )
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
