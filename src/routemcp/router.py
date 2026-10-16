from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .analyzer import analyze
from .domain import Attempt, ChatRequest, RouteOptions, RoutingDecision
from .providers import ProviderError, ProviderFactory
from .registry import ModelRegistry
from .scoring import decide, estimate_cost
from .telemetry import TelemetryStore
from .validation import ValidationFailure, validate_response


class RoutingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503, trace_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.trace_id = trace_id


class Router:
    def __init__(self, registry: ModelRegistry, providers: ProviderFactory, telemetry: TelemetryStore):
        self.registry = registry
        self.providers = providers
        self.telemetry = telemetry

    def decision(self, request: ChatRequest) -> RoutingDecision:
        options = request.route or RouteOptions()
        features = analyze(request)
        if request.model == "auto":
            models = self.registry.list()
        else:
            try:
                models = [self.registry.get(request.model)]
            except KeyError as exc:
                raise RoutingError(str(exc), status_code=404) from exc
        health = {model.id: status for model in models if (status := self.registry.health(model.id)) is not None}
        return decide(models, features, options, health)

    async def execute(self, request: ChatRequest) -> Dict[str, Any]:
        if request.stream:
            raise RoutingError("streaming is not available in v0.1; send stream=false", status_code=422)
        trace_id = uuid.uuid4().hex
        decision = self.decision(request)
        options = request.route or RouteOptions()
        candidates = [candidate for candidate in decision.candidates if candidate.eligible]
        if request.model != "auto":
            candidates = candidates[:1]
        candidates = candidates[: options.max_attempts]
        attempts: List[Attempt] = []
        last_error = "no eligible model"

        for candidate in candidates:
            model = self.registry.get(candidate.model_id)
            started = time.perf_counter()
            try:
                result = await self.providers.create(model).complete(request)
                validate_response(request, result.payload)
                usage = result.usage
                actual_cost = (
                    usage["prompt_tokens"] * model.cost.input_per_million
                    + usage["completion_tokens"] * model.cost.output_per_million
                ) / 1_000_000
                attempts.append(Attempt(
                    model_id=model.id, status="success", latency_ms=round(result.latency_ms, 2),
                    estimated_cost_usd=estimate_cost(model, decision.features),
                    actual_cost_usd=round(actual_cost, 8),
                ))
                self._record(trace_id, model.id, "success", request, decision, attempts)
                result.payload["_routemcp"] = {
                    "trace_id": trace_id,
                    "selected_model": model.id,
                    "attempts": len(attempts),
                    "estimated_cost_usd": candidate.estimated_cost_usd,
                    "actual_cost_usd": round(actual_cost, 8),
                    "reason": decision.explanation if len(attempts) == 1 else "selected after escalation",
                }
                return result.payload
            except (ProviderError, ValidationFailure) as exc:
                last_error = str(exc)
                attempts.append(Attempt(
                    model_id=model.id,
                    status="failed",
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    estimated_cost_usd=estimate_cost(model, decision.features),
                    actual_cost_usd=0,
                    error_type=exc.error_type,
                    error=str(exc)[:500],
                ))
                if isinstance(exc, ProviderError) and not exc.retryable:
                    # Configuration errors are model-local; auto mode can still try another model.
                    continue

        self._record(trace_id, None, "failed", request, decision, attempts)
        raise RoutingError(f"all eligible models failed: {last_error}", trace_id=trace_id)

    def _record(
        self, trace_id: str, selected_model: Optional[str], status: str,
        request: ChatRequest, decision: RoutingDecision, attempts: List[Attempt],
    ) -> None:
        request_data: Dict[str, Any]
        if os.getenv("ROUTEMCP_STORE_PROMPTS", "false").lower() == "true":
            request_data = request.model_dump(mode="json")
        else:
            request_data = {
                "model": request.model,
                "message_count": len(request.messages),
                "tools_count": len(request.tools or []),
                "prompts_stored": False,
            }
        self.telemetry.record(
            trace_id, selected_model, status, decision.profile, request_data,
            decision.model_dump(mode="json"), [attempt.model_dump(mode="json") for attempt in attempts],
        )
