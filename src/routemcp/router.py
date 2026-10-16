from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

from .analyzer import analyze
from .domain import Attempt, CandidateScore, ChatRequest, ModelConfig, RouteOptions, RoutingDecision
from .providers import ProviderError, ProviderFactory, ProviderStream
from .registry import ModelRegistry
from .scoring import DEFAULT_PRIOR_STRENGTH, DEFAULT_QUALITY_PRIOR_STRENGTH, decide, estimate_cost
from .stats import StatsStore
from .telemetry import TelemetryStore
from .validation import ValidationFailure, validate_response

T = TypeVar("T")


class _Exhausted(Exception):
    def __init__(self, message: str, error_type: Optional[str]) -> None:
        super().__init__(message)
        self.error_type = error_type


def answer_confidence(payload: Dict[str, Any]) -> Optional[float]:
    """Lowest token log-probability of the first choice, or None if unknown.

    A tool call whose arguments parse as JSON counts as confident: providers do not
    return log-probabilities for tool calls, and the call is structurally checked.
    """
    choice = (payload.get("choices") or [{}])[0]
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []
    if tool_calls:
        try:
            for call in tool_calls:
                json.loads((call.get("function") or {}).get("arguments") or "")
        except (TypeError, json.JSONDecodeError):
            return None
        return 0.0
    tokens = (choice.get("logprobs") or {}).get("content") or []
    values = [token["logprob"] for token in tokens if isinstance(token.get("logprob"), (int, float))]
    return min(values) if values else None


def actual_cost(model: ModelConfig, payload: Dict[str, Any]) -> float:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return (
        prompt_tokens * model.cost.input_per_million
        + completion_tokens * model.cost.output_per_million
    ) / 1_000_000


class RoutingError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 503,
        trace_id: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.trace_id = trace_id
        self.error_type = error_type


class StreamPlan:
    def __init__(
        self, iterator: AsyncIterator[bytes], trace_id: str, model_id: str, attempts: int
    ) -> None:
        self.iterator = iterator
        self.trace_id = trace_id
        self.model_id = model_id
        self.attempts = attempts


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: ProviderFactory,
        telemetry: TelemetryStore,
        stats: Optional[StatsStore] = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        quality_prior_strength: float = DEFAULT_QUALITY_PRIOR_STRENGTH,
    ):
        self.registry = registry
        self.providers = providers
        self.telemetry = telemetry
        self.stats = stats
        self.prior_strength = prior_strength
        self.quality_prior_strength = quality_prior_strength

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
        stats = self.stats.snapshot() if self.stats is not None else None
        return decide(
            models, features, options, health, stats, self.prior_strength,
            self.quality_prior_strength,
        )

    async def _try_candidates(
        self,
        request: ChatRequest,
        trace_id: str,
        decision: RoutingDecision,
        call: Callable[[ModelConfig], Awaitable[Tuple[T, float]]],
        attempts: List[Attempt],
        exclude: Optional[set] = None,
    ) -> Tuple[ModelConfig, CandidateScore, T]:
        """Try eligible candidates in rank order until one call succeeds.

        ``call`` returns its result and the provider latency. Provider failures feed
        the circuit breaker; provider and validation failures both feed learned
        reliability and escalate to the next candidate. Every try is appended to
        ``attempts``; raises ``_Exhausted`` when no candidate succeeds.
        """
        options = request.route or RouteOptions()
        candidates = [candidate for candidate in decision.candidates if candidate.eligible]
        if request.model != "auto":
            candidates = candidates[:1]
        candidates = [c for c in candidates if c.model_id not in (exclude or set())]
        last_error = "no eligible model"
        last_error_type: Optional[str] = "no_eligible_model"

        for candidate in candidates[: options.max_attempts]:
            model = self.registry.get(candidate.model_id)
            started = time.perf_counter()
            try:
                result, latency_ms = await call(model)
            except (ProviderError, ValidationFailure) as exc:
                last_error = str(exc)
                last_error_type = exc.error_type
                attempts.append(self._failed(model, request, decision, exc, started))
                continue
            self._succeeded(model, request, latency_ms)
            attempts.append(Attempt(
                model_id=model.id, status="success", latency_ms=round(latency_ms, 2),
                estimated_cost_usd=estimate_cost(model, decision.features), actual_cost_usd=0,
            ))
            return model, candidate, result
        raise _Exhausted(f"all eligible models failed: {last_error}", last_error_type)

    def _failed(
        self, model: ModelConfig, request: ChatRequest, decision: RoutingDecision,
        exc: Exception, started: float,
    ) -> Attempt:
        error_type = getattr(exc, "error_type", None)
        if isinstance(exc, ProviderError):
            self.registry.record_failure(model.id, str(exc))
        self._observe(model.id, request, success=False)
        return Attempt(
            model_id=model.id,
            status="failed",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            estimated_cost_usd=estimate_cost(model, decision.features),
            actual_cost_usd=0,
            error_type=error_type,
            error=str(exc)[:500],
        )

    def _succeeded(self, model: ModelConfig, request: ChatRequest, latency_ms: float) -> None:
        self.registry.record_success(model.id)
        self._observe(model.id, request, success=True, latency_ms=latency_ms)

    def _cascade_ladder(self, decision: RoutingDecision) -> List[ModelConfig]:
        """Eligible models cheaper than the routed choice that report logprobs, cheapest first."""
        eligible = [c for c in decision.candidates if c.eligible]
        if not eligible:
            return []
        target = eligible[0]
        ladder = []
        for candidate in sorted(eligible[1:], key=lambda c: c.estimated_cost_usd):
            model = self.registry.get(candidate.model_id)
            if candidate.estimated_cost_usd < target.estimated_cost_usd and model.capabilities.logprobs:
                ladder.append(model)
        return ladder

    async def _cascade(
        self,
        request: ChatRequest,
        decision: RoutingDecision,
        attempts: List[Attempt],
        steps: List[Dict[str, Any]],
    ) -> Tuple[Optional[Tuple[ModelConfig, Dict[str, Any]]], Optional[Tuple[ModelConfig, Dict[str, Any]]], set]:
        """Run the cheap rungs. Returns (accepted, best rejected answer, models tried)."""
        options = request.route or RouteOptions()
        probe = request.model_copy(update={"logprobs": True})
        rejected: Optional[Tuple[ModelConfig, Dict[str, Any]]] = None
        tried: set = set()
        for model in self._cascade_ladder(decision):
            tried.add(model.id)
            started = time.perf_counter()
            try:
                result = await self.providers.create(model).complete(probe)
                validate_response(request, result.payload)
            except (ProviderError, ValidationFailure) as exc:
                attempts.append(self._failed(model, request, decision, exc, started))
                steps.append({"model_id": model.id, "confidence": None, "accepted": False})
                continue
            self._succeeded(model, request, result.latency_ms)
            confidence = answer_confidence(result.payload)
            accepted = confidence is not None and confidence >= options.cascade_min_logprob
            attempts.append(Attempt(
                model_id=model.id,
                status="success" if accepted else "low_confidence",
                latency_ms=round(result.latency_ms, 2),
                estimated_cost_usd=estimate_cost(model, decision.features),
                actual_cost_usd=round(actual_cost(model, result.payload), 8),
            ))
            steps.append({
                "model_id": model.id,
                "confidence": None if confidence is None else round(confidence, 6),
                "accepted": accepted,
            })
            if accepted:
                return (model, result.payload), rejected, tried
            rejected = (model, result.payload)
        return None, rejected, tried

    async def execute(self, request: ChatRequest) -> Dict[str, Any]:
        if request.stream:
            raise RoutingError(
                "use Router.prepare_stream for streaming requests", status_code=500
            )
        trace_id = uuid.uuid4().hex
        decision = self.decision(request)
        options = request.route or RouteOptions()
        attempts: List[Attempt] = []
        steps: List[Dict[str, Any]] = []
        reason = decision.explanation

        async def complete(model: ModelConfig) -> Tuple[Dict[str, Any], float]:
            result = await self.providers.create(model).complete(request)
            validate_response(request, result.payload)
            return result.payload, result.latency_ms

        accepted = rejected = None
        tried: set = set()
        if options.cascade and request.model == "auto":
            accepted, rejected, tried = await self._cascade(request, decision, attempts, steps)
        if accepted is not None:
            model, payload = accepted
            reason = f"cascade: {model.id} answered with confidence above the threshold"
        else:
            try:
                model, _, payload = await self._try_candidates(
                    request, trace_id, decision, complete, attempts, exclude=tried
                )
                attempts[-1].actual_cost_usd = round(actual_cost(model, payload), 8)
                if steps:
                    reason = "cascade: escalated after low-confidence answers"
                elif len(attempts) > 1:
                    reason = "selected after escalation"
            except _Exhausted as exc:
                if rejected is None:
                    self._record(trace_id, None, "failed", request, decision, attempts)
                    raise RoutingError(
                        str(exc), trace_id=trace_id, error_type=exc.error_type
                    ) from exc
                # Stronger models all failed; a low-confidence answer beats an error.
                model, payload = rejected
                reason = "cascade: stronger models failed; returned a low-confidence answer"
        if not request.logprobs:
            for choice in payload.get("choices") or []:
                choice.pop("logprobs", None)
        self._record(trace_id, model.id, "success", request, decision, attempts)
        payload["_routemcp"] = {
            "trace_id": trace_id,
            "selected_model": model.id,
            "attempts": len(attempts),
            "estimated_cost_usd": round(estimate_cost(model, decision.features), 8),
            "actual_cost_usd": round(sum(a.actual_cost_usd for a in attempts), 8),
            "task_type": decision.features.task_type,
            "reason": reason,
        }
        if steps:
            payload["_routemcp"]["cascade"] = {
                "min_logprob": options.cascade_min_logprob, "steps": steps,
            }
        return payload

    async def prepare_stream(self, request: ChatRequest) -> StreamPlan:
        trace_id = uuid.uuid4().hex
        decision = self.decision(request)

        async def start(model: ModelConfig) -> Tuple[ProviderStream, float]:
            stream = await self.providers.create(model).start_stream(request)
            return stream, stream.latency_ms

        attempts: List[Attempt] = []
        try:
            model, _, stream = await self._try_candidates(
                request, trace_id, decision, start, attempts
            )
        except _Exhausted as exc:
            self._record(trace_id, None, "failed", request, decision, attempts)
            raise RoutingError(str(exc), trace_id=trace_id, error_type=exc.error_type) from exc

        async def tracked() -> AsyncIterator[bytes]:
            status = "success"
            try:
                async for chunk in stream.iterator:
                    yield chunk
            except asyncio.CancelledError:
                status = "cancelled"
                attempts[-1].status = "cancelled"
                attempts[-1].error_type = "client_disconnected"
                raise
            except Exception as exc:
                status = "failed"
                self.registry.record_failure(model.id, str(exc))
                self._observe(model.id, request, success=False)
                attempts[-1].status = "failed"
                attempts[-1].error_type = "stream_interrupted"
                attempts[-1].error = str(exc)[:500]
                raise
            finally:
                await stream.close()
                self._record(
                    trace_id,
                    model.id if status == "success" else None,
                    status,
                    request,
                    decision,
                    attempts,
                )

        return StreamPlan(tracked(), trace_id, model.id, len(attempts))

    def _observe(
        self, model_id: str, request: ChatRequest, success: bool, latency_ms: Optional[float] = None
    ) -> None:
        if self.stats is not None:
            self.stats.observe(
                model_id, success=success, latency_ms=latency_ms, tool_call=bool(request.tools)
            )

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
