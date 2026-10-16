from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .api import load_container
from .domain import ChatRequest, Message


class ExpectedResult(BaseModel):
    type: str = "contains"
    value: Optional[Any] = None
    fields: List[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    id: str
    task_type: str
    messages: List[Message]
    expected: ExpectedResult
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None


def _answer(payload: Dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"].get("content") or "")


def validate_answer(answer: str, expected: ExpectedResult) -> bool:
    if expected.type == "exact":
        return answer.strip() == str(expected.value).strip()
    if expected.type == "contains":
        values = expected.value if isinstance(expected.value, list) else [expected.value]
        lowered = answer.lower()
        return all(str(value).lower() in lowered for value in values)
    if expected.type == "regex":
        return re.search(str(expected.value), answer, re.IGNORECASE) is not None
    if expected.type == "json_fields":
        try:
            parsed = json.loads(answer)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(parsed, dict) and all(field in parsed for field in expected.fields)
    raise ValueError(f"unknown validator type '{expected.type}'")


def load_dataset(path: str) -> List[EvaluationCase]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(EvaluationCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid dataset row {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


async def evaluate(
    config: str,
    dataset: str,
    output: str,
    strongest_model: Optional[str] = None,
    cheapest_model: Optional[str] = None,
) -> Dict[str, Any]:
    container = load_container(config, ":memory:")
    cases = load_dataset(dataset)
    models = container.registry.list()
    if not models:
        raise ValueError("no models are registered")
    strongest = strongest_model or max(models, key=lambda item: item.reliability.quality).id
    cheapest = cheapest_model or min(
        models,
        key=lambda item: item.cost.input_per_million + item.cost.output_per_million,
    ).id
    policies = {"routemcp": "auto", "strongest": strongest, "cheapest": cheapest}
    rows = []
    try:
        for case in cases:
            for policy, model_id in policies.items():
                request = ChatRequest(
                    model=model_id,
                    messages=case.messages,
                    tools=case.tools,
                    response_format=case.response_format,
                )
                started = time.perf_counter()
                error = None
                cost = 0.0
                passed = False
                selected_model = None
                try:
                    result = await container.router.execute(request)
                    passed = validate_answer(_answer(result), case.expected)
                    route = result["_routemcp"]
                    cost = float(route["actual_cost_usd"])
                    selected_model = route["selected_model"]
                except Exception as exc:
                    error = str(exc)[:500]
                rows.append(
                    {
                        "case_id": case.id,
                        "task_type": case.task_type,
                        "policy": policy,
                        "requested_model": model_id,
                        "selected_model": selected_model,
                        "passed": passed,
                        "cost_usd": cost,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "error": error,
                    }
                )
    finally:
        await container.providers.close()
        container.telemetry.close()
        container.registry.close()

    summary: Dict[str, Any] = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        successes = sum(row["passed"] for row in policy_rows)
        total_cost = sum(row["cost_usd"] for row in policy_rows)
        summary[policy] = {
            "tasks": len(policy_rows),
            "successes": successes,
            "success_rate": round(successes / len(policy_rows), 4),
            "total_cost_usd": round(total_cost, 8),
            "cost_per_success_usd": round(total_cost / successes, 8) if successes else None,
            "average_latency_ms": round(
                sum(row["latency_ms"] for row in policy_rows) / len(policy_rows), 2
            ),
        }
    router_score = summary["routemcp"]["success_rate"]
    strongest_score = summary["strongest"]["success_rate"]
    router_cost = summary["routemcp"]["total_cost_usd"]
    strongest_cost = summary["strongest"]["total_cost_usd"]
    comparison = {
        "quality_retention_vs_strongest": (
            round(router_score / strongest_score, 4) if strongest_score else None
        ),
        "cost_reduction_vs_strongest": (
            round(1 - router_cost / strongest_cost, 4) if strongest_cost else None
        ),
    }
    report = {
        "dataset": dataset,
        "strongest_model": strongest,
        "cheapest_model": cheapest,
        "summary": summary,
        "comparison": comparison,
        "results": rows,
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_evaluation(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(evaluate(**kwargs))
