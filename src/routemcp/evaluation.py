"""Routing evaluation: calibrate on one split, compare policies on a held-out split.

Policies on the test split:

* ``fixed:<model>`` — every configured model on its own (includes the strongest
  and cheapest baselines).
* ``routemcp:<profile>`` — RouteMCP's routed choice for each requested profile.
* ``oracle`` — the cheapest model that passed each task, computed afterwards. It is
  an unreachable upper bound that shows how much room routing has.

Calibration runs every model on the calibration split and stores per-model,
per-task-type pass rates in the stats store, which scoring then uses as quality
evidence. Test-split answers are never used for calibration.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .analyzer import analyze
from .api import Container, load_container
from .domain import ChatRequest, Message, RouteOptions
from .router import RoutingError
from .stats import StatsStore

RETRYABLE_ERRORS = {"rate_limit", "provider_unavailable", "timeout"}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class ExpectedResult(BaseModel):
    type: str = "contains"
    value: Optional[Any] = None
    fields: List[str] = Field(default_factory=list)
    choices: List[str] = Field(default_factory=list)
    tolerance: float = 1e-6


class EvaluationCase(BaseModel):
    id: str
    task_type: str
    messages: List[Message]
    expected: ExpectedResult
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    difficulty: Optional[str] = None

    def request(self, model: str, profile: str = "balanced") -> ChatRequest:
        # "quality+cascade" evaluates a profile with the cheap-first cascade enabled.
        profile, _, mode = profile.partition("+")
        return ChatRequest(
            model=model,
            messages=self.messages,
            tools=self.tools,
            response_format=self.response_format,
            max_tokens=self.max_tokens,
            temperature=0,
            route=RouteOptions(profile=profile, cascade=mode == "cascade"),
        )


def _answer(payload: Dict[str, Any]) -> str:
    return str(payload["choices"][0]["message"].get("content") or "")


def _strip_fences(answer: str) -> str:
    stripped = answer.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return match.group(1) if match else stripped


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("`\"'.").strip()).lower()


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return isinstance(actual, str) and _normalize(actual) == _normalize(expected)
    return actual == expected


def validate_answer(
    answer: str, expected: ExpectedResult, payload: Optional[Dict[str, Any]] = None
) -> bool:
    kind = expected.type
    if kind == "exact":
        return answer.strip() == str(expected.value).strip()
    if kind == "normalized":
        return _normalize(answer) == _normalize(str(expected.value))
    if kind == "contains":
        values = expected.value if isinstance(expected.value, list) else [expected.value]
        lowered = answer.lower()
        return all(str(value).lower() in lowered for value in values)
    if kind == "regex":
        return re.search(str(expected.value), answer, re.IGNORECASE) is not None
    if kind == "number":
        # The final number in the answer must match, so "3 + 4 = 7" passes for 7.
        numbers = NUMBER.findall(answer)
        if not numbers:
            return False
        try:
            value = float(numbers[-1].replace(",", ""))
        except ValueError:
            return False
        return abs(value - float(expected.value)) <= expected.tolerance
    if kind == "choice":
        # Exactly one of the allowed labels may appear, and it must be the right one.
        found = {
            choice for choice in expected.choices
            if re.search(rf"\b{re.escape(choice)}\b", answer, re.IGNORECASE)
        }
        return found == {str(expected.value)}
    if kind in {"json_fields", "json_match"}:
        try:
            parsed = json.loads(_strip_fences(answer))
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(parsed, dict):
            return False
        if kind == "json_fields":
            return all(field in parsed for field in expected.fields)
        return all(
            key in parsed and _values_equal(parsed[key], value)
            for key, value in dict(expected.value or {}).items()
        )
    if kind == "tool_call":
        calls = (((payload or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
        if not calls:
            return False
        function = calls[0].get("function") or {}
        spec = dict(expected.value or {})
        if function.get("name") != spec.get("name"):
            return False
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            return False
        return all(
            _values_equal(arguments.get(key), value)
            for key, value in dict(spec.get("arguments") or {}).items()
        )
    raise ValueError(f"unknown validator type '{kind}'")


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


def split_cases(
    cases: Sequence[EvaluationCase], calibration_fraction: float, seed: int
) -> Tuple[List[EvaluationCase], List[EvaluationCase]]:
    """Stratified split by dataset task type so every type appears in both halves."""
    if calibration_fraction <= 0:
        return [], list(cases)
    rng = random.Random(seed)
    by_type: Dict[str, List[EvaluationCase]] = defaultdict(list)
    for case in cases:
        by_type[case.task_type].append(case)
    calibration: List[EvaluationCase] = []
    test: List[EvaluationCase] = []
    for task_type in sorted(by_type):
        group = list(by_type[task_type])
        rng.shuffle(group)
        cut = round(len(group) * calibration_fraction)
        calibration.extend(group[:cut])
        test.extend(group[cut:])
    return calibration, test


async def _run_case(
    container: Container,
    case: EvaluationCase,
    policy: str,
    model: str,
    profile: str,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> Dict[str, Any]:
    request = case.request(model, profile)
    routed_task_type = analyze(request).task_type
    async with semaphore:
        started = time.perf_counter()
        error: Optional[str] = None
        passed = False
        cost = 0.0
        selected_model = None
        attempts = 0
        answer: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                result = await container.router.execute(request)
                answer = _answer(result)
                passed = validate_answer(answer, case.expected, result)
                route = result["_routemcp"]
                cost = float(route["actual_cost_usd"])
                selected_model = route["selected_model"]
                attempts = int(route["attempts"])
                error = None
                break
            except RoutingError as exc:
                error = str(exc)[:500]
                if exc.error_type not in RETRYABLE_ERRORS or attempt == retries:
                    break
                await asyncio.sleep(min(30.0, 2.0 ** attempt))
            except Exception as exc:  # pragma: no cover - defensive; keep the run going
                error = str(exc)[:500]
                break
        return {
            "case_id": case.id,
            "task_type": case.task_type,
            "routed_task_type": routed_task_type,
            "difficulty": case.difficulty,
            "policy": policy,
            "requested_model": model,
            "selected_model": selected_model,
            "attempts": attempts,
            "passed": passed,
            "cost_usd": cost,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": error,
            # Kept short so failures can be audited without bloating the report.
            "answer": answer[:300] if answer else answer,
        }


async def _run_all(
    container: Container,
    jobs: Iterable[Tuple[EvaluationCase, str, str, str]],
    concurrency: int,
    retries: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    return list(
        await asyncio.gather(
            *(
                _run_case(container, case, policy, model, profile, semaphore, retries)
                for case, policy, model, profile in jobs
            )
        )
    )


def calibrate_from_rows(stats: StatsStore, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Store per-model, per-routed-task-type pass rates from fixed-model rows."""
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if not row["policy"].startswith("fixed:"):
            continue
        key = (row["requested_model"], row["routed_task_type"])
        counts[key][0] += 1
        counts[key][1] += int(bool(row["passed"]))
    table: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (model_id, task_type), (trials, successes) in sorted(counts.items()):
        stats.record_quality(model_id, task_type, trials, successes)
        table[model_id][task_type] = {
            "trials": trials,
            "successes": successes,
            "pass_rate": round(successes / trials, 4) if trials else None,
        }
    return dict(table)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    successes = sum(row["passed"] for row in rows)
    total_cost = sum(row["cost_usd"] for row in rows)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "tasks": len(rows),
        "successes": successes,
        "success_rate": round(successes / len(rows), 4) if rows else None,
        "errors": sum(1 for row in rows if row["error"]),
        "total_cost_usd": round(total_cost, 8),
        "cost_per_success_usd": round(total_cost / successes, 8) if successes else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def _oracle_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["policy"].startswith("fixed:"):
            by_case[row["case_id"]].append(row)
    oracle = []
    for case_id, case_rows in by_case.items():
        passing = [row for row in case_rows if row["passed"]]
        best = min(passing or case_rows, key=lambda row: row["cost_usd"])
        oracle.append({**best, "policy": "oracle", "passed": bool(passing)})
    return oracle


def build_report(
    rows: List[Dict[str, Any]],
    strongest: str,
    cheapest: str,
    profiles: Sequence[str],
) -> Dict[str, Any]:
    rows = rows + _oracle_rows(rows)
    policies = sorted({row["policy"] for row in rows})
    summary = {policy: _summarize([r for r in rows if r["policy"] == policy]) for policy in policies}
    task_types = sorted({row["task_type"] for row in rows})
    by_task_type = {
        task_type: {
            policy: _summarize(
                [r for r in rows if r["policy"] == policy and r["task_type"] == task_type]
            )["success_rate"]
            for policy in policies
        }
        for task_type in task_types
    }
    strongest_summary = summary[f"fixed:{strongest}"]
    comparison: Dict[str, Any] = {}
    route_distribution: Dict[str, Dict[str, Dict[str, int]]] = {}
    for profile in profiles:
        policy = f"routemcp:{profile}"
        routed = summary[policy]
        retention = (
            round(routed["success_rate"] / strongest_summary["success_rate"], 4)
            if strongest_summary["success_rate"]
            else None
        )
        reduction = (
            round(1 - routed["total_cost_usd"] / strongest_summary["total_cost_usd"], 4)
            if strongest_summary["total_cost_usd"]
            else None
        )
        comparison[policy] = {
            "quality_retention_vs_strongest": retention,
            "cost_reduction_vs_strongest": reduction,
            "success_rate_delta_vs_cheapest": round(
                routed["success_rate"] - summary[f"fixed:{cheapest}"]["success_rate"], 4
            ),
        }
        distribution: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            if row["policy"] == policy and row["selected_model"]:
                distribution[row["task_type"]][row["selected_model"]] += 1
        route_distribution[policy] = {k: dict(v) for k, v in sorted(distribution.items())}
    agreement: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["policy"] == f"fixed:{strongest}":
            agreement[row["task_type"]][row["routed_task_type"]] += 1
    return {
        "strongest_model": strongest,
        "cheapest_model": cheapest,
        "summary": summary,
        "comparison": comparison,
        "by_task_type": by_task_type,
        "route_distribution": route_distribution,
        "analyzer_task_type_agreement": {k: dict(v) for k, v in sorted(agreement.items())},
    }


async def evaluate_container(
    container: Container,
    cases: Sequence[EvaluationCase],
    *,
    strongest_model: Optional[str] = None,
    cheapest_model: Optional[str] = None,
    calibration_fraction: float = 0.0,
    profiles: Sequence[str] = ("balanced",),
    seed: int = 0,
    concurrency: int = 4,
    retries: int = 3,
) -> Dict[str, Any]:
    models = [model for model in container.registry.list() if model.enabled]
    if not models:
        raise ValueError("no models are registered")
    strongest = strongest_model or max(models, key=lambda item: item.reliability.quality).id
    cheapest = cheapest_model or min(
        models, key=lambda item: item.cost.input_per_million + item.cost.output_per_million
    ).id
    calibration_cases, test_cases = split_cases(cases, calibration_fraction, seed)

    calibration_table: Optional[Dict[str, Any]] = None
    calibration_rows: List[Dict[str, Any]] = []
    if calibration_cases:
        if container.stats is None:
            container.stats = StatsStore()
            container.router.stats = container.stats
        calibration_rows = await _run_all(
            container,
            [(case, f"fixed:{m.id}", m.id, "balanced") for case in calibration_cases for m in models],
            concurrency,
            retries,
        )
        calibration_table = calibrate_from_rows(container.stats, calibration_rows)

    jobs = [(case, f"fixed:{m.id}", m.id, "balanced") for case in test_cases for m in models]
    jobs += [(case, f"routemcp:{p}", "auto", p) for case in test_cases for p in profiles]
    rows = await _run_all(container, jobs, concurrency, retries)

    report = build_report(rows, strongest, cheapest, profiles)
    report.update(
        {
            "split": {
                "seed": seed,
                "calibration_fraction": calibration_fraction,
                "calibration_tasks": len(calibration_cases),
                "test_tasks": len(test_cases),
            },
            "calibration": calibration_table,
            "models": {
                m.id: {
                    "provider": m.provider.value,
                    "model": m.model,
                    "input_per_million": m.cost.input_per_million,
                    "output_per_million": m.cost.output_per_million,
                }
                for m in models
            },
            "results": rows,
            "calibration_results": calibration_rows,
        }
    )
    return report


async def evaluate(
    config: str,
    dataset: str,
    output: str,
    strongest_model: Optional[str] = None,
    cheapest_model: Optional[str] = None,
    calibration_fraction: float = 0.0,
    profiles: Sequence[str] = ("balanced",),
    seed: int = 0,
    concurrency: int = 4,
    retries: int = 3,
) -> Dict[str, Any]:
    container = load_container(config, ":memory:")
    try:
        report = await evaluate_container(
            container,
            load_dataset(dataset),
            strongest_model=strongest_model,
            cheapest_model=cheapest_model,
            calibration_fraction=calibration_fraction,
            profiles=profiles,
            seed=seed,
            concurrency=concurrency,
            retries=retries,
        )
    finally:
        await container.providers.close()
        container.close_stores()
    report = {"dataset": dataset, **report}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_evaluation(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(evaluate(**kwargs))


def calibrate(report_path: str, db_path: str) -> Dict[str, Any]:
    """Load every fixed-model result from an evaluation report into a gateway's stats store."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    stats = StatsStore(db_path)
    try:
        return calibrate_from_rows(
            stats, list(report.get("calibration_results") or []) + list(report["results"])
        )
    finally:
        stats.close()


async def replay(
    report_path: str,
    config: str,
    profiles: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-route a finished evaluation offline with the current scoring code.

    Calibration uses the report's calibration rows, and each routed decision is
    scored with the recorded fixed-model result for that task and model, so a
    scoring change can be measured without new API calls. This is exact only when
    upstream answers are deterministic (the harness sends temperature 0).
    """
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    profiles = list(profiles or [p.split(":", 1)[1] for p in report["comparison"]])
    # Cascades depend on live log-probabilities that the report does not hold.
    if any("+" in profile for profile in profiles):
        if not any("+" not in profile for profile in profiles):
            raise ValueError("replay cannot evaluate cascade policies; they need live calls")
        profiles = [profile for profile in profiles if "+" not in profile]
    cases = {case.id: case for case in load_dataset(dataset or report["dataset"])}
    container = load_container(config, ":memory:")
    container.stats = StatsStore()
    container.router.stats = container.stats
    try:
        calibration_table = calibrate_from_rows(container.stats, report["calibration_results"])
        fixed = [row for row in report["results"] if row["policy"].startswith("fixed:")]
        outcome = {(row["case_id"], row["requested_model"]): row for row in fixed}
        rows = list(fixed)
        for case_id in sorted({row["case_id"] for row in fixed}):
            for profile in profiles:
                decision = container.router.decision(cases[case_id].request("auto", profile))
                recorded = outcome[(case_id, decision.selected_model)]
                rows.append({**recorded, "policy": f"routemcp:{profile}", "requested_model": "auto"})
    finally:
        await container.providers.close()
        container.close_stores()
    replayed = build_report(rows, report["strongest_model"], report["cheapest_model"], profiles)
    replayed.update(
        {
            "dataset": dataset or report["dataset"],
            "mode": "replay",
            "source_report": report_path,
            "split": report["split"],
            "calibration": calibration_table,
            "models": report.get("models"),
            "results": rows,
            "calibration_results": report["calibration_results"],
        }
    )
    return replayed


def run_replay(**kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(replay(**kwargs))
