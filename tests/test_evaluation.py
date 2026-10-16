import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest

from routemcp.api import Container
from routemcp.domain import ModelConfig
from routemcp.evaluation import (
    EvaluationCase,
    ExpectedResult,
    calibrate,
    evaluate_container,
    load_dataset,
    replay,
    split_cases,
    validate_answer,
)
from routemcp.providers import ProviderFactory
from routemcp.registry import ModelRegistry
from routemcp.stats import StatsStore
from routemcp.telemetry import TelemetryStore

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_dataset_and_validators() -> None:
    cases = load_dataset(str(ROOT / "evaluation/sample.jsonl"))
    assert len(cases) == 3
    assert validate_answer("The answer is 60.", ExpectedResult(type="regex", value=r"\b60\b"))
    assert validate_answer(
        '{"name": "Ada", "role": "engineer"}',
        ExpectedResult(type="json_fields", fields=["name", "role"]),
    )
    assert not validate_answer("not json", ExpectedResult(type="json_fields", fields=["name"]))


def test_extended_validators() -> None:
    number = ExpectedResult(type="number", value=39.5, tolerance=0.01)
    assert validate_answer("Total: 1,000 then 39.50", number)
    assert not validate_answer("39.6", number)
    assert not validate_answer("no digits", number)

    choice = ExpectedResult(type="choice", value="positive", choices=["positive", "negative"])
    assert validate_answer("Positive.", choice)
    assert not validate_answer("positive or negative", choice)
    assert not validate_answer("non-positive-ish", ExpectedResult(type="choice", value="x", choices=["x", "y"]))

    match = ExpectedResult(type="json_match", value={"name": "Ada Lovelace", "age": 36})
    assert validate_answer('```json\n{"name": "ada lovelace", "age": 36.0}\n```', match)
    assert not validate_answer('{"name": "Ada Lovelace", "age": 37}', match)

    assert validate_answer("  `RIVER Ember`.", ExpectedResult(type="normalized", value="river ember"))

    tool = ExpectedResult(type="tool_call", value={"name": "get_weather", "arguments": {"unit": "celsius"}})
    payload = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "Oslo", "unit": "Celsius"}'}}
    ]}}]}
    assert validate_answer("", tool, payload)
    assert not validate_answer("", tool, {"choices": [{"message": {"content": "sunny"}}]})
    with pytest.raises(ValueError):
        validate_answer("x", ExpectedResult(type="unknown"))


def test_generated_dataset_is_deterministic_and_self_consistent(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "evaluation/build_dataset.py")
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    rows = builder.build(7)
    assert rows == builder.build(7)
    assert len(rows) >= 200 and len({row["id"] for row in rows}) == len(rows)
    checked = (ROOT / "evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in checked] == rows
    for row in rows:
        case = EvaluationCase.model_validate(row)
        expected = case.expected
        if expected.type == "tool_call":
            payload = {"choices": [{"message": {"tool_calls": [{"function": {
                "name": expected.value["name"], "arguments": json.dumps(expected.value["arguments"]),
            }}]}}]}
            assert validate_answer("", expected, payload), case.id
        else:
            answer = json.dumps(expected.value) if expected.type == "json_match" else str(expected.value)
            assert validate_answer(answer, expected), case.id


def test_split_is_stratified_and_disjoint() -> None:
    cases = [
        EvaluationCase(id=f"{kind}-{i}", task_type=kind, messages=[{"role": "user", "content": "x"}],
                       expected=ExpectedResult())
        for kind in ("a", "b") for i in range(10)
    ]
    calibration, test = split_cases(cases, 0.5, seed=3)
    assert len(calibration) == len(test) == 10
    assert {c.id for c in calibration}.isdisjoint({c.id for c in test})
    assert sorted(c.task_type for c in calibration).count("a") == 5
    assert split_cases(cases, 0.5, seed=3)[0] == calibration
    assert split_cases(cases, 0, seed=3) == ([], cases)


def math_case(i: int) -> dict:
    return {"id": f"math-{i}", "task_type": "math",
            "messages": [{"role": "user", "content": f"Compute the total of {i} and 10."}],
            "expected": {"type": "number", "value": i + 10}}


def classify_case(i: int) -> dict:
    return {"id": f"cls-{i}", "task_type": "classification",
            "messages": [{"role": "user", "content": f"Classify the sentiment of review {i}: great!"}],
            "expected": {"type": "choice", "value": "positive", "choices": ["positive", "negative"]}}


def fake_upstream(request: httpx.Request) -> httpx.Response:
    """'strong' answers everything; 'cheap' only gets classification right."""
    host = request.url.host.split(".")[0]
    prompt = json.loads(request.content)["messages"][-1]["content"]
    if "sentiment" in prompt:
        answer = "positive"
    else:
        numbers = [int(n) for n in re.findall(r"\d+", prompt)]
        answer = str(sum(numbers)) if host == "strong" else "I am not sure"
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    })


def remote(model_id: str, quality: float, price: float) -> ModelConfig:
    return ModelConfig.model_validate({
        "id": model_id, "provider": "openai-compatible", "base_url": f"http://{model_id}.test/v1",
        "model": model_id, "cost": {"input_per_million": price, "output_per_million": price * 4},
        "reliability": {"quality": quality, "median_latency_ms": 10},
    })


@pytest.mark.asyncio
async def test_calibrated_routing_beats_fixed_baselines(tmp_path: Path) -> None:
    cases = [EvaluationCase.model_validate(math_case(i)) for i in range(20)]
    cases += [EvaluationCase.model_validate(classify_case(i)) for i in range(20)]
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream)))
    container = Container(
        ModelRegistry([remote("cheap", 0.8, 0.1), remote("strong", 0.85, 10.0)]),
        providers,
        TelemetryStore(str(tmp_path / "t.db")),
        StatsStore(),
    )
    report = await evaluate_container(
        container, cases, calibration_fraction=0.5, profiles=["balanced", "quality"], seed=1,
        concurrency=8,
    )
    await providers.close()

    assert report["split"]["calibration_tasks"] == report["split"]["test_tasks"] == 20
    assert report["calibration"]["cheap"]["math"]["pass_rate"] == 0.0
    assert report["calibration"]["cheap"]["classification"]["pass_rate"] == 1.0
    summary = report["summary"]
    assert summary["fixed:strong"]["success_rate"] == 1.0
    assert summary["fixed:cheap"]["success_rate"] == 0.5
    assert summary["oracle"]["success_rate"] == 1.0
    assert summary["routemcp:balanced"]["success_rate"] == 1.0
    assert report["route_distribution"]["routemcp:balanced"] == {
        "classification": {"cheap": 10}, "math": {"strong": 10},
    }
    comparison = report["comparison"]["routemcp:balanced"]
    assert comparison["quality_retention_vs_strongest"] == 1.0
    assert comparison["cost_reduction_vs_strongest"] > 0.4
    assert report["analyzer_task_type_agreement"]["math"] == {"math": 10}
    assert len(report["calibration_results"]) == 40

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    table = calibrate(str(report_path), str(tmp_path / "gateway.db"))
    assert table["cheap"]["math"]["trials"] == 20
    stored = StatsStore(str(tmp_path / "gateway.db"))
    cheap = stored.get("cheap")
    assert cheap is not None and cheap.quality["classification"].successes == 20
    stored.close()


@pytest.mark.asyncio
async def test_replay_reproduces_a_deterministic_run(tmp_path: Path) -> None:
    rows = [math_case(i) for i in range(10)] + [classify_case(i) for i in range(10)]
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    models = [remote("cheap", 0.8, 0.1), remote("strong", 0.85, 10.0)]
    providers = ProviderFactory(httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream)))
    container = Container(
        ModelRegistry(models), providers, TelemetryStore(str(tmp_path / "t.db")), StatsStore()
    )
    report = await evaluate_container(
        container, load_dataset(str(dataset)), calibration_fraction=0.5,
        profiles=["balanced", "quality"], seed=2,
    )
    await providers.close()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = tmp_path / "models.yaml"
    config.write_text(json.dumps({"models": [m.model_dump(mode="json") for m in models]}), encoding="utf-8")

    replayed = await replay(str(report_path), str(config), dataset=str(dataset))
    assert replayed["mode"] == "replay"
    # Latency is taken from the recorded fixed-model row, so only it may differ.
    def without_latency(summary: dict) -> dict:
        return {p: {k: v for k, v in row.items() if k != "average_latency_ms"} for p, row in summary.items()}

    assert without_latency(replayed["summary"]) == without_latency(report["summary"])
    assert replayed["route_distribution"] == report["route_distribution"]


def test_cascade_policies_enable_the_cascade() -> None:
    case = EvaluationCase.model_validate(math_case(1))
    plain = case.request("auto", "quality").route
    cascade = case.request("auto", "quality+cascade").route
    assert plain is not None and not plain.cascade
    assert cascade is not None and cascade.cascade and cascade.profile == "quality"
