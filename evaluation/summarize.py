#!/usr/bin/env python3
"""Print a Markdown summary of a `routemcp evaluate` report."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


def pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def usd(value: Any) -> str:
    return "—" if value is None else f"${value:.4f}"


def main(path: str) -> None:
    report: Dict[str, Any] = json.loads(open(path, encoding="utf-8").read())
    summary = report["summary"]
    split = report["split"]
    print(
        f"Held-out test: {split['test_tasks']} tasks "
        f"(calibration: {split['calibration_tasks']}, seed {split['seed']})\n"
    )
    print("| Policy | Success | Total cost | Cost / success | Avg latency |")
    print("| --- | ---: | ---: | ---: | ---: |")
    order = sorted(summary, key=lambda p: (not p.startswith("fixed:"), p == "oracle", p))
    for policy in order:
        row = summary[policy]
        print(
            f"| {policy} | {pct(row['success_rate'])} | {usd(row['total_cost_usd'])} | "
            f"{usd(row['cost_per_success_usd'])} | {row['average_latency_ms']:.0f} ms |"
        )
    print(f"\nVersus fixed {report['strongest_model']}:\n")
    print("| Routed policy | Quality retained | Cost reduction |")
    print("| --- | ---: | ---: |")
    for policy, comparison in report["comparison"].items():
        print(
            f"| {policy} | {pct(comparison['quality_retention_vs_strongest'])} | "
            f"{pct(comparison['cost_reduction_vs_strongest'])} |"
        )
    policies = [p for p in order if p.startswith("fixed:") or p.startswith("routemcp:")]
    print("\nSuccess rate by task type:\n")
    print("| Task type | " + " | ".join(policies) + " |")
    print("| --- |" + " ---: |" * len(policies))
    for task_type, rates in report["by_task_type"].items():
        print(f"| {task_type} | " + " | ".join(pct(rates.get(p)) for p in policies) + " |")
    for policy, distribution in report["route_distribution"].items():
        print(f"\n{policy} final model by task type:\n")
        for task_type, choices in distribution.items():
            print(f"- {task_type}: " + ", ".join(f"{m} ×{n}" for m, n in sorted(choices.items())))


if __name__ == "__main__":
    main(sys.argv[1])
