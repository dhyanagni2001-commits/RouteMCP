#!/usr/bin/env python3
"""Measure RouteMCP's added latency against the same provider called directly."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any, Dict, List

import httpx


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


async def measure(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    requests: int,
    concurrency: int,
) -> Dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors = 0

    async def one() -> None:
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(requests)))
    duration = time.perf_counter() - started
    return {
        "requests": requests,
        "errors": errors,
        "throughput_rps": round(requests / duration, 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--direct-url", required=True)
    parser.add_argument("--gateway-model", required=True)
    parser.add_argument("--direct-model", required=True)
    parser.add_argument("--gateway-key", default="")
    parser.add_argument("--direct-key", default="")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="benchmark-report.json")
    args = parser.parse_args()

    base_payload = {
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    gateway_payload = {**base_payload, "model": args.gateway_model}
    direct_payload = {**base_payload, "model": args.direct_model}
    gateway_headers = {"Authorization": f"Bearer {args.gateway_key}"} if args.gateway_key else {}
    direct_headers = {"Authorization": f"Bearer {args.direct_key}"} if args.direct_key else {}

    async with httpx.AsyncClient(timeout=120) as client:
        await measure(client, args.gateway_url, gateway_headers, gateway_payload, 3, 1)
        await measure(client, args.direct_url, direct_headers, direct_payload, 3, 1)
        runs = []
        for round_index in range(args.rounds):
            calls = {
                "gateway": lambda: measure(
                    client,
                    args.gateway_url,
                    gateway_headers,
                    gateway_payload,
                    args.requests,
                    args.concurrency,
                ),
                "direct": lambda: measure(
                    client,
                    args.direct_url,
                    direct_headers,
                    direct_payload,
                    args.requests,
                    args.concurrency,
                ),
            }
            order = ["gateway", "direct"] if round_index % 2 == 0 else ["direct", "gateway"]
            result = {}
            for name in order:
                result[name] = await calls[name]()
            runs.append(result)

    def aggregate(name: str) -> Dict[str, Any]:
        return {
            "requests_per_round": args.requests,
            "rounds": args.rounds,
            "errors": sum(run[name]["errors"] for run in runs),
            "median_throughput_rps": round(
                statistics.median(run[name]["throughput_rps"] for run in runs), 2
            ),
            "latency_ms": {
                metric: round(
                    statistics.median(run[name]["latency_ms"][metric] for run in runs), 2
                )
                for metric in ("mean", "p50", "p95", "p99")
            },
        }

    gateway = aggregate("gateway")
    direct = aggregate("direct")
    report = {
        "gateway": gateway,
        "direct": direct,
        "added_latency_ms": {
            key: round(gateway["latency_ms"][key] - direct["latency_ms"][key], 2)
            for key in ("mean", "p50", "p95", "p99")
        },
        "methodology": (
            "Alternating direct and gateway phases against the same upstream model; warm-up "
            "requests excluded. Reported values are medians across rounds. Run close to both "
            "endpoints and repeat before publishing results."
        ),
        "runs": runs,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
