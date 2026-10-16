#!/usr/bin/env python3
"""Exercise a running RouteMCP gateway and report reproducible E2E numbers."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

import httpx


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


async def load_test(
    operation: Callable[[], Awaitable[None]], requests: int, concurrency: int
) -> Dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors = 0

    async def one() -> None:
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                await operation()
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(requests)))
    duration = time.perf_counter() - started
    return {
        "requests": requests,
        "concurrency": concurrency,
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
    parser.add_argument("--url", default="http://127.0.0.1:18800")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--output", default="e2e-report.json")
    args = parser.parse_args()

    completion = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    }
    decision = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Debug this distributed race condition"}],
    }
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        health = await client.get("/healthz")
        health.raise_for_status()

        registration = await client.post(
            "/v1/router/models",
            json={"id": "e2e-persisted", "provider": "mock", "model": "e2e-persisted"},
        )
        if registration.status_code not in {201, 409}:
            registration.raise_for_status()

        probe = await client.post("/v1/router/models/e2e-persisted/probe")
        probe.raise_for_status()

        workflow = await client.post(
            "/v1/router/workflows",
            json={
                "steps": [
                    {"id": "plan", "request": {"messages": [{"role": "user", "content": "plan"}]}},
                    {"id": "format", "request": {"messages": [{"role": "user", "content": "format"}]}},
                ]
            },
        )
        workflow.raise_for_status()

        async def complete_once() -> None:
            response = await client.post("/v1/chat/completions", json=completion)
            response.raise_for_status()

        async def decide_once() -> None:
            response = await client.post("/v1/router/decide", json=decision)
            response.raise_for_status()

        async def stream_once() -> None:
            async with client.stream(
                "POST", "/v1/chat/completions", json={**completion, "stream": True}
            ) as response:
                response.raise_for_status()
                body = b"".join([chunk async for chunk in response.aiter_bytes()])
                if b"data: [DONE]" not in body:
                    raise RuntimeError("stream did not terminate correctly")

        completion_result = await load_test(complete_once, args.requests, args.concurrency)
        decision_result = await load_test(decide_once, args.requests, args.concurrency)
        streaming_result = await load_test(
            stream_once, max(50, args.requests // 5), args.concurrency
        )
        metrics = await client.get("/v1/router/metrics")
        metrics.raise_for_status()

    report = {
        "functional": {
            "health": health.json(),
            "registration_status": registration.status_code,
            "probe_healthy": probe.json()["healthy"],
            "workflow_steps": len(workflow.json()["steps"]),
            "stream_terminated": streaming_result["errors"] == 0,
        },
        "completion": completion_result,
        "decision": decision_result,
        "streaming": streaming_result,
        "recorded_metrics": metrics.json(),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
