#!/usr/bin/env python3
"""Exercise a running RouteMCP gateway and report reproducible E2E numbers.

Run against a fresh deployment (see ``benchmarks/run_e2e.sh``), then rerun with
``--verify-persisted`` after replacing the containers to confirm that models
registered through the API survive a restart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

PERSISTED_MODEL = "e2e-persisted"
UNREACHABLE_MODEL = "e2e-unreachable"


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


async def registered_model_ids(client: httpx.AsyncClient) -> List[str]:
    response = await client.get("/v1/router/models")
    response.raise_for_status()
    return [model["id"] for model in response.json()["data"]]


async def failover_scenario(client: httpx.AsyncClient, circuit_failures: int) -> Dict[str, Any]:
    """Prefer an unreachable upstream, then confirm fallback and circuit opening."""
    unreachable = {
        "id": UNREACHABLE_MODEL,
        "provider": "openai-compatible",
        # Port 9 (discard) is closed on loopback, so connections are refused immediately.
        "base_url": "http://127.0.0.1:9/v1",
        "model": "unreachable",
        "reliability": {
            "quality": 1.0,
            "success_rate": 1.0,
            "tool_success_rate": 1.0,
            "median_latency_ms": 1,
        },
    }
    request = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "route": {"profile": "quality"},
    }
    await client.delete(f"/v1/router/models/{UNREACHABLE_MODEL}")
    registration = await client.post("/v1/router/models", json=unreachable)
    registration.raise_for_status()
    try:
        fallbacks: List[Dict[str, Any]] = []
        failed_requests = 0
        for _ in range(circuit_failures):
            started = time.perf_counter()
            response = await client.post("/v1/chat/completions", json=request)
            latency = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                failed_requests += 1
                continue
            routed = response.json()["_routemcp"]
            fallbacks.append({**routed, "latency_ms": latency})

        models = (await client.get("/v1/router/models")).json()["data"]
        health = next(m["health"] for m in models if m["id"] == UNREACHABLE_MODEL) or {}

        started = time.perf_counter()
        after = await client.post("/v1/chat/completions", json=request)
        post_circuit_latency = (time.perf_counter() - started) * 1000
        after.raise_for_status()
        after_routed = after.json()["_routemcp"]
        trace = await client.get(f"/v1/router/traces/{after_routed['trace_id']}")
        trace.raise_for_status()
        rejected = next(
            c for c in trace.json()["decision"]["candidates"] if c["model_id"] == UNREACHABLE_MODEL
        )
    finally:
        await client.delete(f"/v1/router/models/{UNREACHABLE_MODEL}")

    successful = [
        f for f in fallbacks if f["attempts"] == 2 and f["selected_model"] != UNREACHABLE_MODEL
    ]
    return {
        "scenario": "Unreachable OpenAI-compatible upstream followed by healthy mock model",
        "successful_fallbacks": len(successful),
        "failed_requests": failed_requests,
        "attempts_before_circuit": max((f["attempts"] for f in fallbacks), default=0),
        "attempts_after_circuit": after_routed["attempts"],
        "circuit_opened_after_failures": health.get("consecutive_failures"),
        "circuit_open": health.get("circuit_open") is True,
        "post_circuit_rejection": rejected["reasons"],
        "median_failover_latency_ms": round(
            statistics.median(f["latency_ms"] for f in fallbacks), 2
        )
        if fallbacks
        else None,
        "post_circuit_latency_ms": round(post_circuit_latency, 2),
    }


async def mcp_scenario(url: str, api_key: Optional[str]) -> Dict[str, Any]:
    try:
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        return {"skipped": "MCP client requires Python 3.11+ and `pip install 'routemcp[mcp]'`"}

    unauthenticated_status = None
    if api_key:
        async with httpx.AsyncClient(timeout=10) as probe:
            unauthenticated_status = (await probe.post(url, json={})).status_code
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    http_client = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=300))
    messages = [{"role": "user", "content": "Debug this distributed race condition"}]
    async with http_client, Client(streamable_http_client(url, http_client=http_client)) as client:
        tools = sorted(tool.name for tool in (await client.list_tools()).tools)
        decide = await client.call_tool("route_decide", {"messages": messages})
        complete = await client.call_tool("route_complete", {"messages": messages})
        listed = await client.call_tool("list_routed_models", {})
        listed_ids = [model["id"] for model in (listed.structured_content or {}).get("result", [])]
        selected = ((decide.structured_content or {}).get("result") or {}).get("selected_model")
        return {
            "transport": "Streamable HTTP",
            "protocol_version": getattr(client.session, "protocol_version", None),
            "handshake": True,
            "tools": tools,
            "tool_discovery": tools
            == sorted(["route_decide", "route_complete", "list_routed_models"]),
            "route_decide_call": not decide.is_error and selected is not None,
            "route_complete_call": not complete.is_error,
            "list_routed_models_call": not listed.is_error,
            "listed_models": listed_ids,
            "auth_enforced": unauthenticated_status == 401 if api_key else None,
        }


async def verify_persisted(args: argparse.Namespace, headers: Dict[str, str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=args.url, timeout=30, headers=headers) as client:
        (await client.get("/healthz")).raise_for_status()
        model_ids = await registered_model_ids(client)
    result: Dict[str, Any] = {
        "registration_survived_container_replacement": PERSISTED_MODEL in model_ids,
        "registered_models_after_restart": len(model_ids),
        "unreachable_model_removed": UNREACHABLE_MODEL not in model_ids,
    }
    checks = [
        result["registration_survived_container_replacement"],
        result["unreachable_model_removed"],
    ]
    if args.mcp_url:
        mcp = await mcp_scenario(args.mcp_url, args.mcp_api_key)
        if "skipped" not in mcp:
            result["mcp_sees_persisted_model"] = PERSISTED_MODEL in mcp["listed_models"]
            checks.append(result["mcp_sees_persisted_model"])
    result["passed"] = all(checks)
    return result


async def run(args: argparse.Namespace, headers: Dict[str, str]) -> Dict[str, Any]:
    completion = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    }
    decision = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Debug this distributed race condition"}],
    }
    async with httpx.AsyncClient(base_url=args.url, timeout=30, headers=headers) as client:
        health = await client.get("/healthz")
        health.raise_for_status()

        registration = await client.post(
            "/v1/router/models",
            json={"id": PERSISTED_MODEL, "provider": "mock", "model": PERSISTED_MODEL},
        )
        if registration.status_code not in {201, 409}:
            registration.raise_for_status()

        probe = await client.post(f"/v1/router/models/{PERSISTED_MODEL}/probe")
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
        failover = await failover_scenario(client, args.circuit_failures)
        metrics = await client.get("/v1/router/metrics")
        metrics.raise_for_status()

    mcp = await mcp_scenario(args.mcp_url, args.mcp_api_key) if args.mcp_url else None

    functional = {
        "health_check": health.json().get("status") == "ok",
        "model_registration": registration.status_code in {201, 409},
        "model_probe": probe.json()["healthy"] is True,
        "two_step_workflow": len(workflow.json()["steps"]) == 2,
        "stream_terminated": streaming_result["errors"] == 0,
    }
    checks = list(functional.values()) + [
        completion_result["errors"] == 0,
        decision_result["errors"] == 0,
        failover["successful_fallbacks"] == args.circuit_failures,
        failover["failed_requests"] == 0,
        failover["attempts_after_circuit"] == 1,
        failover["circuit_open"],
    ]
    if mcp is not None and "skipped" not in mcp:
        checks += [mcp["tool_discovery"], mcp["route_decide_call"], mcp["route_complete_call"]]
        if mcp["auth_enforced"] is not None:
            checks.append(mcp["auth_enforced"])

    report: Dict[str, Any] = {
        "passed": all(checks),
        "functional": functional,
        "completion": completion_result,
        "decision": decision_result,
        "streaming": streaming_result,
        "failover": failover,
        "recorded_metrics": metrics.json(),
    }
    if mcp is not None:
        report["mcp"] = mcp
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mcp-url", help="Streamable HTTP MCP endpoint, e.g. http://127.0.0.1:8001/mcp")
    parser.add_argument("--api-key", default=os.getenv("ROUTEMCP_API_KEY") or None)
    parser.add_argument(
        "--mcp-api-key",
        default=os.getenv("ROUTEMCP_MCP_API_KEY") or os.getenv("ROUTEMCP_API_KEY") or None,
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument(
        "--circuit-failures",
        type=int,
        default=int(os.getenv("ROUTEMCP_CIRCUIT_FAILURES", "3")),
        help="the gateway's ROUTEMCP_CIRCUIT_FAILURES setting",
    )
    parser.add_argument(
        "--verify-persisted",
        action="store_true",
        help="only check that the model registered by a previous run survived a restart",
    )
    parser.add_argument("--output", default="e2e-report.json")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    if args.verify_persisted:
        report = asyncio.run(verify_persisted(args, headers))
    else:
        report = asyncio.run(run(args, headers))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
