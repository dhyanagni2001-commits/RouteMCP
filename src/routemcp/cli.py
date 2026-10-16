from __future__ import annotations

import argparse
import json
import os

from .registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(prog="routemcp", description="RouteMCP model gateway")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the OpenAI-compatible gateway")
    serve.add_argument("--config", default="models.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    check = subparsers.add_parser("check-config", help="validate and summarize a model config")
    check.add_argument("--config", default="models.yaml")
    mcp = subparsers.add_parser("mcp", help="start the RouteMCP Streamable HTTP server")
    mcp.add_argument("--config", default="models.yaml")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8001)
    evaluation = subparsers.add_parser(
        "evaluate", help="compare RouteMCP with strongest and cheapest fixed-model baselines"
    )
    evaluation.add_argument("--config", default="models.yaml")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--output", default="evaluation-report.json")
    evaluation.add_argument("--strongest-model")
    evaluation.add_argument("--cheapest-model")
    evaluation.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.0,
        help="share of each task type used to calibrate quality before the held-out test",
    )
    evaluation.add_argument(
        "--profiles", default="balanced", help="comma-separated routing profiles to evaluate"
    )
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--concurrency", type=int, default=4)
    evaluation.add_argument("--retries", type=int, default=3)
    calibration = subparsers.add_parser(
        "calibrate", help="load per-task-type quality from an evaluation report into the gateway"
    )
    calibration.add_argument("--report", required=True)
    calibration.add_argument("--db", default=os.getenv("ROUTEMCP_DB", "routemcp.db"))
    replaying = subparsers.add_parser(
        "replay", help="re-route a finished evaluation offline with the current scoring code"
    )
    replaying.add_argument("--report", required=True)
    replaying.add_argument("--config", default="models.yaml")
    replaying.add_argument("--profiles", help="comma-separated profiles (default: the report's)")
    replaying.add_argument("--output", default="replay-report.json")
    args = parser.parse_args()

    if args.command == "serve":
        os.environ["ROUTEMCP_CONFIG"] = args.config
        import uvicorn

        uvicorn.run("routemcp.api:app", host=args.host, port=args.port, workers=args.workers)
    elif args.command == "check-config":
        registry = ModelRegistry.from_yaml(args.config)
        print(
            json.dumps(
                {
                    "valid": True,
                    "models": [model.id for model in registry.list()],
                    "count": len(registry.list()),
                }
            )
        )
    elif args.command == "mcp":
        from .api import load_container
        from .mcp_server import mcp_api_key, serve_mcp

        container = load_container(args.config)
        try:
            serve_mcp(container, args.host, args.port, mcp_api_key())
        finally:
            container.close_stores()
    elif args.command == "evaluate":
        from .evaluation import run_evaluation

        report = run_evaluation(
            config=args.config,
            dataset=args.dataset,
            output=args.output,
            strongest_model=args.strongest_model,
            cheapest_model=args.cheapest_model,
            calibration_fraction=args.calibration_fraction,
            profiles=[profile.strip() for profile in args.profiles.split(",") if profile.strip()],
            seed=args.seed,
            concurrency=args.concurrency,
            retries=args.retries,
        )
        print(json.dumps({"summary": report["summary"], "comparison": report["comparison"]}, indent=2))
    elif args.command == "calibrate":
        from .evaluation import calibrate

        print(json.dumps({"calibrated": calibrate(args.report, args.db)}, indent=2))
    elif args.command == "replay":
        from pathlib import Path

        from .evaluation import run_replay

        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()] if args.profiles else None
        report = run_replay(report_path=args.report, config=args.config, profiles=profiles)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"summary": report["summary"], "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
