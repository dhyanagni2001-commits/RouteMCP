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
        from .mcp_server import create_mcp_server

        container = load_container(args.config)
        server = create_mcp_server(container)
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path="/mcp",
        )
    elif args.command == "evaluate":
        from .evaluation import run_evaluation

        report = run_evaluation(
            config=args.config,
            dataset=args.dataset,
            output=args.output,
            strongest_model=args.strongest_model,
            cheapest_model=args.cheapest_model,
        )
        print(json.dumps({"summary": report["summary"], "comparison": report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
