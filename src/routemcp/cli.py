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


if __name__ == "__main__":
    main()
