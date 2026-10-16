from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .domain import ChatRequest, ModelConfig, RouteOptions, WorkflowRequest
from .providers import ProviderFactory
from .registry import ModelRegistry
from .router import Router, RoutingError
from .telemetry import TelemetryStore


class Container:
    def __init__(self, registry: ModelRegistry, providers: ProviderFactory, telemetry: TelemetryStore):
        self.registry = registry
        self.providers = providers
        self.telemetry = telemetry
        self.router = Router(registry, providers, telemetry)


def load_container(config_path: Optional[str] = None, db_path: Optional[str] = None) -> Container:
    config = config_path or os.getenv("ROUTEMCP_CONFIG", "models.yaml")
    registry = ModelRegistry.from_yaml(config) if Path(config).exists() else ModelRegistry()
    telemetry = TelemetryStore(db_path or os.getenv("ROUTEMCP_DB", "routemcp.db"))
    return Container(registry, ProviderFactory(), telemetry)


def create_app(container: Optional[Container] = None) -> FastAPI:
    owned = container is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or load_container()
        yield
        if owned:
            await app.state.container.providers.close()
            app.state.container.telemetry.close()

    app = FastAPI(
        title="RouteMCP",
        version="0.1.0",
        description="Capability-aware OpenAI-compatible model router",
        lifespan=lifespan,
    )
    # Also initialize injected containers eagerly for ASGI clients that do not
    # drive lifespan events (common in tests and embedded deployments).
    if container is not None:
        app.state.container = container

    def get_container(request: Request) -> Container:
        return request.app.state.container

    async def authenticate(authorization: Optional[str] = Header(default=None)) -> None:
        configured = os.getenv("ROUTEMCP_API_KEY")
        if not configured:
            return
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not hmac.compare_digest(supplied, configured):
            raise HTTPException(status_code=401, detail="invalid RouteMCP API key")

    protected = [Depends(authenticate)]

    @app.exception_handler(RoutingError)
    async def routing_error_handler(_: Request, exc: RoutingError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc), "type": "routing_error", "trace_id": exc.trace_id}},
        )

    @app.get("/healthz")
    async def health(container: Container = Depends(get_container)) -> Dict[str, Any]:
        return {"status": "ok", "models": len(container.registry.list())}

    @app.get("/v1/router/models", dependencies=protected)
    async def list_models(container: Container = Depends(get_container)) -> Dict[str, Any]:
        return {
            "data": [
                {
                    **model.model_dump(mode="json", by_alias=True, exclude={"headers"}),
                    "health": container.registry.health(model.id),
                }
                for model in container.registry.list()
            ]
        }

    @app.post("/v1/router/models", status_code=201, dependencies=protected)
    async def register_model(model: ModelConfig, container: Container = Depends(get_container)) -> Dict[str, Any]:
        try:
            container.registry.register(model)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return model.model_dump(mode="json", by_alias=True, exclude={"headers"})

    @app.put("/v1/router/models/{model_id}", dependencies=protected)
    async def update_model(model_id: str, model: ModelConfig, container: Container = Depends(get_container)) -> Dict[str, Any]:
        if model.id != model_id:
            raise HTTPException(status_code=400, detail="path and body model ids must match")
        try:
            container.registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        container.registry.register(model, replace=True)
        return model.model_dump(mode="json", by_alias=True, exclude={"headers"})

    @app.delete("/v1/router/models/{model_id}", status_code=204, dependencies=protected)
    async def delete_model(model_id: str, container: Container = Depends(get_container)) -> None:
        try:
            container.registry.remove(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/router/models/{model_id}/probe", dependencies=protected)
    async def probe_model(model_id: str, container: Container = Depends(get_container)) -> Dict[str, Any]:
        try:
            model = container.registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = await container.providers.create(model).health_check()
        container.registry.record_health(model_id, result)
        return {
            "model_id": model_id,
            **result,
            "capabilities": model.capabilities.model_dump(by_alias=True),
        }

    @app.post("/v1/router/decide", dependencies=protected)
    async def route_decision(request: ChatRequest, container: Container = Depends(get_container)) -> Dict[str, Any]:
        return container.router.decision(request).model_dump(mode="json")

    @app.post("/v1/chat/completions", dependencies=protected)
    async def chat_completion(request: ChatRequest, container: Container = Depends(get_container)) -> Dict[str, Any]:
        return await container.router.execute(request)

    @app.post("/v1/router/workflows", dependencies=protected)
    async def workflow(request: WorkflowRequest, container: Container = Depends(get_container)) -> Dict[str, Any]:
        results = []
        total_cost = 0.0
        for step in request.steps:
            step_request = step.request.model_copy(deep=True)
            remaining = None if request.max_total_cost_usd is None else request.max_total_cost_usd - total_cost
            if remaining is not None:
                if remaining < 0:
                    raise RoutingError("workflow cost budget exhausted", status_code=422)
                options = step_request.route or RouteOptions()
                if options.max_cost_usd is None or options.max_cost_usd > remaining:
                    step_request.route = options.model_copy(update={"max_cost_usd": remaining})
            result = await container.router.execute(step_request)
            cost = float(result["_routemcp"]["actual_cost_usd"])
            total_cost += cost
            results.append({"step_id": step.id, "response": result})
        return {"object": "router.workflow", "steps": results, "total_cost_usd": round(total_cost, 8)}

    @app.get("/v1/router/metrics", dependencies=protected)
    async def metrics(container: Container = Depends(get_container)) -> Dict[str, Any]:
        return container.telemetry.metrics()

    @app.get("/v1/router/traces/{trace_id}", dependencies=protected)
    async def trace(trace_id: str, container: Container = Depends(get_container)) -> Dict[str, Any]:
        result = container.telemetry.get(trace_id)
        if result is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return result

    return app


app = create_app()
