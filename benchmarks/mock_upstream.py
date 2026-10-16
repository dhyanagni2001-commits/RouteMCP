"""Deterministic OpenAI-compatible upstream for gateway-overhead benchmarks only."""

import asyncio
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI

app = FastAPI()


@app.get("/v1/models")
async def models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": "benchmark-model", "object": "model"}]}


@app.post("/v1/chat/completions")
async def completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.005)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "benchmark-model"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 6, "completion_tokens": 1},
    }
