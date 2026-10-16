from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from .domain import ChatRequest, ModelConfig, ProviderKind


class ProviderError(RuntimeError):
    def __init__(self, message: str, error_type: str = "provider_error", retryable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class ProviderResult:
    def __init__(self, payload: Dict[str, Any], latency_ms: float):
        self.payload = payload
        self.latency_ms = latency_ms

    @property
    def usage(self) -> Dict[str, int]:
        raw = self.payload.get("usage") or {}
        return {
            "prompt_tokens": int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0),
            "completion_tokens": int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0),
        }


class ModelProvider(ABC):
    def __init__(self, config: ModelConfig, client: httpx.AsyncClient):
        self.config = config
        self.client = client

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ProviderResult:
        raise NotImplementedError

    async def health_check(self) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            response = await self.client.get(self._health_url(), headers=self._headers(), timeout=5)
            healthy = response.status_code < 500
            detail = f"HTTP {response.status_code}"
        except (httpx.HTTPError, ProviderError) as exc:
            healthy, detail = False, str(exc)
        return {
            "healthy": healthy,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "detail": detail,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def _headers(self) -> Dict[str, str]:
        headers = dict(self.config.headers)
        if self.config.api_key_env:
            key = os.getenv(self.config.api_key_env)
            if not key:
                raise ProviderError(
                    f"required environment variable {self.config.api_key_env} is not set",
                    "configuration_error",
                    retryable=False,
                )
            headers.setdefault("Authorization", f"Bearer {key}")
        return headers

    def _health_url(self) -> str:
        if self.config.provider == ProviderKind.OPENAI:
            return (self.config.base_url or "https://api.openai.com/v1").rstrip("/") + "/models"
        if self.config.provider == ProviderKind.GEMINI:
            base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            return base.rstrip("/") + "/models"
        if self.config.provider == ProviderKind.ANTHROPIC:
            return (self.config.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/models"
        return (self.config.base_url or "").rstrip("/") + "/models"


class OpenAICompatibleProvider(ModelProvider):
    async def complete(self, request: ChatRequest) -> ProviderResult:
        base_url = self.config.base_url
        if self.config.provider == ProviderKind.OPENAI:
            base_url = base_url or "https://api.openai.com/v1"
        elif self.config.provider == ProviderKind.GEMINI:
            base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
        assert base_url
        payload = request.model_dump(exclude={"route"}, exclude_none=True)
        payload["model"] = self.config.model
        payload["stream"] = False
        started = time.perf_counter()
        try:
            response = await self.client.post(
                base_url.rstrip("/") + "/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderError(str(exc), "timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            error_type = "rate_limit" if status == 429 else "provider_unavailable" if status >= 500 else "provider_error"
            raise ProviderError(
                f"provider returned HTTP {status}: {exc.response.text[:300]}",
                error_type,
                retryable=status == 429 or status >= 500,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(str(exc), "provider_unavailable") from exc
        return ProviderResult(data, (time.perf_counter() - started) * 1000)


class AnthropicProvider(ModelProvider):
    def _headers(self) -> Dict[str, str]:
        headers = dict(self.config.headers)
        if not self.config.api_key_env or not os.getenv(self.config.api_key_env):
            raise ProviderError("Anthropic API key environment variable is not set", "configuration_error", False)
        headers.setdefault("x-api-key", os.environ[self.config.api_key_env])
        headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    async def complete(self, request: ChatRequest) -> ProviderResult:
        messages = [m.model_dump(exclude_none=True) for m in request.messages if m.role != "system"]
        system = "\n".join(str(m.content) for m in request.messages if m.role == "system") or None
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = []
            for tool in request.tools:
                function = tool.get("function", tool)
                converted = {
                    "name": function["name"],
                    "input_schema": function.get("parameters", function.get("input_schema", {})),
                }
                if function.get("description"):
                    converted["description"] = function["description"]
                payload["tools"].append(converted)
        started = time.perf_counter()
        try:
            response = await self.client.post(
                (self.config.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/messages",
                headers=self._headers(), json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderError(str(exc), "timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            error_type = "rate_limit" if status == 429 else "provider_unavailable" if status >= 500 else "provider_error"
            raise ProviderError(
                f"provider returned HTTP {status}: {exc.response.text[:300]}",
                error_type,
                retryable=status == 429 or status >= 500,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(str(exc), "provider_unavailable") from exc
        content = raw.get("content", [])
        text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        tool_calls = [
            {
                "id": part.get("id", f"toolu_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": part["name"],
                    "arguments": json.dumps(part.get("input", {})),
                },
            }
            for part in content
            if part.get("type") == "tool_use"
        ]
        message: Dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
        }.get(raw.get("stop_reason"), raw.get("stop_reason"))
        data = {
            "id": raw.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": raw.get("usage", {}),
        }
        return ProviderResult(data, (time.perf_counter() - started) * 1000)


class MockProvider(ModelProvider):
    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "latency_ms": 0,
            "detail": "mock provider ready",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    async def complete(self, request: ChatRequest) -> ProviderResult:
        content = str(request.messages[-1].content)
        if "FAIL_JSON" in content and self.config.reliability.quality < 0.8:
            answer = "not valid json"
        elif request.response_format:
            answer = json.dumps({"result": content[:80], "model": self.config.id})
        else:
            answer = f"[{self.config.id}] {content}"
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, len(content) // 4), "completion_tokens": max(1, len(answer) // 4)},
        }
        return ProviderResult(payload, self.config.reliability.median_latency_ms)


class ProviderFactory:
    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))

    def create(self, config: ModelConfig) -> ModelProvider:
        if config.provider == ProviderKind.ANTHROPIC:
            return AnthropicProvider(config, self.client)
        if config.provider == ProviderKind.MOCK:
            return MockProvider(config, self.client)
        return OpenAICompatibleProvider(config, self.client)

    async def close(self) -> None:
        await self.client.aclose()
