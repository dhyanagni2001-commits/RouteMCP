from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderKind(str, Enum):
    OPENAI_COMPATIBLE = "openai-compatible"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    MOCK = "mock"


class Cost(BaseModel):
    input_per_million: float = Field(default=0, ge=0)
    output_per_million: float = Field(default=0, ge=0)


class Capabilities(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tools: bool = False
    json_output: bool = Field(default=True, alias="json")
    vision: bool = False
    context_window: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)


class Reliability(BaseModel):
    success_rate: float = Field(default=0.95, ge=0, le=1)
    tool_success_rate: float = Field(default=0.85, ge=0, le=1)
    median_latency_ms: float = Field(default=1000, ge=0)
    quality: float = Field(default=0.70, ge=0, le=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    provider: ProviderKind = ProviderKind.OPENAI_COMPATIBLE
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    enabled: bool = True
    local: bool = False
    privacy: Literal["local", "standard", "restricted"] = "standard"
    cost: Cost = Field(default_factory=Cost)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    reliability: Reliability = Field(default_factory=Reliability)
    headers: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "ModelConfig":
        if self.provider in {ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OLLAMA} and not self.base_url:
            raise ValueError("base_url is required for OpenAI-compatible and Ollama models")
        if self.provider in {ProviderKind.OPENAI, ProviderKind.ANTHROPIC, ProviderKind.GEMINI} and not self.api_key_env:
            raise ValueError(f"api_key_env is required for {self.provider.value} models")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message] = Field(min_length=1)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: bool = False
    route: Optional["RouteOptions"] = None


class RouteOptions(BaseModel):
    profile: Literal["cheapest", "fastest", "balanced", "quality"] = "balanced"
    max_cost_usd: Optional[float] = Field(default=None, ge=0)
    max_latency_ms: Optional[float] = Field(default=None, gt=0)
    privacy: Literal["standard", "local-only"] = "standard"
    task_type: Optional[str] = None
    complexity: Optional[float] = Field(default=None, ge=0, le=1)
    validate_json: bool = True
    max_attempts: int = Field(default=3, ge=1, le=10)


class RequestFeatures(BaseModel):
    estimated_input_tokens: int
    expected_output_tokens: int
    required_tokens: int
    task_type: str
    complexity: float
    requires_tools: bool
    requires_json: bool
    requires_vision: bool
    privacy: str


class CandidateScore(BaseModel):
    model_id: str
    eligible: bool
    score: Optional[float] = None
    estimated_cost_usd: float = 0
    reasons: List[str] = Field(default_factory=list)
    components: Dict[str, float] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    selected_model: Optional[str]
    profile: str
    features: RequestFeatures
    candidates: List[CandidateScore]
    explanation: str


class Attempt(BaseModel):
    model_id: str
    status: str
    latency_ms: float
    estimated_cost_usd: float
    actual_cost_usd: float
    error_type: Optional[str] = None
    error: Optional[str] = None


class WorkflowStep(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    request: ChatRequest


class WorkflowRequest(BaseModel):
    steps: List[WorkflowStep] = Field(min_length=1, max_length=50)
    max_total_cost_usd: Optional[float] = Field(default=None, ge=0)


ChatRequest.model_rebuild()
