from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from .domain import ModelConfig


class ConfigFile(BaseModel):
    models: List[ModelConfig] = Field(default_factory=list)


class ModelRegistry:
    def __init__(self, models: Iterable[ModelConfig] = ()) -> None:
        self._models: Dict[str, ModelConfig] = {}
        self._health: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        for model in models:
            self.register(model)

    def register(self, model: ModelConfig, replace: bool = False) -> ModelConfig:
        with self._lock:
            if model.id in self._models and not replace:
                raise ValueError(f"model '{model.id}' already exists")
            self._models[model.id] = model
            if replace:
                self._health.pop(model.id, None)
        return model

    def get(self, model_id: str) -> ModelConfig:
        with self._lock:
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise KeyError(f"unknown model '{model_id}'") from exc

    def list(self) -> List[ModelConfig]:
        with self._lock:
            return list(self._models.values())

    def remove(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"unknown model '{model_id}'")
            del self._models[model_id]
            self._health.pop(model_id, None)

    def record_health(self, model_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"unknown model '{model_id}'")
            self._health[model_id] = dict(result)

    def health(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            result = self._health.get(model_id)
            return dict(result) if result is not None else None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelRegistry":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(ConfigFile.model_validate(raw).models)
