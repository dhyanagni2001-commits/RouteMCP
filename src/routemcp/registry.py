from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from .domain import ModelConfig


class ConfigFile(BaseModel):
    models: List[ModelConfig] = Field(default_factory=list)


class ModelRegistry:
    """Thread-safe registry with optional SQLite persistence and circuit breakers."""

    def __init__(
        self,
        models: Iterable[ModelConfig] = (),
        storage_path: Optional[Union[str, Path]] = None,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30,
    ) -> None:
        self._models: Dict[str, ModelConfig] = {}
        self._health: Dict[str, Dict[str, Any]] = {}
        self._failures: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_seconds = max(1, cooldown_seconds)
        self._lock = RLock()
        self._connection: Optional[sqlite3.Connection] = None
        for model in models:
            self.register(model)
        if storage_path is not None:
            self._open_storage(storage_path)
            self._load_persisted()

    def _open_storage(self, path: Union[str, Path]) -> None:
        storage = Path(path)
        storage.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(storage, check_same_thread=False)
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS registered_models (
                    id TEXT PRIMARY KEY, config_json TEXT, deleted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def _load_persisted(self) -> None:
        assert self._connection is not None
        rows = self._connection.execute(
            "SELECT id, config_json, deleted FROM registered_models"
        ).fetchall()
        for model_id, config_json, deleted in rows:
            if deleted:
                self._models.pop(model_id, None)
            else:
                model = ModelConfig.model_validate_json(config_json)
                self._models[model.id] = model

    def _persist(self, model_id: str, model: Optional[ModelConfig]) -> None:
        if self._connection is None:
            return
        payload = model.model_dump_json(by_alias=True) if model is not None else None
        with self._connection:
            self._connection.execute(
                """INSERT INTO registered_models (id, config_json, deleted, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(id) DO UPDATE SET config_json=excluded.config_json,
                   deleted=excluded.deleted, updated_at=CURRENT_TIMESTAMP""",
                (model_id, payload, int(model is None)),
            )

    def register(
        self, model: ModelConfig, replace: bool = False, persist: bool = False
    ) -> ModelConfig:
        with self._lock:
            if model.id in self._models and not replace:
                raise ValueError(f"model '{model.id}' already exists")
            self._models[model.id] = model
            if replace:
                self._health.pop(model.id, None)
                self._failures.pop(model.id, None)
                self._open_until.pop(model.id, None)
            if persist:
                self._persist(model.id, model)
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

    def remove(self, model_id: str, persist: bool = False) -> None:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"unknown model '{model_id}'")
            del self._models[model_id]
            self._health.pop(model_id, None)
            self._failures.pop(model_id, None)
            self._open_until.pop(model_id, None)
            if persist:
                self._persist(model_id, None)

    def record_health(self, model_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"unknown model '{model_id}'")
            self._health[model_id] = dict(result)
            if result.get("healthy"):
                self.record_success(model_id)
            else:
                # A failed probe opens the circuit like repeated live failures do, so the
                # model is retried (half-open) after the cooldown instead of staying
                # excluded until someone probes it again.
                self._failures[model_id] = max(
                    self._failures.get(model_id, 0), self._failure_threshold
                )
                self._open_until[model_id] = time.time() + self._cooldown_seconds
                status = self._health[model_id]
                status.update(
                    {
                        "passive_healthy": False,
                        "consecutive_failures": self._failures[model_id],
                        "circuit_open": True,
                        "circuit_open_until": self._open_until[model_id],
                    }
                )

    def record_success(self, model_id: str) -> None:
        with self._lock:
            self._failures.pop(model_id, None)
            self._open_until.pop(model_id, None)
            status = self._health.setdefault(model_id, {})
            status.update(
                {"passive_healthy": True, "consecutive_failures": 0, "circuit_open": False}
            )
            status.pop("half_open", None)
            status.pop("circuit_open_until", None)
            if "healthy" in status:
                # A successful live call supersedes an older failed probe.
                status["healthy"] = True

    def record_failure(self, model_id: str, error: str) -> None:
        with self._lock:
            failures = self._failures.get(model_id, 0) + 1
            self._failures[model_id] = failures
            status = self._health.setdefault(model_id, {})
            status.update(
                {
                    "passive_healthy": False,
                    "consecutive_failures": failures,
                    "last_error": error[:300],
                    "circuit_open": False,
                }
            )
            if failures >= self._failure_threshold:
                self._open_until[model_id] = time.time() + self._cooldown_seconds
                status["circuit_open"] = True
                status["circuit_open_until"] = self._open_until[model_id]

    def health(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            result = dict(self._health.get(model_id, {}))
            open_until = self._open_until.get(model_id)
            if open_until is not None and open_until <= time.time():
                self._open_until.pop(model_id, None)
                result["circuit_open"] = False
                result["half_open"] = True
                self._health[model_id] = dict(result)
            elif open_until is not None:
                result["circuit_open"] = True
                result["circuit_open_until"] = open_until
            return result or None

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path],
        storage_path: Optional[Union[str, Path]] = None,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30,
    ) -> "ModelRegistry":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(
            ConfigFile.model_validate(raw).models,
            storage_path=storage_path,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )
