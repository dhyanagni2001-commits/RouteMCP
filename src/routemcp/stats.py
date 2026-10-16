"""Learned per-model statistics that feed back into routing.

Two kinds of evidence are kept, because they come from different sources:

* Operational signals (latency, success rate, tool-call success) are observed on
  every live attempt and tracked as exponentially weighted moving averages.
* Answer quality cannot be observed from live traffic without ground truth, so it
  is calibrated per model and task type from evaluation runs (``routemcp
  calibrate``), where each answer is checked by a deterministic validator.

Scoring blends both with the YAML priors; see ``scoring.effective_profile``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class QualityEvidence(BaseModel):
    trials: int = 0
    successes: int = 0


class ModelStats(BaseModel):
    samples: int = 0
    success_rate: Optional[float] = None
    latency_samples: int = 0
    latency_ms: Optional[float] = None
    tool_samples: int = 0
    tool_success_rate: Optional[float] = None
    quality: Dict[str, QualityEvidence] = Field(default_factory=dict)


def _ewma(previous: Optional[float], value: float, alpha: float) -> float:
    return value if previous is None else previous + alpha * (value - previous)


class StatsStore:
    """Thread-safe learned statistics with optional SQLite persistence."""

    def __init__(self, path: Optional[Union[str, Path]] = None, alpha: float = 0.1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")
        self.alpha = alpha
        self._stats: Dict[str, ModelStats] = {}
        self._lock = RLock()
        self._connection: Optional[sqlite3.Connection] = None
        if path is not None:
            if str(path) != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(path), check_same_thread=False)
            with self._connection:
                self._connection.execute("PRAGMA busy_timeout=5000")
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS model_stats (
                        model_id TEXT PRIMARY KEY, stats_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
            for model_id, stats_json in self._connection.execute(
                "SELECT model_id, stats_json FROM model_stats"
            ):
                self._stats[model_id] = ModelStats.model_validate_json(stats_json)

    def _persist(self, model_id: str) -> None:
        if self._connection is None:
            return
        stats = self._stats.get(model_id)
        with self._connection:
            if stats is None:
                self._connection.execute("DELETE FROM model_stats WHERE model_id = ?", (model_id,))
            else:
                self._connection.execute(
                    """INSERT INTO model_stats (model_id, stats_json, updated_at)
                       VALUES (?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(model_id) DO UPDATE SET stats_json=excluded.stats_json,
                       updated_at=CURRENT_TIMESTAMP""",
                    (model_id, stats.model_dump_json()),
                )

    def observe(
        self,
        model_id: str,
        *,
        success: bool,
        latency_ms: Optional[float] = None,
        tool_call: bool = False,
    ) -> None:
        """Record one live attempt. Latency is only learned from successful attempts."""
        with self._lock:
            stats = self._stats.setdefault(model_id, ModelStats())
            stats.samples += 1
            stats.success_rate = _ewma(stats.success_rate, float(success), self.alpha)
            if success and latency_ms is not None:
                stats.latency_samples += 1
                stats.latency_ms = _ewma(stats.latency_ms, latency_ms, self.alpha)
            if tool_call:
                stats.tool_samples += 1
                stats.tool_success_rate = _ewma(stats.tool_success_rate, float(success), self.alpha)
            self._persist(model_id)

    def record_quality(self, model_id: str, task_type: str, trials: int, successes: int) -> None:
        """Replace calibrated quality evidence for one model and task type."""
        if trials < 0 or not 0 <= successes <= trials:
            raise ValueError("successes must be between 0 and trials")
        with self._lock:
            stats = self._stats.setdefault(model_id, ModelStats())
            stats.quality[task_type] = QualityEvidence(trials=trials, successes=successes)
            self._persist(model_id)

    def get(self, model_id: str) -> Optional[ModelStats]:
        with self._lock:
            stats = self._stats.get(model_id)
            return stats.model_copy(deep=True) if stats is not None else None

    def snapshot(self) -> Dict[str, ModelStats]:
        with self._lock:
            return {key: value.model_copy(deep=True) for key, value in self._stats.items()}

    def reset(self, model_id: str) -> None:
        with self._lock:
            self._stats.pop(model_id, None)
            self._persist(model_id)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
