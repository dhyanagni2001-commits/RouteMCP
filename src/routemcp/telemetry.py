from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional


class TelemetryStore:
    def __init__(self, path: str = "routemcp.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = RLock()
        with self.connection:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA busy_timeout=5000")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    selected_model TEXT, status TEXT NOT NULL, profile TEXT NOT NULL,
                    request_json TEXT NOT NULL, decision_json TEXT NOT NULL,
                    attempts_json TEXT NOT NULL, total_cost_usd REAL NOT NULL DEFAULT 0,
                    total_latency_ms REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
            """)

    def record(
        self, trace_id: str, selected_model: Optional[str], status: str, profile: str,
        request: Dict[str, Any], decision: Dict[str, Any], attempts: List[Dict[str, Any]],
    ) -> None:
        cost = sum(float(a.get("actual_cost_usd", 0)) for a in attempts)
        latency = sum(float(a.get("latency_ms", 0)) for a in attempts)
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO traces
                (id, selected_model, status, profile, request_json, decision_json, attempts_json,
                 total_cost_usd, total_latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trace_id, selected_model, status, profile, json.dumps(request), json.dumps(decision),
                 json.dumps(attempts), cost, latency),
            )

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.connection.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in ("request_json", "decision_json", "attempts_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def metrics(self) -> Dict[str, Any]:
        with self.lock:
            summary = self.connection.execute("""
                SELECT COUNT(*) requests,
                       COALESCE(SUM(total_cost_usd), 0) total_cost_usd,
                       COALESCE(AVG(total_latency_ms), 0) average_latency_ms,
                       COALESCE(AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0 END), 0) success_rate
                FROM traces
            """).fetchone()
            distribution = self.connection.execute("""
                SELECT selected_model, COUNT(*) count FROM traces
                WHERE selected_model IS NOT NULL GROUP BY selected_model ORDER BY count DESC
            """).fetchall()
            escalations = self.connection.execute("""
                SELECT COALESCE(AVG(CASE WHEN json_array_length(attempts_json) > 1 THEN 1.0 ELSE 0 END), 0)
                FROM traces
            """).fetchone()[0]
        result = dict(summary)
        result["success_rate"] = round(result["success_rate"], 4)
        result["escalation_rate"] = round(escalations, 4)
        result["model_distribution"] = {row["selected_model"]: row["count"] for row in distribution}
        result["upper_bound_cost_avoidance_usd"] = round(
            self._upper_bound_cost_avoidance(), 8
        )
        return result

    def _upper_bound_cost_avoidance(self) -> float:
        """Diagnostic upper bound, not a quality-adjusted savings claim."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT decision_json, total_cost_usd FROM traces WHERE status = 'success'"
            ).fetchall()
        savings = 0.0
        for row in rows:
            decision = json.loads(row["decision_json"])
            eligible_costs = [
                float(item.get("estimated_cost_usd", 0))
                for item in decision.get("candidates", [])
                if item.get("eligible")
            ]
            if eligible_costs:
                savings += max(0.0, max(eligible_costs) - float(row["total_cost_usd"]))
        return savings

    def close(self) -> None:
        self.connection.close()
