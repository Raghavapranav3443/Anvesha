"""SQLite persistence: run history + result cache. Stdlib only — designed
for air-gapped deployment (single file, zero infrastructure)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                job_id TEXT PRIMARY KEY,
                created_at TEXT,
                query TEXT,
                task_override TEXT,
                status TEXT,
                selected_task TEXT,
                answer TEXT,
                confidence REAL,
                run_id TEXT,
                error TEXT,
                result_json TEXT,
                cache_key TEXT,
                cached INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_runs_cache ON runs(cache_key);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_start(self, job_id: str, created_at: str, query: str,
                     task_override: str, status: str, cache_key: str) -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs (job_id, created_at, query, "
                "task_override, status, cache_key) VALUES (?,?,?,?,?,?)",
                (job_id, created_at, query, task_override or "auto", status,
                 cache_key))

    def finish(self, job_id: str, status: str, selected_task: str,
               answer: str, confidence: float, run_id: str, error: str,
               result: Optional[Dict], cached: bool) -> None:
        with _LOCK, self._conn() as c:
            c.execute(
                "UPDATE runs SET status=?, selected_task=?, answer=?, "
                "confidence=?, run_id=?, error=?, result_json=?, cached=? "
                "WHERE job_id=?",
                (status, selected_task, answer, confidence, run_id, error,
                 json.dumps(result, default=str) if result else None,
                 1 if cached else 0, job_id))

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT job_id, created_at, query, status, selected_task, "
                "answer, confidence, run_id, cached FROM runs "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE job_id=?",
                          (job_id,)).fetchone()
        return dict(r) if r else None

    def get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        with self._conn() as c:
            r = c.execute(
                "SELECT job_id, result_json FROM runs WHERE cache_key=? AND "
                "status='done' ORDER BY created_at DESC LIMIT 1",
                (cache_key,)).fetchone()
        if r and r["result_json"]:
            try:
                return {"job_id": r["job_id"],
                        "result": json.loads(r["result_json"])}
            except Exception:
                return None
        return None

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"]
            done = c.execute(
                "SELECT COUNT(*) n FROM runs WHERE status='done'").fetchone()["n"]
        return {"total_runs": total, "completed": done}
