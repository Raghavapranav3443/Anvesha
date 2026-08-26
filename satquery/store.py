"""SQLite persistence: run history + result cache. Stdlib only — designed
for air-gapped deployment (single file, zero infrastructure).

Cache policy
------------
* Result rows are valid for ``SATQUERY_CACHE_TTL`` days (default 7).
* ``get_cached()`` silently deletes corrupt ``result_json`` rows and returns
  ``None`` so the job re-runs cleanly.
* ``purge_stale()`` hard-deletes rows beyond the TTL; called at startup and
  periodically by the server.
* ``delete_cache_entry(cache_key)`` busts one specific cache key (used by the
  ``DELETE /api/cache/{key}`` endpoint).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache TTL — default 7 days, override via env
# ---------------------------------------------------------------------------
CACHE_TTL_DAYS: int = max(1, int(os.environ.get("SATQUERY_CACHE_TTL", "7")))


class Store:
    """Thread-safe SQLite store.  Each instance owns its own lock so that
    multiple Store objects (e.g. in tests) do not serialise on a global."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()          # instance-level; no cross-instance contention
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
            CREATE INDEX IF NOT EXISTS idx_runs_cache   ON runs(cache_key);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
            """)

    # ------------------------------------------------------------------ #
    # Connection helper
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # Write helpers
    # ------------------------------------------------------------------ #

    def upsert_start(self, job_id: str, created_at: str, query: str,
                     task_override: str, status: str, cache_key: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs (job_id, created_at, query, "
                "task_override, status, cache_key) VALUES (?,?,?,?,?,?)",
                (job_id, created_at, query, task_override or "auto", status,
                 cache_key))

    def finish(self, job_id: str, status: str, selected_task: str,
               answer: str, confidence: float, run_id: str, error: str,
               result: Optional[Dict], cached: bool) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE runs SET status=?, selected_task=?, answer=?, "
                "confidence=?, run_id=?, error=?, result_json=?, cached=? "
                "WHERE job_id=?",
                (status, selected_task, answer, confidence, run_id, error,
                 json.dumps(result, default=str) if result else None,
                 1 if cached else 0, job_id))

    # ------------------------------------------------------------------ #
    # Read helpers
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Return the most-recent successful result for *cache_key* if it
        is still within the TTL window, otherwise ``None``.

        Corrupt ``result_json`` rows are automatically deleted so they
        don't permanently poison the cache.
        """
        if not cache_key:
            return None

        cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)) \
            .strftime("%Y-%m-%dT%H:%M:%S")

        with self._conn() as c:
            r = c.execute(
                "SELECT job_id, result_json FROM runs "
                "WHERE cache_key=? AND status='done' AND created_at >= ? "
                "ORDER BY created_at DESC LIMIT 1",
                (cache_key, cutoff),
            ).fetchone()

        if r is None:
            return None

        if not r["result_json"]:
            # Row exists but payload is empty — treat as miss
            return None

        try:
            return {"job_id": r["job_id"],
                    "result": json.loads(r["result_json"])}
        except (json.JSONDecodeError, ValueError):
            # Corrupt JSON — delete the offending row and log a warning
            logger.warning("Deleting corrupt cache row for job_id=%s cache_key=%.16s…",
                           r["job_id"], cache_key)
            self._delete_job_row(r["job_id"])
            return None

    def delete_cache_entry(self, cache_key: str) -> int:
        """Delete **all** rows matching *cache_key* (explicit cache bust).

        Returns the number of rows deleted.
        """
        if not cache_key:
            return 0
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM runs WHERE cache_key=?", (cache_key,))
            deleted = cur.rowcount
        logger.info("Cache invalidated: cache_key=%.16s… rows_deleted=%d",
                    cache_key, deleted)
        return deleted

    def purge_stale(self) -> int:
        """Delete rows older than the TTL window that are in terminal states.

        Running/queued jobs are never deleted regardless of age.
        Returns the number of rows purged.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)) \
            .strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM runs WHERE created_at < ? "
                "AND status NOT IN ('queued', 'running')",
                (cutoff,))
            purged = cur.rowcount
        if purged:
            logger.info("Cache purge: removed %d stale rows (TTL=%d days)",
                        purged, CACHE_TTL_DAYS)
        return purged

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"]
            done = c.execute(
                "SELECT COUNT(*) n FROM runs WHERE status='done'").fetchone()["n"]
            cached = c.execute(
                "SELECT COUNT(*) n FROM runs WHERE cached=1").fetchone()["n"]
        return {
            "total_runs": total,
            "completed": done,
            "cache_hits_persisted": cached,
            "cache_ttl_days": CACHE_TTL_DAYS,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _delete_job_row(self, job_id: str) -> None:
        """Remove a single row; best-effort, swallows exceptions."""
        try:
            with self._lock, self._conn() as c:
                c.execute("DELETE FROM runs WHERE job_id=?", (job_id,))
        except Exception:
            pass
