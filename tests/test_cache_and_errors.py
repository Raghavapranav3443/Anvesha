"""Tests for cache TTL / integrity and structured HTTP error responses.

These tests exercise the new behaviour added to store.py and main.py:
  - Cache TTL expiry (get_cached returns None for old rows)
  - Corrupt result_json auto-deletion
  - purge_stale() removes only terminal, expired rows
  - Structured error responses (detail / code / hint fields)
  - Queue-full 429 structured response
  - DELETE /api/cache/{key} endpoint
"""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Store-level tests  (no FastAPI, direct SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    from anvesha.store import Store
    return Store(tmp_path / "test.db")


def _insert_row(db_path: str, job_id: str, cache_key: str,
                created_at: str, status: str = "done",
                result_json: str | None = None) -> None:
    """Helper: insert a run row directly into the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO runs "
        "(job_id, created_at, query, task_override, status, cache_key, result_json, cached) "
        "VALUES (?,?,?,?,?,?,?,0)",
        (job_id, created_at, "test query", "auto", status, cache_key, result_json),
    )
    conn.commit()
    conn.close()


class TestCacheTTL:
    def test_fresh_entry_is_returned(self, tmp_store):
        """A row created moments ago should be returned by get_cached."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        result = {"answer": "yes", "confidence": 0.9}
        _insert_row(tmp_store.db_path, "job-fresh", "key-fresh", now,
                    result_json=json.dumps(result))
        hit = tmp_store.get_cached("key-fresh")
        assert hit is not None
        assert hit["result"]["answer"] == "yes"

    def test_expired_entry_is_not_returned(self, tmp_store):
        """A row older than CACHE_TTL_DAYS should be treated as a miss."""
        import anvesha.store as store_mod
        # Temporarily shrink TTL to 1 day so we can easily go past it
        original_ttl = store_mod.CACHE_TTL_DAYS
        store_mod.CACHE_TTL_DAYS = 1
        try:
            old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).strftime(
                "%Y-%m-%dT%H:%M:%S")
            _insert_row(tmp_store.db_path, "job-old", "key-old", old_ts,
                        result_json=json.dumps({"answer": "stale"}))
            hit = tmp_store.get_cached("key-old")
            assert hit is None, "Expired entry must not be served from cache"
        finally:
            store_mod.CACHE_TTL_DAYS = original_ttl

    def test_corrupt_json_is_deleted_and_returns_none(self, tmp_store):
        """A row with corrupt result_json must be auto-deleted and return None."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        _insert_row(tmp_store.db_path, "job-corrupt", "key-corrupt", now,
                    result_json="NOT VALID JSON {{{{")
        hit = tmp_store.get_cached("key-corrupt")
        assert hit is None, "Corrupt cache entry must return None"
        # Verify the row was actually removed
        row = tmp_store.get("job-corrupt")
        assert row is None, "Corrupt row must be deleted from the DB"

    def test_purge_stale_removes_old_terminal_rows(self, tmp_store):
        """purge_stale() must delete expired done/error rows but keep active ones."""
        import anvesha.store as store_mod
        original_ttl = store_mod.CACHE_TTL_DAYS
        store_mod.CACHE_TTL_DAYS = 1
        try:
            old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).strftime(
                "%Y-%m-%dT%H:%M:%S")
            now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

            _insert_row(tmp_store.db_path, "j-stale-done",  "k1", old_ts, "done")
            _insert_row(tmp_store.db_path, "j-stale-error", "k2", old_ts, "error")
            _insert_row(tmp_store.db_path, "j-active",      "k3", old_ts, "running")  # must survive
            _insert_row(tmp_store.db_path, "j-fresh-done",  "k4", now_ts, "done")     # must survive

            purged = tmp_store.purge_stale()
            assert purged == 2, f"Expected 2 rows purged, got {purged}"
            assert tmp_store.get("j-active")     is not None, "Running job must not be purged"
            assert tmp_store.get("j-fresh-done") is not None, "Fresh done job must not be purged"
        finally:
            store_mod.CACHE_TTL_DAYS = original_ttl

    def test_delete_cache_entry(self, tmp_store):
        """delete_cache_entry() must remove all rows for that cache key."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        _insert_row(tmp_store.db_path, "j-del-1", "key-del", now,
                    result_json=json.dumps({"answer": "a"}))
        _insert_row(tmp_store.db_path, "j-del-2", "key-del", now,
                    result_json=json.dumps({"answer": "b"}))
        _insert_row(tmp_store.db_path, "j-keep",  "key-keep", now,
                    result_json=json.dumps({"answer": "keep"}))

        deleted = tmp_store.delete_cache_entry("key-del")
        assert deleted == 2
        assert tmp_store.get_cached("key-del")  is None
        assert tmp_store.get_cached("key-keep") is not None


# ---------------------------------------------------------------------------
# HTTP-level tests  (FastAPI TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from anvesha.server.main import app
    return TestClient(app)


class TestStructuredErrors:
    def test_bad_extension_returns_structured_error(self, client):
        """Uploading a .bmp file must return 400 with detail/code/hint."""
        r = client.post(
            "/api/jobs",
            data={"query": "describe", "sample_names": ""},
            files=[("files", ("photo.bmp", b"BM\x00\x00fake", "image/bmp"))],
        )
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
        assert "code" in body
        assert body["code"] == "unsupported_format"
        assert "hint" in body

    def test_no_images_returns_structured_error(self, client):
        """Posting with no files and no sample_names must return 400 no_images."""
        r = client.post(
            "/api/jobs",
            data={"query": "describe", "sample_names": ""},
        )
        assert r.status_code == 400
        body = r.json()
        assert body.get("code") == "no_images"

    def test_unknown_job_returns_structured_404(self, client):
        """GET /api/jobs/<bogus> must return 404 with code=not_found."""
        r = client.get("/api/jobs/doesnotexist123")
        assert r.status_code == 404
        body = r.json()
        assert body.get("code") == "not_found"

    def test_delete_cache_endpoint(self, client):
        """DELETE /api/cache/{key} must return 204 (even for missing keys)."""
        r = client.delete("/api/cache/nonexistent-key-xyz")
        assert r.status_code == 204

    def test_stats_endpoint(self, client):
        """/api/stats must return a dict with at least the expected keys."""
        r = client.get("/api/stats")
        assert r.status_code == 200
        body = r.json()
        for key in ("in_memory", "running", "pending", "total_runs", "completed"):
            assert key in body, f"Missing key '{key}' in /api/stats response"

    def test_queue_full_returns_structured_429(self, client, monkeypatch):
        """Filling the queue must return 429 with structured detail/code/hint."""
        import base64
        import anvesha.server.jobs as jobs_mod

        # Set queue limit to 0 so the very next create() call hits the cap
        monkeypatch.setattr(jobs_mod, "MAX_QUEUE", 0)

        # Minimal valid 1×1 PNG so the request passes upload validation
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        r = client.post(
            "/api/jobs",
            data={"query": "describe", "sample_names": ""},
            files=[("files", ("pixel.png", tiny_png, "image/png"))],
        )
        assert r.status_code == 429
        body = r.json()
        assert "detail" in body
        assert "code" in body
        assert body["code"] == "queue_full"
        assert "hint" in body
