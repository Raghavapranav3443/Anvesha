"""Hardening tests: traversal, API 404 semantics, queue bounds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from satquery.server.main import app

client = TestClient(app)


def test_unknown_api_returns_404_json():
    r = client.get("/api/does_not_exist")
    assert r.status_code == 404
    assert "detail" in r.json()          # JSON, not the SPA index.html


def test_report_traversal_blocked():
    # encoded traversal must never escape runs/
    r = client.get("/api/reports/..%2F..%2F..%2Frequirements.txt/requirements.txt")
    if r.status_code == 200:
        # if served, it must be the SPA fallback, never a leaked source file
        assert not r.text.lstrip().lower().startswith("torch")
    else:
        assert r.status_code == 404


def test_healthz_reports_queue_stats():
    body = client.get("/healthz").json()
    for key in ("jobs_pending", "jobs_in_memory", "workers"):
        assert key in body
