"""Concurrency correctness + health + cache tests."""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from anvesha.store import Store  # noqa: F401 (exercised via API)
from anvesha.server.main import app

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["workers"] >= 1


def test_upload_limit_rejected():
    r = client.post("/api/jobs", data={"query": "x"},
                    files=[("files", ("a.txt", b"junk"))])
    assert r.status_code == 400


def test_concurrent_jobs_traces_isolated():
    """Two parallel jobs must each see their own 5-step trace (race regression)."""
    results = {}
    errors = []

    def run_job(tag, f1, f2, query):
        try:
            with open(SAMPLES / f1, "rb") as a, open(SAMPLES / f2, "rb") as b:
                r = client.post("/api/jobs",
                                data={"query": query, "sample_names": ""},
                                files=[("files", ("a.tif", a)),
                                       ("files", ("b.tif", b))])
            jid = r.json()["job_id"]
            for _ in range(400):
                st = client.get(f"/api/jobs/{jid}").json()
                if st["status"] in ("done", "error"):
                    break
                time.sleep(0.4)
            names = [s["name"] for s in st["trace"]]
            results[tag] = (st["status"], names)
        except Exception as e:
            errors.append(str(e))

    t1 = threading.Thread(target=run_job, args=(
        "change", "demo_change_2020.tif", "demo_change_2024.tif",
        "What changed between these two dates?"))
    t2 = threading.Thread(target=run_job, args=(
        "optsar", "demo_isroformat_optical.tif", "demo_isroformat_sar.tif",
        "Use the optical and SAR images together."))
    t1.start(); t2.start(); t1.join(180); t2.join(180)

    assert not errors, errors
    assert results["change"][0] == "done"
    assert results["optsar"][0] == "done"
    # each trace must contain its own tool, not a mix
    assert any("change" in n for n in results["change"][1])
    assert any("optical_sar" in n for n in results["optsar"][1])


def test_result_cache_hit():
    q = "Is there water in this image?"
    with open(SAMPLES / "demo_single_optical.png", "rb") as f:
        r1 = client.post("/api/jobs", data={"query": q, "sample_names": ""},
                         files=[("files", ("s.png", f))])
    jid = r1.json()["job_id"]
    for _ in range(240):
        st1 = client.get(f"/api/jobs/{jid}").json()
        if st1["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert st1["status"] == "done"

    with open(SAMPLES / "demo_single_optical.png", "rb") as f:
        r2 = client.post("/api/jobs", data={"query": q, "sample_names": ""},
                         files=[("files", ("s.png", f))])
    assert r2.json().get("cached") is True
    st2 = client.get(f"/api/jobs/{r2.json()['job_id']}").json()
    assert st2["status"] == "done"
    assert st2["result"]["answer"] == st1["result"]["answer"]


def test_history_endpoint():
    r = client.get("/api/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list) and len(rows) >= 1
