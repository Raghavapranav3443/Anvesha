"""API-level tests for the FastAPI service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from satquery.server.main import app

client = TestClient(app)


def test_samples_endpoint():
    r = client.get("/api/samples")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert any("isro" in n for n in names)
    modalities = {s["modality"] for s in r.json()}
    assert "sar" in modalities


def test_provenance_endpoint():
    r = client.get("/api/provenance")
    assert r.status_code == 200
    body = r.json()
    assert len(body["models"]) == 4


def test_job_lifecycle():
    samples_dir = Path(__file__).resolve().parents[1] / "samples"
    with open(samples_dir / "demo_isroformat_optical.tif", "rb") as f1, \
            open(samples_dir / "demo_isroformat_sar.tif", "rb") as f2:
        r = client.post("/api/jobs",
                        data={"query": "Use the optical and SAR images together "
                                       "to identify built-up regions.",
                              "sample_names": ""},
                        files=[("files", ("a.tif", f1)),
                               ("files", ("b.tif", f2))])
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    import time
    for _ in range(120):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st.get("error")
    res = st["result"]
    assert res["selected_task"] == "optical_sar"
    assert st["trace"][0]["name"] == "validate_inputs"


def test_stats_freshness_legend():
    """C3/R5: /api/stats carries the additive freshness legend."""
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    legend = body.get("freshness_legend")
    assert legend is not None
    assert "thresholds_by_task_days" in legend
    assert "single_vqa" in legend["thresholds_by_task_days"]
    assert set(legend["flags"]) == {"ok", "degraded"}
    assert "orbital" in legend["method_note"] or "age" in legend["method_note"]
