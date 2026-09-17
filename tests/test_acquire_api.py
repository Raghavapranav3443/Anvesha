"""HTTP-level tests for the online acquisition endpoints.

Offline throughout, like the rest of the suite: these assert the *contract* --
which failures map to which structured status codes, and that an ``acquire_id``
cannot be used to read arbitrary files. Real network behaviour is covered by
``scripts/acquire_smoke.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from satquery.config import CONFIG
from satquery.server.main import app

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

def test_acquire_status_reports_the_isro_service_as_credential_free():
    resp = client.get("/api/acquire/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["providers"], "the UI needs the provider list to explain itself"
    assert body["isro"]["credential_free"] is True
    assert "cache" in body and "defaults" in body


def test_acquire_status_lists_bhoonidhi_as_not_yet_configured():
    body = client.get("/api/acquire/status").json()
    by_name = {p["name"]: p for p in body["providers"]}
    assert "stac_earth_search" in by_name
    assert by_name["stac_earth_search"]["requires_signing"] is False
    assert by_name.get("stac_bhoonidhi", {}).get("requires_signing") is True


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #

def test_plan_requires_a_place():
    resp = client.post("/api/acquire/plan", json={"query": "   "})
    assert resp.status_code == 400
    assert resp.json()["code"] == "no_place"


def test_plan_refuses_in_airgap_mode_with_409(monkeypatch):
    """The air-gap promise has to hold at the API boundary, not just in prose.

    Mode is forced here rather than inherited from ``data/settings.json``: a
    test that silently depends on the machine's last-used mode would make a live
    network call on a developer's box and quietly pass.
    """
    from satquery.acquire import http as http_mod

    monkeypatch.setattr(http_mod, "current_mode", lambda: "airgap")
    resp = client.post("/api/acquire/plan", json={"query": "Assam"})
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "airgap_mode"
    assert "online mode" in body.get("hint", "").lower()


def test_blocked_network_is_never_reported_as_no_imagery(monkeypatch):
    """A policy refusal must not masquerade as an empty catalogue."""
    from satquery.acquire.errors import NetworkBlockedAirgap
    from satquery.acquire.providers.stac import (StacProvider,
                                                 search_with_fallback)

    def refuse(*_a, **_k):
        raise NetworkBlockedAirgap("air-gap mode: blocked", event="x",
                                   target="earth-search")

    provider = StacProvider(name="p", endpoint="https://example.invalid/v1")
    monkeypatch.setattr(provider, "search", refuse)
    with pytest.raises(NetworkBlockedAirgap):
        search_with_fallback([provider], (0, 0, 1, 1), start="2026-01-01",
                             end="2026-02-01", transport=object())


def test_plan_rejects_a_bad_numeric_parameter():
    resp = client.post("/api/acquire/plan",
                       json={"query": "Assam", "days": "not-a-number"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_parameter"


def test_plan_maps_unknown_place_to_422(monkeypatch):
    from satquery.acquire import service as service_mod
    from satquery.acquire.aoi import PlaceNotFound

    def boom(*_a, **_k):
        raise PlaceNotFound("could not find a place matching 'zzz'")

    monkeypatch.setattr(service_mod, "analysis_window", boom)
    resp = client.post("/api/acquire/plan", json={"query": "zzz"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "place_not_found"


def test_plan_maps_empty_catalogue_to_422(monkeypatch):
    from satquery.acquire import service as service_mod
    from satquery.acquire.errors import NoSceneFound

    def boom(*_a, **_k):
        raise NoSceneFound("no usable image pair")

    monkeypatch.setattr(service_mod, "plan", boom)
    resp = client.post("/api/acquire/plan", json={"query": "Assam"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "no_scene_found"


# --------------------------------------------------------------------------- #
# Fetch + provenance
# --------------------------------------------------------------------------- #

def test_provenance_lookup_for_unknown_acquisition_is_404():
    assert client.get("/api/acquire/does-not-exist").status_code == 404


def _write_fake_acquisition(acquire_id: str, dates=("2026-04-17", "2026-08-10")):
    folder = CONFIG.runs_dir / acquire_id / "acquired"
    folder.mkdir(parents=True, exist_ok=True)
    import rasterio
    from rasterio.transform import from_origin

    for date in dates:
        path = folder / f"s2_{date}.tif"
        data = np.full((4, 16, 16), 0.15, dtype="float32")
        with rasterio.open(
            path, "w", driver="GTiff", height=16, width=16, count=4,
            dtype="float32", crs="EPSG:32646",
            transform=from_origin(680000, 3040000, 10, 10),
        ) as dst:
            dst.write(data)
            dst.update_tags(acquisition_date=date, provider="test")
    # A non-imagery file that must never be handed to the pipeline.
    (folder / "provenance.provenance.json").write_text("{}", encoding="utf-8")
    return folder


def test_acquired_images_returns_only_geotiffs_in_date_order():
    from satquery.acquire.service import acquired_images

    folder = _write_fake_acquisition("acq-order-test")
    names = [p.name for p in acquired_images("acq-order-test")]
    assert names == ["s2_2026-04-17.tif", "s2_2026-08-10.tif"]
    assert all(p.suffix == ".tif" for p in acquired_images("acq-order-test"))
    assert not any("provenance" in n for n in names)


def test_acquire_id_cannot_escape_the_runs_directory():
    """A traversal attempt must yield nothing, not a readable path."""
    from satquery.acquire.service import acquired_images

    for hostile in ("../../etc", "..\\..\\windows\\system32", "a/../../b",
                    "/absolute/path", "....//....//etc"):
        assert acquired_images(hostile) == [], f"{hostile!r} escaped the runs dir"


def test_dates_are_recovered_from_the_files_themselves():
    from satquery.acquire.service import acquired_images, dates_for_images

    _write_fake_acquisition("acq-dates-test")
    files = acquired_images("acq-dates-test")
    before, after = dates_for_images(files)
    assert before == "2026-04-17"
    assert after == "2026-08-10"


def test_job_creation_rejects_an_unknown_acquire_id():
    resp = client.post("/api/jobs", data={"query": "what changed",
                                          "acquire_id": "no-such-acquisition"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "acquire_not_found"


def test_job_creation_without_any_input_explains_the_online_option():
    resp = client.post("/api/jobs", data={"query": "what changed"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "no_images"
    assert "fetch" in body.get("hint", "").lower()
