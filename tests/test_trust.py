"""D2 trust: composers + confmeta + freshness + dossier (pure logic, no
model loads). Validates the additive contract: raw values never lost,
clauses traceable, freshness honest."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from satquery.answers.compose import compose_answer
from satquery.captions.compose import compose_caption
from satquery.confmeta import get, formula, stamp
from satquery.freshness import clocks_for, _THRESHOLDS
from satquery.dossier import emit, DossierDoc
from satquery.io_utils import RSImage


def _z():
    import numpy as np
    return np.zeros((8, 8, 3), np.float32)


# ---- B6 caption composer ------------------------------------------------- #

def test_caption_water_query_leads_with_water():
    out = {"caption": "A generic scene.", "labels": [["River", 0.8]],
           "layout": {"lower": "water-covered"}}
    c = compose_caption(None, "where is the water body?", out)
    assert "water" in c["caption"].lower()
    assert c["query_concept"] == "water"
    assert c["caption_learned"] == "A generic scene."


def test_caption_preserves_learned_verbatim():
    out = {"caption": "Learned sentence here.", "labels": [], "layout": {}}
    c = compose_caption(None, "describe this image", out)
    assert c["caption_learned"] == "Learned sentence here."
    assert c["caption"].startswith("Learned sentence here")


def test_caption_no_crash_on_empty():
    c = compose_caption(None, "", {"caption": "", "labels": [], "layout": {}})
    assert isinstance(c["caption"], str)


# ---- B9 answer composer -------------------------------------------------- #

def test_answer_raw_never_lost():
    out = {"answer": "yes", "confidence": 0.9, "caption": "A scene."}
    a = compose_answer("is there water?", out, [])
    assert a["answer_raw"] == "yes"
    assert a["answer"] != ""
    assert any(c["kind"] == "verdict" for c in a["answer_clauses"])


def test_answer_clauses_are_traceable():
    out = {"answer": "vegetation", "confidence": 0.7,
           "caption": "Green scene.", "evidence": ["ev1", "ev2"]}
    a = compose_answer("what covers the land?", out, [], run_id="R1")
    for cl in a["answer_clauses"]:
        assert "source_key" in cl and "kind" in cl
    assert a["trace_link"] == "/api/reports/R1"
    assert "source" in (a["provenance_pitch"] or "")


def test_answer_no_crash_empty():
    a = compose_answer("", {}, [])
    assert "answer_raw" in a and "answer_clauses" in a


# ---- B7 confmeta --------------------------------------------------------- #

def test_confmeta_formula_fallback(tmp_path, monkeypatch):
    import satquery.confmeta as cm
    monkeypatch.setattr(cm, "_CACHE", None)
    monkeypatch.setattr(cm.CONFIG, "weights_dir", tmp_path)
    m = get("grounding", 0.5)
    assert m.method == "formula" and m.n_cal is None and m.value == 0.5


def test_confmeta_stamp_additive_and_noop():
    assert "confidence_meta" in stamp({"answer": "yes", "confidence": 0.8},
                                      "vqa")
    out = {"answer": "yes", "confidence": 0.8}
    assert stamp(out, "vqa")["answer"] == "yes"          # untouched


def test_formula_equations_documented():
    for comp in ("grounding", "change_analysis", "optical_sar", "change_vqa"):
        assert formula(comp)


# ---- C3 freshness -------------------------------------------------------- #

def _res(task="single_vqa", conf=0.8, acquired="2024-05-01"):
    img = RSImage(array=_z(), format="geotiff", modality="rgb",
                  acquired=acquired)
    return SimpleNamespace(selected_task=task, outputs={"confidence": conf},
                           trace=[], run_id="R1", configuration={},
                           answer="ok"), [img]


def test_freshness_ok_when_recent():
    res, imgs = _res(acquired="2099-01-01")
    clk = clocks_for(res, imgs)
    assert clk.quality == "ok" and clk.why is None
    assert clk.threshold_days == _THRESHOLDS["single_vqa"]


def test_freshness_degraded_low_confidence():
    clk = clocks_for(*_res(conf=0.1))
    assert clk.quality == "degraded" and "confidence" in (clk.why or "")


def test_freshness_degraded_when_stale():
    clk = clocks_for(*_res(task="change_analysis", acquired="2020-01-01"))
    assert clk.quality == "degraded"
    assert clk.staleness_days and clk.staleness_days > 16


def test_freshness_never_claims_satellite_pass():
    clk = clocks_for(*_res())
    assert "pass" not in clk.method_note and "revisit" not in clk.method_note


# ---- C2 dossier ---------------------------------------------------------- #

def test_dossier_emit_builds_all_sections():
    img = RSImage(array=_z(), format="geotiff", modality="sar",
                  acquired="2024-05-01", crs="EPSG:4326",
                  transform_bounds=(77.0, 28.0, 77.1, 28.1))
    res = SimpleNamespace(run_id="R1", selected_task="grounding",
                         configuration={"c": "single"}, trace=[],
                         outputs={"answer": "Grounded 1 water region.",
                                  "confidence": 0.6, "labels": [],
                                  "source_model": "test"})
    dd = emit(res, [img]).to_dict()
    assert dd["run_id"] == "R1"
    assert len(dd["key_facts"]) >= 1
    assert isinstance(dd["timeline"], list) and isinstance(dd["sources"], list)


def test_dossier_preserves_raw_answer():
    img = RSImage(array=_z(), format="geotiff", modality="rgb")
    res = SimpleNamespace(run_id="R2", selected_task="single_vqa",
                         configuration={}, trace=[],
                         outputs={"answer": "yes", "confidence": 0.9,
                                  "labels": []})
    d = emit(res, [img]).to_dict()
    assert d["head"]["answer"] == d["head"]["answer_raw"] == "yes"


def test_dossier_geo_aoi_when_crs():
    img = RSImage(array=_z(), format="geotiff", modality="rgb",
                  crs="EPSG:4326", transform_bounds=(77.0, 28.0, 77.1, 28.1))
    res = SimpleNamespace(run_id="R3", selected_task="grounding",
                         configuration={}, answer="ok", trace=[])
    res.outputs = {"ranked": SimpleNamespace(
                        primary=SimpleNamespace(box=[2, 2, 5, 5]), alternates=[]),
                   "confidence": 0.5, "labels": []}
    assert emit(res, [img]).to_dict()["aoi"]["kind"] == "geo"


def test_dossier_pixel_space_aoi_without_crs():
    img = RSImage(array=_z(), format="geotiff", modality="rgb")
    res = SimpleNamespace(run_id="R4", selected_task="grounding",
                         configuration={}, answer="ok", trace=[])
    res.outputs = {"ranked": SimpleNamespace(
                        primary=SimpleNamespace(box=[1, 1, 3, 3]), alternates=[]),
                   "confidence": 0.5, "labels": []}
    assert emit(res, [img]).to_dict()["aoi"]["kind"] == "pixel-space"

