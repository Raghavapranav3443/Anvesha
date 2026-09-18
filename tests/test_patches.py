"""Patch-layer tests (D0): the seam must be a verified no-op today and a
working additive mechanism tomorrow. Kill-switch must restore byte-identical
baseline behaviour. No models are loaded in this file."""
from __future__ import annotations

import os
from types import SimpleNamespace

from anvesha import patches
from anvesha.registry import ToolSpec, build_default_registry


def _fake_registry() -> dict:
    return {
        "dummy_a": ToolSpec(name="dummy_a", description="d", requires="single",
                            needs_query=False, fn=lambda ctx: {"x": 1}),
        "dummy_b": ToolSpec(name="dummy_b", description="d", requires="single",
                            needs_query=False, fn=lambda ctx: {"y": 2}),
    }


def test_no_enrichers_is_identity():
    reg = _fake_registry()
    patches.apply_patches(reg)
    assert reg["dummy_a"].fn({}) == {"x": 1}
    assert reg["dummy_b"].fn({}) == {"y": 2}


def test_enricher_adds_keys_only():
    reg = _fake_registry()

    def _enrich(ctx, out):
        out.setdefault("patched", True)     # additive: never replaces

    patches._ENRICHERS["dummy_a"] = _enrich
    try:
        patches.apply_patches(reg)
        out = reg["dummy_a"].fn({"q": 1})
        assert out == {"x": 1, "patched": True}
        assert reg["dummy_b"].fn({}) == {"y": 2}   # untouched task
    finally:
        patches._ENRICHERS.pop("dummy_a", None)


def test_apply_patches_is_idempotent():
    reg = _fake_registry()
    patches._ENRICHERS["dummy_a"] = lambda ctx, out: out.setdefault("p", 1)
    try:
        patches.apply_patches(reg)
        first = reg["dummy_a"].fn
        patches.apply_patches(reg)
        assert reg["dummy_a"].fn is first          # never double-wrapped
        assert first._anvesha_patched is True
    finally:
        patches._ENRICHERS.pop("dummy_a", None)


def test_kill_switch_disables_layer(monkeypatch):
    monkeypatch.setenv("ANVESHA_PATCHES", "0")
    assert patches.patches_enabled() is False
    reg = _fake_registry()
    patches._ENRICHERS["dummy_a"] = lambda ctx, out: out.setdefault("p", 1)
    try:
        patches.apply_patches(reg)
        assert reg["dummy_a"].fn({}) == {"x": 1}   # baseline, byte-identical
    finally:
        patches._ENRICHERS.pop("dummy_a", None)


def test_enricher_failure_never_fails_tool():
    reg = _fake_registry()

    def _boom(ctx, out):
        raise RuntimeError("enricher bug")

    patches._ENRICHERS["dummy_a"] = _boom
    try:
        patches.apply_patches(reg)
        out = reg["dummy_a"].fn({})
        assert out["x"] == 1                        # tool result survives
        assert "_patch_error" in out                # defect is observable
    finally:
        patches._ENRICHERS.pop("dummy_a", None)


def test_enrich_result_additive_keys_present():
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.5,
                                  "labels": []},
                          run_id="r1", selected_task="single_vqa",
                          trace=[], configuration={})
    patches.enrich_result(res, [])
    assert res.outputs["answer"] == "a"                 # existing preserved
    assert "dossier" in res.outputs                    # R1 additive keys
    assert "freshness" in res.outputs


def test_enrich_result_kill_switch(monkeypatch):
    monkeypatch.setenv("ANVESHA_PATCHES", "0")
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.5},
                          run_id="r1")
    patches.enrich_result(res, [])
    assert "dossier" not in res.outputs                 # disabled


def test_real_registry_signatures_unchanged():
    expected = {"single_vqa", "captioning", "grounding", "change_analysis",
                "change_vqa", "optical_sar", "impact_analysis"}
    reg = build_default_registry()
    assert set(reg) == expected
    patches.apply_patches(reg)
    for name, spec in reg.items():
        sig = spec.signature()
        assert sig["name"] == name
        assert sig["requires"] in ("single", "bitemporal", "crossmodal")
        assert isinstance(sig["description"], str) and sig["description"]


def test_frozen_splits_contract_exists():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    doc = json.loads((root / "scripts" / "frozen_splits.json").read_text(encoding="utf-8"))
    assert doc["vrsbench_val"]["n"] == 200
    assert doc["levir_fulltest"]["n"] == 1500
    assert doc["ben_s1s2_val"]["seed"] == 42


def test_enrich_result_emits_honesty_key():
    """G1: the R1 frozen ``honesty`` dict must be emitted on every enriched run."""
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.5,
                                   "labels": [], "confidence_meta": {
                                       "value": 0.5, "method": "formula",
                                       "n_cal": None, "component": "grounding"}},
                          run_id="r1", selected_task="grounding",
                          trace=[], configuration={})
    patches.enrich_result(res, [])
    h = res.outputs.get("honesty")
    assert isinstance(h, dict)
    # Frozen R1 shape
    assert set(h) >= {"fallback_active", "below_gate", "pixel_space",
                      "limitation_refs"}
    assert isinstance(h["below_gate"], list)
    assert isinstance(h["limitation_refs"], list)
    # 0.5 >= structured-equation gate -> not below gate
    assert h["below_gate"] == []


def test_honesty_below_gate_flags_low_formula_confidence():
    """A formula-blended confidence below the 0.45 gate is honestly flagged."""
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.3,
                                   "confidence_meta": {
                                       "value": 0.3, "method": "formula",
                                       "n_cal": None, "component": "optical_sar"}},
                          run_id="r1", selected_task="optical_sar",
                          trace=[], configuration={})
    patches.enrich_result(res, [])
    h = res.outputs.get("honesty") or {}
    assert len(h["below_gate"]) == 1
    assert h["below_gate"][0]["component"] == "optical_sar"
    assert h["below_gate"][0]["gate"] == 0.45


def test_honesty_pixel_space_flag_with_no_crs():
    """No-CRS inputs must be reported as pixel-space (coordinates not guessed)."""
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.8},
                          run_id="r1", selected_task="single_vqa",
                          trace=[], configuration={})
    img = SimpleNamespace(crs=None)          # no georeferencing
    patches.enrich_result(res, [img])
    h = res.outputs.get("honesty") or {}
    assert h["pixel_space"] is True
    # and geo attachments still absent for pixel-space results
    assert "geo" not in res.outputs or res.outputs["geo"].get("kind") == "pixel-space"


def test_honesty_pixel_space_false_with_crs():
    img = SimpleNamespace(crs="EPSG:32633")
    res = SimpleNamespace(outputs={"answer": "a", "confidence": 0.8},
                          run_id="r1", selected_task="single_vqa",
                          trace=[], configuration={})
    patches.enrich_result(res, [img])
    assert res.outputs["honesty"]["pixel_space"] is False
