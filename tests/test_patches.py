"""Patch-layer tests (D0): the seam must be a verified no-op today and a
working additive mechanism tomorrow. Kill-switch must restore byte-identical
baseline behaviour. No models are loaded in this file."""
from __future__ import annotations

import os
from types import SimpleNamespace

from satquery import patches
from satquery.registry import ToolSpec, build_default_registry


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
        assert first._satquery_patched is True
    finally:
        patches._ENRICHERS.pop("dummy_a", None)


def test_kill_switch_disables_layer(monkeypatch):
    monkeypatch.setenv("SATQUERY_PATCHES", "0")
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
    monkeypatch.setenv("SATQUERY_PATCHES", "0")
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
