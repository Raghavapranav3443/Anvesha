"""Confidence honesty: trust must reflect method reliability, not self-belief.

An audit of the confidence layer found three ways it flattered itself:

1. ``weights/calibration.json`` labels vqa/counting/cdvqa ``method: "temp"``
   ("temperature-scaled") while every ``param`` is ``1.0`` and ``n_cal`` is
   null -- i.e. the identity transform. The numbers are raw softmax wearing a
   calibration label.
2. The honesty below-gate check fired only when ``method == "formula"``, so
   every ``"temp"`` component could report an arbitrarily low confidence
   *without ever being flagged* -- including single_vqa, the most-used
   specialist.
3. ``impact_analysis`` had no enricher, so it reported no ``confidence_meta``
   at all: no trust, and invisible to the gate.

These tests pin the fixes. They also pin the *honest* direction of travel: a
confident answer from a weak component must read as weak evidence.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from anvesha import patches
from anvesha.confmeta import (RELIABILITY, effective_trust, reliability_of,
                               trust_band)


def _enriched(component: str, confidence: float, method: str = "formula",
              task: str = None):
    res = SimpleNamespace(
        outputs={"answer": "a", "confidence": confidence,
                 "confidence_meta": {"value": confidence, "method": method,
                                     "n_cal": None, "component": component}},
        run_id="r", selected_task=task or component, trace=[], configuration={})
    patches.enrich_result(res, [])
    return res.outputs


# --------------------------------------------------------------------------- #
# 1. Relaying a temperature of 1.0 is not calibration
# --------------------------------------------------------------------------- #

def test_no_component_claims_calibration_without_evidence():
    """A 'temp' label must be earned: non-identity temperature AND a sample count.

    The sidecar this project shipped broke exactly this rule -- it labelled three
    heads "temp" while every ``param`` was 1.0 (the identity transform) and
    ``n_cal`` was null, so the runtime was uncalibrated while the documents
    claimed T=1.55. This test makes that class of drift fail loudly.
    """
    from anvesha.confmeta import claim_problems, load_calibration
    assert claim_problems() == {}, f"unearned calibration labels: {claim_problems()}"
    for comp, v in load_calibration().items():
        if v.get("method") != "temp":
            continue
        assert v.get("param") not in (None, 1.0), f"{comp} claims temp at identity"
        assert v.get("n_cal"), f"{comp} claims temp with no fitted sample count"


def test_a_hand_edited_sidecar_cannot_re_assert_calibration(tmp_path, monkeypatch):
    """Simulate the original defect and prove the audit catches it."""
    import json
    import anvesha.confmeta as cm
    (tmp_path / "calibration.json").write_text(json.dumps({
        "vqa": {"method": "temp", "param": 1.0, "n_cal": None},
        "cdvqa": {"method": "temp", "param": 1.7, "n_cal": 0},
    }), encoding="utf-8")
    monkeypatch.setattr(cm, "_CACHE", None)
    monkeypatch.setattr(cm.CONFIG, "weights_dir", tmp_path)
    problems = cm.claim_problems()
    assert "vqa" in problems and "cdvqa" in problems
    # And the runtime must not present either as calibrated.
    assert cm.get("vqa", 0.9).calibrated is False
    assert cm.get("cdvqa", 0.9).calibrated is False


# --------------------------------------------------------------------------- #
# 2. The honesty gate is method-agnostic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["temp", "formula", "platt"])
def test_low_confidence_flagged_regardless_of_method(method):
    out = _enriched("vqa", 0.20, method=method)
    assert len(out["honesty"]["below_gate"]) == 1, \
        f"{method}-labelled low confidence must still be flagged"
    assert out["honesty"]["below_gate"][0]["method"] == method


def test_high_confidence_not_flagged():
    out = _enriched("vqa", 0.80, method="temp")
    assert out["honesty"]["below_gate"] == []


def test_below_gate_records_the_gate_actually_used():
    out = _enriched("cdvqa", 0.10, method="temp")
    entry = out["honesty"]["below_gate"][0]
    assert entry["gate"] == 0.45
    assert entry["component"] == "cdvqa"


# --------------------------------------------------------------------------- #
# 3. Trust = reported confidence x measured reliability
# --------------------------------------------------------------------------- #

def test_every_stamped_component_carries_both_factors():
    """Stamping happens per-tool via confmeta.stamp (not in enrich_result),
    so exercise the unit that actually attaches the factors."""
    from anvesha.confmeta import stamp
    for component in ("vqa", "change_analysis", "grounding"):
        cm = stamp({"answer": "a", "confidence": 0.90}, component)["confidence_meta"]
        assert cm["reliability"] is not None, component
        assert cm["trust"] is not None, component
        assert cm["reliability_evidence"], component
        assert cm["trust_band"] and cm["trust_advice"], component
        # additive: the original keys survive
        assert cm["value"] == 0.90 and cm["component"] == component


def test_trust_prefers_measurement_over_model_opinion():
    """The model's opinion is always reported, but where a reliability table has
    been measured, trust is that measurement rather than the product."""
    t = effective_trust("vqa", 0.90)
    assert t["model_opinion_trust"] == pytest.approx(0.90 * 0.700, abs=1e-3)
    assert t["trust_source"] in ("measured_band", "confidence_x_reliability")
    if t["trust_source"] == "measured_band":
        ev = t["band_evidence"]
        assert ev["observed_accuracy"] == t["trust"]
        assert ev["n"] >= 30
        assert ev["source"]


def test_measured_trust_is_monotone_in_confidence():
    """A low claim must not come back with more trust than a high one."""
    low = effective_trust("vqa", 0.30)["trust"]
    high = effective_trust("vqa", 0.95)["trust"]
    assert low < high
    assert low < 0.25, "a 30% claim must not read as trustworthy"


def test_confident_weak_component_reads_as_weak_evidence():
    """The whole point: grounding is known-weak (IoU@0.5 ~0.126) and must not
    be able to present a confident-sounding answer as trustworthy."""
    t = effective_trust("grounding", 0.95)
    assert t["trust"] < 0.20
    assert t["band"] in ("very_low", "low")
    assert t["limiting_factor"] == "reliability"


def test_strong_component_can_reach_high_trust():
    assert effective_trust("change_analysis", 0.95)["band"] == "high"


def test_reliability_numbers_are_evidenced():
    """No invented reliabilities: every entry must cite a measurement."""
    for component, entry in RELIABILITY.items():
        assert isinstance(entry["reliability"], float), component
        assert 0.0 <= entry["reliability"] <= 1.0, component
        assert entry.get("evidence"), f"{component} has no cited evidence"


def test_unknown_reliability_is_not_presented_as_high_trust():
    t = effective_trust("method_we_never_measured", 0.99)
    assert t["reliability"] is None
    assert t["band"] == "unverified"
    assert t["limiting_factor"] == "unknown reliability"


def test_reliability_of_unknown_component_says_so():
    r = reliability_of("nope")
    assert r["reliability"] is None
    assert "no measured reliability" in r["evidence"]


@pytest.mark.parametrize("trust,expected", [(0.9, "high"), (0.5, "moderate"),
                                            (0.3, "low"), (0.1, "very_low")])
def test_trust_bands(trust, expected):
    assert trust_band(trust)["band"] == expected


def test_bands_carry_plain_english_advice():
    """The decision layer surfaces these verbatim to non-experts."""
    for trust in (0.9, 0.5, 0.3, 0.1):
        advice = trust_band(trust)["advice"]
        assert isinstance(advice, str) and len(advice) > 10
        for jargon in ("IoU", "F1", "BLEU", "calibrat", "softmax", "threshold"):
            assert jargon.lower() not in advice.lower()


# --------------------------------------------------------------------------- #
# 4. impact_analysis is no longer invisible
# --------------------------------------------------------------------------- #

def test_impact_task_is_stamped():
    """impact_analysis had no enricher: no trust, and no honesty check."""
    reg = patches._ENRICHERS
    assert "impact_analysis" in reg, \
        "the decision-grade task must carry a confidence stamp"


def test_impact_enricher_does_not_clobber_its_own_transitions():
    """impact emits a hectare-based table; the change enricher must not
    overwrite it with the pixel-fraction shape."""
    stamped = patches._ENRICHERS["impact_analysis"]
    out = {"confidence": 0.7, "transitions": {"built_up_new_ha": 1.0}}
    stamped({"images": []}, out)
    assert out["transitions"] == {"built_up_new_ha": 1.0}
