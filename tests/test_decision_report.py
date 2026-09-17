"""The decision must reach what the user actually opens.

Machine-readability was never the problem: `report.json` carried the decision
from the moment it was attached, because `write_report` dumps `outputs`
wholesale. The gap was the *human-readable* exports. A user who downloads the
Markdown or PDF is exactly the user the decision layer was written for, and both
rendered it as an entry inside a generic ``## Outputs`` JSON preview -- present,
and unusable.

These tests pin the fix, and pin it at the level a user experiences: "if I open
the downloaded report, is the answer to *what do I do* in it, in words?"
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from satquery import patches
from satquery.agent import AgentResult, write_report
from satquery.io_utils import RSImage
from satquery.decision.render import as_markdown, decision_heading


def _img():
    return RSImage(array=np.zeros((32, 32, 3), np.float32), format="geotiff",
                   modality="rgb")


def _result(run_id: str = "test-decision-report"):
    return AgentResult(
        run_id=run_id, query="is there new construction near the river?",
        configuration={"configuration_label": "single"},
        selected_task="impact_analysis", answer="Changed area ~4.9 ha",
        confidence=0.8,
        outputs={
            "changed_area_ha": 4.9, "changed_fraction": 0.031, "gsd_m": 10.0,
            "gsd_assumed": False, "confidence": 0.8,
            "confidence_meta": {
                "value": 0.8, "method": "formula",
                "component": "impact_analysis", "calibrated": False,
                "reliability": 0.818, "trust": 0.654, "trust_band": "high",
                "trust_advice": "The evidence supports acting on this.",
                "trust_source": "confidence_x_reliability",
                "limiting_factor": "reliability",
            },
            "transitions": {"built_up_new_ha": 3.2, "built_up_lost_ha": 0.0,
                            "vegetation_lost_ha": 0.1, "vegetation_gained_ha": 0.0},
            "near_water": {"within_250m": 0.1, "within_500m": 0.2,
                           "within_1000m": 0.4},
        },
        visuals={"prob_map": np.ones((32, 32), np.float32)},
        trace=[], report_paths={})


@pytest.fixture()
def reported(tmp_path, monkeypatch):
    import satquery.agent as agent
    from satquery.config import CONFIG
    monkeypatch.setattr(CONFIG, "runs_dir", tmp_path)
    res = _result()
    patches.enrich_result(res, [_img()])
    return res, write_report(res, [_img()])


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #

def test_enrichment_attaches_a_decision():
    res = _result()
    patches.enrich_result(res, [_img()])
    assert "decision" in res.outputs
    assert res.outputs["decision"]["advice"]["what_to_do"]


def test_kill_switch_still_suppresses_it(monkeypatch):
    """The frozen additive contract: no patch keys when patches are off."""
    monkeypatch.setenv("SATQUERY_PATCHES", "0")
    res = _result("test-decision-kill")
    patches.enrich_result(res, [_img()])
    assert "decision" not in res.outputs


# --------------------------------------------------------------------------- #
# report.json
# --------------------------------------------------------------------------- #

def test_json_report_carries_the_full_decision(reported):
    _, paths = reported
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    decision = payload["outputs"]["decision"]
    assert decision["schema"] == "anvesha.decision/1"
    for key in ("outcome", "rule_id", "advice", "facts", "trust"):
        assert key in decision
    assert decision["advice"]["what_to_do"]


# --------------------------------------------------------------------------- #
# report.md -- the readable export
# --------------------------------------------------------------------------- #

def test_markdown_report_presents_the_decision_in_words(reported):
    _, paths = reported
    md = paths["markdown"].read_text(encoding="utf-8").lower()
    for phrase in ("what to do", "who to tell", "how far to trust",
                   "cannot tell you", "what it means"):
        assert phrase in md, f"{phrase!r} missing from the downloadable report"


def test_markdown_decision_section_precedes_the_raw_dump(reported):
    _, paths = reported
    md = paths["markdown"].read_text(encoding="utf-8")
    assert md.find("## What to do") < md.find("## Outputs"), \
        "the decision must be read before the JSON preview, not buried in it"


def test_markdown_leads_with_the_action_not_a_metric(reported):
    _, paths = reported
    md = paths["markdown"].read_text(encoding="utf-8")
    section = md.split("## What to do", 1)[1]
    assert "New construction has appeared" in section


def test_markdown_contains_no_jargon(reported):
    from satquery.decision import find_jargon
    _, paths = reported
    md = paths["markdown"].read_text(encoding="utf-8")
    section = md.split("## What to do", 1)[1].split("## Outputs", 1)[0]
    assert find_jargon(section) == []


# --------------------------------------------------------------------------- #
# Renderer contract
# --------------------------------------------------------------------------- #

def test_renderer_is_empty_without_a_decision():
    for value in (None, {}, {"advice": {}}, "not a dict"):
        assert as_markdown(value) == []


def test_heading_names_the_outcome_for_a_reader():
    assert "Act on this" in decision_heading(
        {"outcome": "act", "advice": {"outcome_label": "Act on this"}})


def test_renderer_survives_a_partial_record():
    """A record from an older run must render what it has, not raise."""
    lines = as_markdown({"outcome": "monitor",
                         "advice": {"headline": "Worth watching"}})
    assert any("Worth watching" in ln for ln in lines)
    assert any("cannot tell you" not in ln for ln in lines)


def test_measured_reliability_is_quoted_when_present():
    lines = as_markdown({
        "outcome": "act",
        "advice": {"headline": "Act on this", "outcome_label": "Act on this"},
        "trust": {"band_evidence": {"claimed": 0.927, "observed_accuracy": 0.892,
                                    "n": 148}},
    })
    joined = " ".join(lines)
    assert "93%" in joined and "89%" in joined and "148" in joined
