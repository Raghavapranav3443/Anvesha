"""Authority lane in the decision layer: ISRO's own map beside our measurement.

Three ways this layer could go wrong, one test each:

  * **Fabricated official agreement.** Bhuvan answers HTTP 200 with a valid PNG
    for a layer that has no data in the window, so a client that counts rendered
    tiles reports "ISRO records this as forest" over a blank image. Verified-only
    is what stops that, and it is asserted here rather than assumed.
  * **The flagship path refusing.** An investigation run nests its tool outputs
    under ``investigation``; the fact layer used to read only the top level, so
    every "full analysis" run reported "we could not find anything to base a
    decision on" while holding hectares, water proximity and ranked zones.
  * **Context quietly changing an uploaded pair.** No acquisition context must
    mean byte-identical advice to what the layer produced before it existed.

Plus the honesty rule that matters most: presence is not agreement. A map
recording open land under new construction sharpens *who to tell*; it never
turns a weak detection into a confident one, and only a genuine conflict with
water is allowed to move the verdict -- downward.
"""
from __future__ import annotations

import pytest

from anvesha.decision import decide
from anvesha.decision.authority import MIN_CONFLICT_AREA_HA
from anvesha.decision.rules import SIGNIFICANT_AREA_HA

ACT_META = {
    "value": 0.8, "method": "formula", "component": "impact_analysis",
    "calibrated": False, "reliability": 0.818, "trust": 0.654,
    "trust_band": "high", "trust_advice": "The evidence supports acting on this.",
    "trust_source": "confidence_x_reliability", "limiting_factor": "reliability",
}


def impact(**over) -> dict:
    out = {
        "changed_area_ha": 4.9, "changed_fraction": 0.031, "gsd_m": 10.0,
        "gsd_assumed": False, "confidence": 0.8, "confidence_meta": ACT_META,
        "transitions": {"built_up_new_ha": 3.2, "built_up_lost_ha": 0.1,
                        "vegetation_lost_ha": 1.4, "vegetation_gained_ha": 0.2},
        "near_water": {"within_250m": 0.41, "within_500m": 0.20,
                       "within_1000m": 0.30},
        "num_change_regions": 3, "priority_zone": "north-east",
    }
    out.update(over)
    return out


# Real measurements from a live run over Dibrugarh, Assam: a no-data render sits
# at the service's own ~3% framing baseline, a layer with data clears it by a wide
# margin. The numbers are kept so the threshold is pinned against reality.
def layer(theme: str, label: str, coverage: float, control: float) -> dict:
    return {"layer": f"nuis:AS_DI_{theme}", "theme": theme, "theme_label": label,
            "state": "AS", "coverage": coverage, "control_coverage": control,
            "evidence": round(max(0.0, coverage - control), 4)}


FOREST = layer("forest", "forest", 0.288, 0.016)
WATER = layer("waterbody", "water body", 0.180, 0.016)
BUILT = layer("builtup_urban", "built-up (urban)", 0.150, 0.016)
BLANK = layer("wetland", "wetland", 0.031, 0.030)          # no data in this window


# --------------------------------------------------------------------------- #
# 1. A blank tile is not corroboration
# --------------------------------------------------------------------------- #

def test_a_blank_isro_render_is_never_reported_as_agreement():
    """Regression guard for the fabrication risk this lane exists to prevent."""
    rec = decide(impact(isro_context=[BLANK]))
    assert rec["authority"]["available"] is False
    assert rec["facts"]["values"].get("isro_context_present") is None
    assert "ISRO" not in rec["advice"]["what_we_found"]
    assert rec["advice"]["what_the_authority_says"] == ""


def test_evidence_is_recomputed_so_it_cannot_be_laundered():
    """Without ``evidence``, coverage minus control still decides presence."""
    unverified = dict(BLANK)
    unverified.pop("evidence")
    assert decide(impact(isro_context=[unverified]))["authority"]["available"] is False
    verified = dict(FOREST)
    verified.pop("evidence")
    assert decide(impact(isro_context=[verified]))["authority"]["available"] is True


# --------------------------------------------------------------------------- #
# 2. Presence is not agreement
# --------------------------------------------------------------------------- #

def test_open_land_under_new_construction_is_named_in_the_finding():
    rec = decide(impact(isro_context=[FOREST]))
    assert rec["outcome"] == "act"                    # never upgraded by context
    assert rec["rule_id"] == "V2_encroachment"
    assert "ISRO" in rec["advice"]["what_we_found"]
    assert "forest" in rec["advice"]["what_we_found"]
    assert rec["advice"]["authority_available"] is True


def test_a_record_that_already_shows_built_up_reads_differently():
    """Mapped built-up ground is development, not construction on open land.

    The same finding must not be described the same way in both cases: telling a
    revenue officer "no official record of this building" when the official map
    already shows built-up ground would send them after the wrong question.
    """
    rec = decide(impact(isro_context=[BUILT]))
    text = rec["advice"]["what_we_found"] + " " + rec["advice"]["what_it_means"]
    assert "already" in text and "built-up" in text
    assert "not something the official map shows" not in text


def test_context_survives_a_refusal_without_being_mixed_into_the_measurement():
    """A refusal still shows the official record, in its own field."""
    rec = decide(impact(changed_area_ha=0.05, changed_fraction=0.0003,
                        isro_context=[FOREST]))
    assert rec["outcome"] == "no_action"
    assert "ISRO" not in rec["advice"]["what_we_found"]
    assert "ISRO" in rec["advice"]["what_the_authority_says"]


# --------------------------------------------------------------------------- #
# 3. The one conflict that moves the verdict — downward
# --------------------------------------------------------------------------- #

def test_building_on_officially_mapped_water_stops_to_check():
    rec = decide(impact(isro_context=[WATER]))
    assert rec["rule_id"] == "V0_isro_water_conflict"
    assert rec["outcome"] == "verify_first"
    # The construction finding is not discarded, it stops being the headline.
    assert any(c["rule_id"] == "V2_encroachment" for c in rec["contributing"])
    assert "water" in rec["advice"]["what_it_means"]


def test_a_small_structure_on_mapped_water_does_not_move_the_verdict():
    small = {"built_up_new_ha": 0.2, "built_up_lost_ha": 0.0,
             "vegetation_lost_ha": 0.0, "vegetation_gained_ha": 0.0}
    rec = decide(impact(transitions=small, isro_context=[WATER]))
    assert rec["rule_id"] != "V0_isro_water_conflict"


def test_the_conflict_floor_matches_the_significance_floor():
    """Two constants for one concept drift apart; this makes it fail loudly."""
    assert MIN_CONFLICT_AREA_HA == SIGNIFICANT_AREA_HA


# --------------------------------------------------------------------------- #
# 4. Absent context changes nothing
# --------------------------------------------------------------------------- #

def test_no_context_produces_the_same_advice_as_before_this_existed():
    without = decide(impact())
    assert without["authority"] == {"available": False, "layers": [], "themes": [],
                                    "sentences": []}
    assert without["advice"]["what_the_authority_says"] == ""
    assert without["advice"]["authority_available"] is False
    assert without["advice"]["what_we_found"] == decide(impact())["advice"]["what_we_found"]


def test_an_empty_or_malformed_context_is_ignored_rather_than_guessed_at():
    for bad in ([], {}, None, "ISRO says forest", [1, 2, 3],
                [{"theme": "forest"}], {"layers": "nope"}):
        rec = decide(impact(isro_context=bad))
        assert rec["authority"]["available"] is False
        assert rec["rule_id"] == "V2_encroachment"


# --------------------------------------------------------------------------- #
# 5. Investigation runs reach the decision layer at all
# --------------------------------------------------------------------------- #

def test_investigation_outputs_are_read_by_the_fact_layer():
    """Regression: the flagship \"full analysis\" path refused on every run.

    ``_run_investigation`` nests each tool output under ``investigation``
    (``investigation.impact``, ``investigation.change``). The fact layer read only
    the top level, so ``changed_area_ha`` was absent, ``R0_no_measurement`` fired,
    and the user was told there was nothing to base a decision on while the run
    held hectares, water proximity and ranked zones.
    """
    outputs = {"investigation": {
        "impact": impact(),
        "change": {"changed_area_fraction": 0.031},
    }, "suggestions": []}
    rec = decide(outputs, task="investigation")
    assert rec["rule_id"] != "R0_no_measurement"
    assert rec["outcome"] != "insufficient_evidence"
    assert rec["facts"]["values"]["changed_area_ha"] == 4.9
    assert rec["rule_id"] == "V2_encroachment"


def test_investigation_runs_get_their_trust_from_the_nested_stamp():
    """The trust gate must read the same confidence the advice quotes."""
    outputs = {"investigation": {"impact": impact()}, "suggestions": []}
    rec = decide(outputs, task="investigation")
    assert rec["trust"]["value"] == pytest.approx(0.654)
    assert rec["trust"]["band"] == "high"
    assert "no measured accuracy" not in rec["advice"]["how_far_to_trust"]


# --------------------------------------------------------------------------- #
# 6. The plain-language promise holds for the new sentences too
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("context", [[FOREST], [WATER], [BUILT], [FOREST, WATER, BUILT]])
def test_authority_advice_carries_no_jargon(context):
    rec = decide(impact(isro_context=context))
    assert rec["advice"]["jargon_found"] == [], rec["advice"]["jargon_found"]
