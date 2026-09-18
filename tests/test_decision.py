"""Decision layer: the last stage must conclude honestly, or refuse.

These tests target the ways a decision layer goes wrong. Not "does it produce a
string" but: can it refuse, does it gate on measured trust, does a missing fact
become a refusal rather than a default, and does the advice stay readable by
someone without a GIS background.

Two of the fixed bugs have regression tests here by name, because both were
logic inversions that produced plausible-looking wrong answers:

  * a rule written to accept *either* a changed area *or* a fraction demanded
    both, so fraction-only runs could never conclude anything;
  * the "nothing to decide from" refusal was expressed as a requirement, so it
    fired on every run that had data.
"""
from __future__ import annotations

import numpy as np
import pytest

from anvesha.decision import (JARGON, compose, decide, find_jargon,
                               human_area, threshold_sweep)
from anvesha.decision.rules import RULES, refusal_rules, verdict_rules

ACT_META = {
    "value": 0.8, "method": "formula", "component": "impact_analysis",
    "calibrated": False, "reliability": 0.818, "trust": 0.654,
    "trust_band": "high", "trust_advice": "The evidence supports acting on this.",
    "trust_source": "confidence_x_reliability", "limiting_factor": "reliability",
}
WEAK_META = {**ACT_META, "trust": 0.12, "trust_band": "very_low",
             "trust_source": "unverified"}
CALIBRATED_META = {**ACT_META, "trust": 0.93, "trust_band": "high",
                   "calibrated": True, "band_evidence": {
                       "band": "0.90-0.95", "claimed": 0.927,
                       "observed_accuracy": 0.892, "n": 148, "optimism": 0.035,
                       "source": "measured"}}


def impact(**over):
    out = {
        "changed_area_ha": 4.9, "changed_fraction": 0.031, "gsd_m": 10.0,
        "gsd_assumed": False, "confidence": 0.8, "confidence_meta": ACT_META,
        "transitions": {"built_up_new_ha": 3.2, "built_up_lost_ha": 0.1,
                        "vegetation_lost_ha": 1.4, "vegetation_gained_ha": 0.2},
        "near_water": {"within_250m": 0.41, "within_500m": 0.86,
                       "within_1000m": 1.0},
        "num_change_regions": 3, "priority_zone": "north-east",
    }
    out.update(over)
    return out


NO_BUILD = {"built_up_new_ha": 0.0, "built_up_lost_ha": 0.0,
            "vegetation_lost_ha": 0.0, "vegetation_gained_ha": 0.0}
FAR_FROM_WATER = {"within_250m": 0.0, "within_500m": 0.02, "within_1000m": 0.05}


# --------------------------------------------------------------------------- #
# 1. The inversion bugs, by name
# --------------------------------------------------------------------------- #

def test_either_fact_is_enough_not_both():
    """A fraction-only run (change_analysis shape) must still reach a verdict.

    Regression: ``requires_any`` was implemented as "all", so every rule that
    accepted an area *or* a fraction silently demanded both and no
    fraction-only run could ever conclude.
    """
    rec = decide({"changed_area_fraction": 0.12, "confidence": 0.8,
                  "confidence_meta": ACT_META, "num_change_regions": 2})
    assert rec["outcome"] != "insufficient_evidence"
    assert rec["rule_id"] == "V4_modest_change"


def test_the_nothing_to_decide_rule_does_not_fire_when_there_is_data():
    """Regression: the absence-checked refusal fired on every run with a result."""
    for out in (impact(), {"changed_area_fraction": 0.12, "confidence": 0.8},
                impact(near_water=FAR_FROM_WATER, transitions=NO_BUILD)):
        rec = decide(out)
        assert rec["rule_id"] != "R0_no_measurement", \
            "the no-measurement refusal must only fire when no measurement exists"
        assert rec["outcome"] != "insufficient_evidence"


def test_no_measurement_refusal_fires_when_genuinely_absent():
    rec = decide({"answer": "a caption", "confidence": 0.9})
    assert rec["outcome"] == "insufficient_evidence"
    assert rec["rule_id"] == "R0_no_measurement"


# --------------------------------------------------------------------------- #
# 2. It can refuse, and says why
# --------------------------------------------------------------------------- #

def test_weak_trust_yields_verify_not_act():
    """Advice must never outrun the trust figure printed beside it."""
    rec = decide(impact(confidence_meta=WEAK_META))
    assert rec["outcome"] == "verify_first"
    assert rec["advice"]["outcome_label"] == "Check before acting"
    assert "field visit" in rec["advice"]["what_to_do"]


def test_unmeasured_method_is_flagged_rather_than_assumed():
    rec = decide(impact(confidence_meta={**ACT_META, "trust": 0.5,
                                         "trust_source": "unverified"}))
    assert rec["outcome"] == "verify_first"
    assert rec["rule_id"] == "R2_unmeasured_method"


def test_below_detection_floor_refuses_to_call_it_a_finding():
    rec = decide(impact(changed_area_ha=0.05, changed_fraction=0.0003))
    assert rec["outcome"] == "no_action"
    assert rec["rule_id"] == "R4_below_detection_floor"
    assert "smaller than" in rec["advice"]["what_it_means"]


def test_zero_change_is_not_reported_as_too_small_to_see():
    """Nothing changed and something too small to trust are different answers."""
    rec = decide(impact(changed_area_ha=0.0, changed_fraction=0.0,
                        near_water=FAR_FROM_WATER, transitions=NO_BUILD))
    assert rec["rule_id"] != "R4_below_detection_floor"
    assert rec["outcome"] == "no_action"


def test_a_refusal_never_reports_missing_facts_as_blockers_once_decided():
    rec = decide(impact())
    assert rec["outcome"] == "act"
    assert rec["blocked_by"] == [], \
        "gaps for rules that were not needed must not read as blockers"


def test_insufficient_evidence_names_what_was_missing():
    rec = decide({"confidence": 0.5, "confidence_meta": ACT_META})
    assert rec["outcome"] == "insufficient_evidence"
    assert rec["advice"]["what_we_cannot_tell"]


def test_engine_never_raises_on_junk():
    for junk in ({}, {"changed_area_ha": None}, {"near_water": None},
                 {"transitions": []}, {"changed_fraction": "not a number"}):
        rec = decide(junk)
        assert rec["outcome"] in ("insufficient_evidence", "no_action",
                                 "verify_first", "monitor", "act")


# --------------------------------------------------------------------------- #
# 3. The outcome vocabulary is closed
# --------------------------------------------------------------------------- #

def test_only_known_outcomes_are_produced():
    allowed = {"act", "verify_first", "monitor", "no_action",
               "insufficient_evidence"}
    assert {r.outcome for r in RULES} <= allowed


def test_refusals_are_all_evaluated_before_any_verdict():
    """Position in the list is the semantic guarantee."""
    kinds = [r.kind for r in RULES]
    first_verdict = kinds.index("verdict")
    assert all(k == "refusal" for k in kinds[:first_verdict]), \
        "every refusal must precede every verdict in evaluation order"


def test_rule_ids_are_unique():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# 4. Multiple true conclusions survive
# --------------------------------------------------------------------------- #

def test_construction_beside_water_leads_with_the_construction():
    rec = decide(impact())
    assert rec["rule_id"] == "V2_encroachment"
    assert rec["advice"]["headline"].startswith("New construction")
    # ...and the water context is not thrown away.
    assert any("water" in h.lower() for h in rec["advice"]["also_true"])


def test_near_water_alone_still_reaches_the_flood_conclusion():
    rec = decide(impact(transitions=NO_BUILD))
    assert rec["rule_id"] == "V1_flood_exposure"
    assert rec["outcome"] == "act"


def test_the_report_never_contradicts_itself():
    """The generic tiers are fallbacks and must not fire beside a real finding.

    Regression: V4/V5 had no exclusivity guard, so a run with new construction
    also reported "nothing here needs action" in the same output. Both were true
    statements about the data; only one of them was the answer.
    """
    rec = decide(impact())
    assert rec["rule_id"] == "V2_encroachment"
    for other in rec["advice"]["also_true"]:
        assert "nothing here needs action" not in other.lower(), \
            "a fallback tier leaked into a decision that found something"
        assert "nothing that needs urgent action" not in other.lower()


def test_fallback_tiers_are_reachable_when_nothing_specific_applies():
    quiet = decide(impact(near_water=FAR_FROM_WATER, transitions=NO_BUILD))
    assert quiet["rule_id"] == "V4_modest_change"
    assert quiet["outcome"] == "monitor"


def test_no_verdict_rule_is_both_specific_and_fallback():
    """A rule that is specific must be consultable in the first pass."""
    from anvesha.decision.rules import fallback_rules
    fb = {r.id for r in fallback_rules()}
    assert fb <= {"V4_modest_change", "V5_no_action"}
    # Every non-fallback verdict must be evaluable without the fallback pass.
    assert "V2_encroachment" not in fb
    assert "V1_flood_exposure" not in fb


# --------------------------------------------------------------------------- #
# 5. Facts carry provenance
# --------------------------------------------------------------------------- #

def test_every_reported_fact_records_where_it_came_from():
    rec = decide(impact())
    prov = rec["facts"]["provenance"]
    assert prov.get("changed_area_ha")
    assert prov.get("near_water_500m")
    assert rec["facts"]["absent"]


def test_transition_shape_is_identified_not_assumed():
    rec = decide(impact())
    assert rec["facts"]["values"]["transitions_shape"] == "per-class hectares"
    summary = decide({"changed_area_fraction": 0.1, "confidence": 0.8,
                      "transitions": {"changed_fraction": 0.1, "top": []}})
    assert summary["facts"]["values"]["transitions_shape"] == "summary table"
    # The hectare-based rule must not fire on the summary shape.
    assert summary["rule_id"] != "V2_encroachment"


# --------------------------------------------------------------------------- #
# 6. Plain language is enforced, not promised
# --------------------------------------------------------------------------- #

def _every_text(rec):
    a = rec["advice"]
    out = [v for v in a.values() if isinstance(v, str)]
    for v in a.values():
        if isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, str))
    return out


@pytest.mark.parametrize("record", [
    decide(impact()),
    decide(impact(transitions=NO_BUILD)),
    decide(impact(confidence_meta=WEAK_META)),
    decide(impact(changed_area_ha=0.05, changed_fraction=0.0003)),
    decide(impact(changed_area_ha=0.0, changed_fraction=0.0,
                  near_water=FAR_FROM_WATER, transitions=NO_BUILD)),
    decide({"answer": "junk"}),
    decide({"changed_area_fraction": 0.12, "confidence": 0.8,
            "confidence_meta": CALIBRATED_META}),
])
def test_advice_contains_no_jargon(record):
    leaked = record["advice"]["jargon_found"]
    assert leaked == [], f"jargon reached the user: {leaked}"


def test_jargon_matcher_uses_word_boundaries():
    """\"previously\" contains \"iou\" and must not trigger the guard."""
    assert find_jargon("water spread onto land that was previously dry") == []
    assert find_jargon("the F1 score is 0.82") == ["f1"]
    assert find_jargon("this is calibrated") == ["calibrat"]


def test_every_rule_renders_jargon_free_text():
    """A future rule that leaks a metric name fails here, not in front of a user."""
    facts = None
    from anvesha.decision.facts import assemble
    facts = assemble(impact())
    for rule in RULES:
        if not rule.render:
            continue
        text = rule.render(facts)
        for key in ("headline", "why", "action", "who", "confidence_note"):
            for term in find_jargon(str(text.get(key, ""))):
                pytest.fail(f"{rule.id}.{key} contains {term!r}")


def test_trust_sentence_quotes_the_measured_number():
    rec = decide(impact(confidence_meta=CALIBRATED_META))
    sentence = rec["advice"]["how_far_to_trust"]
    assert "89%" in sentence          # the observed accuracy, not the claim
    assert "148" in sentence          # and the sample it came from


def test_what_we_cannot_tell_states_the_imagery_limit():
    rec = decide(impact())
    limits = " ".join(rec["advice"]["what_we_cannot_tell"]).lower()
    assert "hectares" in limits
    assert "10 metres" in limits or "10 m" in limits


# --------------------------------------------------------------------------- #
# 7. Sensitivity
# --------------------------------------------------------------------------- #

def test_threshold_sweep_is_monotone():
    prob = np.linspace(0.0, 1.0, 10000).reshape(100, 100)
    sweep = threshold_sweep(prob)
    fractions = [p["changed_fraction"] for p in sweep["points"]]
    assert fractions == sorted(fractions, reverse=True), \
        "a stricter cutoff can never find more change"


def test_sensitivity_states_a_margin_and_brackets_the_nominal_setting():
    prob = np.full((200, 200), 0.95, dtype=np.float32)
    stab = decide(impact(), prob_map=prob)["sensitivity"]
    assert stab["available"] is True
    assert stab["stable_from"] <= stab["nominal_threshold"] <= stab["stable_to"]
    assert stab["plain"] and not find_jargon(stab["plain"])


def test_an_unambiguous_map_holds_across_every_setting():
    """Every pixel saturated: no cutoff can undo the finding."""
    prob = np.ones((200, 200), dtype=np.float32)
    stab = decide(impact(), prob_map=prob)["sensitivity"]
    assert stab["holds_across_full_range"] is True
    assert "does not depend" in stab["plain"]


def test_a_borderline_map_reports_a_margin_not_a_certainty():
    """Change sitting right at the cutoff must be reported as threshold-dependent."""
    prob = np.full((200, 200), 0.60, dtype=np.float32)
    stab = decide(impact(), prob_map=prob)["sensitivity"]
    assert stab["available"] is True
    assert stab["holds_across_full_range"] is False
    assert stab["reverses_outside"] is True
    assert "margin" in stab["plain"]


def test_missing_map_is_stated_not_hidden():
    rec = decide(impact())
    if not rec["sensitivity"].get("available"):
        assert rec["sensitivity"]["reason"]
        assert any("strict" in t.lower()
                   for t in rec["advice"]["what_we_cannot_tell"])


# --------------------------------------------------------------------------- #
# 8. Everyday units
# --------------------------------------------------------------------------- #

def test_human_area_uses_familiar_comparisons():
    assert "football fields" in human_area(4.9)
    assert "acres" in human_area(0.2)
    assert human_area(None) == "an unknown area"


# --------------------------------------------------------------------------- #
# 9. Offline by construction
# --------------------------------------------------------------------------- #

def test_decision_layer_needs_no_network_and_no_models(monkeypatch):
    """The decision layer must work with the grid down."""
    import socket
    import anvesha.decision.engine as eng
    import anvesha.decision.facts as fct
    import anvesha.decision.advise as adv
    import anvesha.decision.sensitivity as sen

    def _boom(*a, **k):
        raise AssertionError("decision layer attempted network access")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    rec = decide(impact(), prob_map=np.full((64, 64), 0.9, dtype=np.float32))
    assert rec["outcome"] == "act"


def test_public_api_is_composable_from_a_plain_dict():
    from anvesha.decision import decide as d
    rec = d(impact())
    assert rec["schema"] == "anvesha.decision/1"
    assert compose(rec)["outcome_label"]
