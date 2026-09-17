"""Decision engine -- the final layer: "what do I do with this analysis?".

What this layer is for
----------------------
Every stage before it answers *what is there*. This one answers the question a
non-expert actually has: **now what?** Not a restatement of the analysis, but a
conclusion plus a course of action, with the limits stated.

How it stays honest
-------------------
1. **Refusals run before verdicts.** If the question was ambiguous, the method's
   reliability is unmeasured, trust is too low, or the change is smaller than the
   imagery can resolve, the layer stops and says so. A decision layer that cannot
   refuse will manufacture an answer from whatever it was handed.
2. **Trust is carried, not assumed.** The verdict is gated on the same trust
   figure the confidence layer computed from measured reliability, so the advice
   and the number beside it can never disagree.
3. **A missing fact refuses; it never defaults.** Rules declare what they need,
   and an unmet requirement produces ``insufficient_evidence`` naming the gap
   rather than a fall through to "no action".
4. **Offline by construction.** Pure functions over artefacts already on disk --
   no network, no model loads. It works when the grid is down, which is exactly
   when a flood-impact question gets asked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import advise as _advise
from .facts import Facts, assemble
from .rules import (MINOR_FRACTION, SIGNIFICANT_AREA_HA, SIGNIFICANT_FRACTION,
                    Rule, fallback_rules, refusal_rules, verdict_rules)
from .sensitivity import (detection_floor_ha, detection_floor_text,
                          stability_range, stability_text, threshold_sweep)

SCHEMA = "anvesha.decision/1"


def _augment(facts: Facts, outputs: Dict[str, Any],
             intent_confidence: Optional[float]) -> Facts:
    """Add the facts that come from the run envelope rather than the tool output."""
    if intent_confidence is not None:
        facts._set("intent_confidence", float(intent_confidence),
                   "reported by the intent router")
    meta = (outputs or {}).get("confidence_meta") or {}
    if meta.get("trust_source") == "unverified":
        facts._set("method_unmeasured", True, "confidence layer reported no "
                                              "measured reliability")
    for key in ("date_a", "date_b"):
        if (outputs or {}).get(key) is not None:
            facts._set(key, outputs[key], "reported")
    return facts


def _total_area_ha(facts: Facts) -> Optional[float]:
    """Scene area, recovered exactly from the area/fraction pair.

    ``changed_area_ha`` is pixels x pixel-area and ``changed_fraction`` is
    pixels / total-pixels, so their ratio is the scene area -- no assumption
    needed. That lets an area-based rule be re-tested across thresholds as well.
    """
    ha, frac = facts.get("changed_area_ha"), facts.get("changed_fraction")
    if ha is None or not frac:
        return None
    return float(ha) / float(frac)


def _predicate(rule: Rule, facts: Facts):
    """The rule's own significance test, expressed on a changed fraction.

    Used to re-evaluate the decision at other thresholds. For a class-specific
    rule (say new construction) the probability field only carries *total*
    change, so the sweep re-tests the overall extent; the record says so rather
    than implying the class breakdown was re-derived.
    """
    total = _total_area_ha(facts)

    def crosses(fraction: float) -> bool:
        if rule.id in ("V2_encroachment", "V3_vegetation_loss"):
            if total is None:
                return fraction >= SIGNIFICANT_FRACTION
            return fraction * total >= SIGNIFICANT_AREA_HA
        if rule.id == "V1_flood_exposure":
            return fraction >= SIGNIFICANT_FRACTION
        if rule.id == "V4_modest_change":
            return fraction >= MINOR_FRACTION
        return fraction >= MINOR_FRACTION

    return crosses


def _sensitivity(rule: Rule, facts: Facts,
                 prob_map: Optional[np.ndarray]) -> Dict[str, Any]:
    if prob_map is None:
        return {"available": False,
                "reason": "this analysis did not keep the raw probability field"}
    sweep = threshold_sweep(np.asarray(prob_map))
    report = stability_range(sweep, _predicate(rule, facts))
    if report.get("available"):
        report["plain"] = stability_text(report, n_settings=len(sweep["points"]))
        if rule.id in ("V2_encroachment", "V3_vegetation_loss"):
            report["applies_to"] = ("the overall extent of change; the split "
                                    "between land types is not re-tested")
    return report


def decide(outputs: Dict[str, Any], task: str = "",
           visuals: Optional[Dict[str, Any]] = None,
           images: Optional[List[Any]] = None,
           intent_confidence: Optional[float] = None,
           prob_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Produce a decision record for one analysis run.

    Never raises on missing data: an absent fact is a finding about the evidence,
    not an error, and it is reported as one.
    """
    facts = assemble(outputs or {}, task=task, visuals=visuals, images=images)
    _augment(facts, outputs or {}, intent_confidence)

    if prob_map is None:
        vis = visuals or {}
        cand = vis.get("prob_map")
        if cand is None:
            src = (outputs or {})
            inner = src.get("impact_analysis") if isinstance(src.get("impact_analysis"), dict) else src
            cand = ((inner or {}).get("_visual") or {}).get("prob_map")
        prob_map = cand

    # ---- trust gate ------------------------------------------------------- #
    meta = (outputs or {}).get("confidence_meta") or {}
    trust: Dict[str, Any] = {}
    if meta:
        trust = {
            "value": meta.get("trust"),
            "band": meta.get("trust_band"),
            "advice": meta.get("trust_advice"),
            "source": meta.get("trust_source"),
            "limiting_factor": meta.get("limiting_factor"),
            "reliability": meta.get("reliability"),
            "calibrated": bool(meta.get("calibrated")),
            "band_evidence": meta.get("band_evidence"),
        }

    # ---- refusals first --------------------------------------------------- #
    checked: List[Dict[str, Any]] = []
    chosen: Optional[Rule] = None
    unmet: List[str] = []

    for rule in refusal_rules():
        if rule.applies(facts):
            chosen = rule
            checked.append({"rule": rule.id, "fired": True,
                            "outcome": rule.outcome})
            break
        ready = rule.facts_ready(facts)
        checked.append({"rule": rule.id, "fired": False,
                        "requirements_met": ready,
                        "missing": rule.missing(facts)})

    # ---- then verdicts ---------------------------------------------------- #
    # Every applicable verdict is collected, not just the first: more than one
    # conclusion can be true at once (new construction that also sits beside
    # water), and dropping the others would hide a real finding. The first is
    # the headline; the rest are reported alongside it.
    contributing: List[Dict[str, Any]] = []
    if chosen is None:
        # Specific verdicts first. The generic tiers are consulted only if none
        # of these applies, so a report can never say "new construction has
        # appeared" and "nothing here needs action" about the same run.
        for rule in verdict_rules():
            if rule.applies(facts):
                checked.append({"rule": rule.id, "fired": True,
                                "outcome": rule.outcome})
                if chosen is None:
                    chosen = rule
                else:
                    contributing.append({
                        "rule_id": rule.id, "outcome": rule.outcome,
                        "domain": rule.domain,
                        "text": rule.render(facts) if rule.render else {},
                    })
                continue
            if not rule.facts_ready(facts):
                unmet.append(f"{rule.id}: " + ", ".join(rule.missing(facts)))

    if chosen is None:
        for rule in fallback_rules():
            if rule.applies(facts):
                chosen = rule
                checked.append({"rule": rule.id, "fired": True,
                                "outcome": rule.outcome})
                break

    if chosen is None:
        # Nothing fired. If rules were blocked by absent facts, say which -- a
        # silent fall-through to "no action" is the failure mode this layer was
        # built to avoid.
        outcome = "insufficient_evidence"
        rule_text = {
            "headline": "We cannot reach a conclusion from this analysis.",
            "why": ("The analysis did not report the information the decision "
                    "needs, so any recommendation would be guesswork."
                    if unmet else
                    "No rule in this system covers what the analysis found."),
            "action": ("Re-run the comparison, or ask a more specific question "
                       "about the area, and try again."),
            "who": "You",
            "confidence_note": "Refused rather than guessed.",
        }
        rule_id = "no_rule_applied"
    else:
        outcome = chosen.outcome
        rule_text = chosen.render(facts) if chosen.render else {}
        rule_id = chosen.id

    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "outcome": outcome,
        "rule_id": rule_id,
        "domain": chosen.domain if chosen else "general",
        "rule": rule_text,
        "trust": trust,
        "confidence": facts.get("confidence"),
        "facts": facts.to_dict(),
        "rules_checked": checked,
        "contributing": contributing,
        # Only populated when nothing could be decided. Gaps that merely made a
        # *less* specific rule inapplicable are not blockers, and reporting them
        # here would fill the user's "what we cannot tell" section with caveats
        # about rules they never needed.
        "blocked_by": unmet if chosen is None else [],
    }

    record["detection_floor"] = {
        "gsd_m": facts.get("gsd_m"),
        "hectares": detection_floor_ha(facts.get("gsd_m")),
        "plain": ("" if facts.get("gsd_m") is None
                  else detection_floor_text(facts.get("gsd_m"))),
        "note": "A conservative operating floor, not a measured quantity.",
    }
    record["sensitivity"] = _sensitivity(chosen, facts, prob_map) if chosen else {
        "available": False, "reason": "no rule applied"}
    record["advice"] = _advise.compose(record)
    return record


def decide_for_result(result: Any, images: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Convenience wrapper over an ``AgentResult`` (works offline, no models)."""
    outputs = getattr(result, "outputs", {}) or {}
    visuals = getattr(result, "visuals", {}) or {}
    probe = outputs if "confidence_meta" in outputs else (
        outputs.get("impact_analysis") if isinstance(outputs.get("impact_analysis"), dict) else outputs)
    return decide(outputs,
                  task=getattr(result, "selected_task", "") or "",
                  visuals=visuals, images=images,
                  intent_confidence=(outputs or {}).get("intent_confidence"),
                  prob_map=(visuals or {}).get("prob_map"))
