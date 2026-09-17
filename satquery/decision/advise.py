"""Plain-English advice -- the layer that answers "so what do I do?".

The audience
------------
A farmer, a district revenue official, or a relief coordinator standing in a
flood-affected block, without a GIS team and often without reliable internet.
They did not ask for a change mask; they asked whether something is happening on
ground they are responsible for and what to do about it.

So the output is written as a short report, not a metrics panel:

  * what changed, in hectares and everyday comparisons
  * what it means, without a single model name
  * what to do about it, addressed to a person
  * who to tell
  * how far to trust it, taken from measured reliability rather than tone
  * what this analysis cannot tell them

That last section is not filler. Telling someone the imagery cannot resolve
changes below a quarter hectare is the difference between a tool they use and a
tool that misleads them.

``JARGON`` lets the test suite enforce the plain-language promise instead of
leaving it as an aspiration.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .facts import human_area, num

# Terms that must never reach a non-expert. Matching is split deliberately:
#
#   * TOKENS are matched on word boundaries. Plain substring matching produced a
#     false positive on "previously" (it contains "iou"), which would have failed
#     the plain-language check on a perfectly ordinary sentence.
#   * STEMS are matched as substrings, because the whole family matters
#     ("calibrated", "calibration").
#
# "pixel" is deliberately absent: it only ever appears inside "metres per dot",
# which is the standard way to explain image sharpness and is explained in place.
JARGON_TOKENS = (
    "f1", "iou", "bleu", "softmax", "logit", "threshold", "inference",
    "embedding", "segmentation", "raster", "geotiff", "epsg", "crs",
    "ndvi", "tensor", "epoch", "recall", "precision", "checkpoint",
    "prob map", "change mask",
)
JARGON_STEMS = ("calibrat", "confidence score")
JARGON = JARGON_TOKENS + JARGON_STEMS

OUTCOME_LABELS = {
    "act": "Act on this",
    "verify_first": "Check before acting",
    "monitor": "Worth watching",
    "no_action": "No action needed",
    "insufficient_evidence": "Cannot conclude from this",
}


def find_jargon(text: str) -> List[str]:
    """Return any forbidden term present in ``text`` (case-insensitive)."""
    low = (text or "").lower()
    found = [term for term in JARGON_STEMS if term in low]
    for term in JARGON_TOKENS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            found.append(term)
    return found


def _area_sentence(facts: Dict[str, Any]) -> str:
    ha = facts.get("changed_area_ha")
    frac = facts.get("changed_fraction")
    if ha is not None:
        return f"{human_area(num(ha))} changed between the two dates."
    if frac is not None:
        return (f"About {num(frac) * 100:.0f}% of the area you asked about "
                f"changed between the two dates.")
    return ""


def _trust_sentence(trust: Dict[str, Any]) -> str:
    if not trust or trust.get("value") is None:
        return ("We have no measured accuracy for the method behind this result, "
                "so treat it as unverified.")
    band = str(trust.get("band") or "").replace("_", " ")
    ev = trust.get("band_evidence") or {}
    base = {
        "high": "This is strong enough to act on.",
        "moderate": "This is indicative — worth acting on after a quick check.",
        "low": "This is a lead rather than a finding. Verify before acting.",
        "very low": "This is not reliable enough to base a decision on.",
        "unverified": ("We have no measured reliability for this method, so treat "
                       "it as unverified."),
    }.get(band, "Treat this with caution.")
    if ev.get("observed_accuracy") is not None and ev.get("n"):
        # The honest part: what we showed, next to how often we were right.
        base += (f" Measured on held-out data: when results landed in this range, "
                 f"they were right about {num(ev['observed_accuracy']) * 100:.0f}% "
                 f"of the time ({ev['n']} cases).")
    return base


def _cannot_tell(record: Dict[str, Any]) -> List[str]:
    limits: List[str] = []
    floor = (record.get("detection_floor") or {})
    if floor.get("plain"):
        limits.append(floor["plain"])
    elif floor.get("gsd_m") is None:
        limits.append("We did not know the ground scale of these images, so we "
                      "cannot say what size of change they would miss.")
    stab = (record.get("sensitivity") or {})
    if stab.get("plain"):
        limits.append(stab["plain"])
    elif not stab.get("available", False) and stab.get("reason"):
        limits.append("We could not check how much this conclusion depends on "
                      "how strict we were about what counts as changed.")
    facts = record.get("facts", {}).get("values", {})
    if facts.get("gsd_assumed"):
        limits.append("We did not have the ground scale of these images, so we "
                      "assumed a typical value. Areas are therefore approximate.")
    for gap in record.get("blocked_by") or []:
        limits.append(f"We could not establish {gap.split(': ', 1)[-1]}, which is "
                      f"why we could not go further.")
    if not (record.get("trust") or {}).get("calibrated"):
        limits.append("The numbers behind this result have not been checked "
                      "against known outcomes, so how reliable it is has not "
                      "been measured for this particular tool.")
    # De-duplicate while keeping order.
    seen, out = set(), []
    for item in limits:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def compose(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build the non-expert narrative for a decision record."""
    facts = (record.get("facts") or {}).get("values", {}) or {}
    trust = record.get("trust") or {}
    rule_text = record.get("rule") or {}

    summary_bit = _area_sentence(facts)
    headline = rule_text.get("headline") or OUTCOME_LABELS.get(
        record.get("outcome", ""), "Result")

    what_we_found_parts = [p for p in (summary_bit,) if p]
    # Spatial detail only makes sense once there is something to locate. Saying
    # "86% of it is near water" next to "nothing large enough to be sure about"
    # reads as a contradiction, so negligible-change refusals skip the detail.
    detail_is_meaningful = record.get("rule_id") not in (
        "R4_below_detection_floor", "R0_no_measurement", "no_rule_applied")
    if detail_is_meaningful:
        nw = facts.get("near_water_500m")
        if nw is not None:
            what_we_found_parts.append(
                f"{num(nw) * 100:.0f}% of that change is within 500 metres of a "
                f"water body.")
        if facts.get("n_change_regions"):
            what_we_found_parts.append(
                f"It appears in {int(facts['n_change_regions'])} separate "
                f"patches.")
        if facts.get("priority_zone"):
            what_we_found_parts.append(
                f"The most affected part is the {facts['priority_zone']} of the "
                f"area.")

    advice = {
        "headline": headline,
        "outcome": record.get("outcome"),
        "outcome_label": OUTCOME_LABELS.get(record.get("outcome", ""), "Result"),
        "what_we_found": " ".join(what_we_found_parts).strip(),
        "what_it_means": rule_text.get("why", ""),
        "what_to_do": rule_text.get("action", ""),
        "who_to_tell": rule_text.get("who", ""),
        "how_far_to_trust": _trust_sentence(trust),
        "what_we_cannot_tell": _cannot_tell(record),
        "result_note": rule_text.get("confidence_note", ""),
        # Other conclusions that also hold. Shown separately so the headline
        # stays readable while nothing true is discarded.
        "also_true": [c.get("text", {}).get("headline", "")
                      for c in (record.get("contributing") or [])
                      if c.get("text", {}).get("headline")],
    }
    advice["one_line"] = " ".join(
        p for p in (advice["headline"], advice["what_to_do"]) if p).strip()

    # Enforce the plain-language promise at the point of composition, so a future
    # rule that leaks a metric name fails loudly rather than shipping.
    texts: List[str] = []
    for value in advice.values():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, (list, tuple)):
            texts.extend(v for v in value if isinstance(v, str))
    advice["jargon_found"] = sorted({t for t in texts for t in find_jargon(t)})
    return advice
