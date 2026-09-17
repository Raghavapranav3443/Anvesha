"""Decision rules -- what to do with the analysis, in plain English.

Design rules that matter more than the individual entries
--------------------------------------------------------
**Refusals are evaluated first.** A decision layer that cannot say "we cannot
conclude this" will manufacture a conclusion from whatever it was handed. The
``kind="refusal"`` rules run before any verdict rule and can end evaluation.

**A rule declares the facts it needs.** If a declared fact is neither reported
nor derivable, the rule does not fire and the engine reports
``insufficient_evidence`` naming the gap. This is what stops a
``change_analysis`` run -- which reports a *fraction* -- from silently
evaluating hectare-based rules against missing keys and concluding "no action".

**The vocabulary is closed.** Exactly five outcomes exist, so a downstream
consumer never has to guess what a new string means:

  ``act``                  the evidence is strong enough to act on now
  ``verify_first``         promising, but confirm before spending effort
  ``monitor``              note it and look again next time
  ``no_action``            nothing found that warrants action
  ``insufficient_evidence`` we cannot conclude this, and here is what is missing

**Every rule writes for a non-expert.** The people this is for are farmers,
district officials and relief coordinators working without a GIS team. No rule
may emit a metric name; ``tests/test_decision.py`` enforces that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .facts import Facts, human_area, num
from .sensitivity import detection_floor_ha

# --------------------------------------------------------------------------- #
# Thresholds. Each is an operating choice, labelled as such -- none of them is
# presented to a user as a measurement.
# --------------------------------------------------------------------------- #

# A change touching this share of the analysed area is worth a person's time.
SIGNIFICANT_FRACTION = 0.02
# ...or this much ground, which is about an acre and a half at field scale.
SIGNIFICANT_AREA_HA = 0.5
# Above this, most of the change sits close enough to water to matter for
# drainage and flood exposure.
WATER_PROXIMITY_STRONG = 0.5
# Trust below this is not enough to act on, only to check.
TRUST_ACTION_FLOOR = 0.25
# Trust below this means the method itself is not dependable.
TRUST_FLOOR = 0.10
# Below this, the router was not confident it picked the right analysis.
ROUTING_CONFIDENCE_FLOOR = 0.5
# Change below this share of the scene is present but not worth a visit.
MINOR_FRACTION = 0.005


@dataclass
class Rule:
    id: str
    kind: str                      # "refusal" | "verdict"
    outcome: str
    requires: Tuple[str, ...] = ()
    requires_any: Tuple[Tuple[str, ...], ...] = ()
    # Fire only when *every* one of these facts is absent. Needed by the
    # "nothing to decide from" refusal, whose condition is the absence of a
    # measurement rather than its presence. Expressing that with ``requires_any``
    # inverts the logic and makes the rule fire on every run that has data --
    # which is exactly the mistake this field was added to fix.
    fires_when_absent: Tuple[str, ...] = ()
    blocker: bool = True           # refusals: True means "stop here"
    # A fallback verdict only applies when no specific verdict did. Without this
    # the generic tiers fire alongside real findings and the report contradicts
    # itself -- "new construction has appeared" next to "nothing here needs
    # action", in the same output. Both are true statements about the data and
    # only one of them is the answer.
    fallback: bool = False
    when: Optional[Callable[[Facts], bool]] = None
    render: Optional[Callable[[Facts], Dict[str, Any]]] = None
    domain: str = "general"
    description: str = ""
    missing_reason: str = ""

    def facts_ready(self, facts: Facts) -> bool:
        """``requires`` = all of these; ``requires_any`` = at least one group.

        The two are different operators and implementing ``requires_any`` as
        "all" made every alternative-fact rule unsatisfiable -- a run reporting
        only a changed *fraction* could not satisfy a rule written to accept
        either a fraction or an area.
        """
        if not facts.has(*self.requires):
            return False
        if self.requires_any and not any(facts.has(*g)
                                         for g in self.requires_any):
            return False
        return True

    def applies(self, facts: Facts) -> bool:
        """Ready *and* the situation actually matches.

        Presence alone is not applicability: a run that reports proximity to
        water still must show enough change for the flood rule to mean anything.
        """
        for key in self.fires_when_absent:
            if facts.has(key):
                return False
        if not self.facts_ready(facts):
            return False
        return True if self.when is None else bool(self.when(facts))

    def missing(self, facts: Facts) -> List[str]:
        out = list(facts.missing(*self.requires))
        if self.requires_any and not any(facts.has(*g)
                                         for g in self.requires_any):
            alt = sorted({k for group in self.requires_any for k in group})
            out.append(" or ".join(alt))
        return out


def _pct(value: float) -> str:
    return f"{float(value) * 100:.0f}%"


def _area_from(facts: Facts) -> Optional[float]:
    """Area in hectares if known, else None -- never a substituted default."""
    if facts.has("changed_area_ha"):
        return num(facts.get("changed_area_ha"))
    return None


def _scale_phrase(facts: Facts) -> str:
    ha = _area_from(facts)
    if ha is not None:
        return human_area(ha)
    if facts.has("changed_fraction"):
        return (f"about {_pct(facts.get('changed_fraction'))} of the area you "
                f"asked about")
    return "part of the area you asked about"


# --------------------------------------------------------------------------- #
# Refusals -- evaluated first, in order. Each ends the evaluation.
# --------------------------------------------------------------------------- #

def _render_nothing_found(f: Facts) -> Dict[str, Any]:
    return {
        "headline": "We could not find anything to base a decision on.",
        "why": ("The analysis did not produce a measurable change result. That "
                "usually means the two images could not be compared — most often "
                "they cover different ground, or one of them is not a geotagged "
                "image of the same place."),
        "action": ("Check that both images are of the same location, then run the "
                   "comparison again."),
        "who": "Whoever prepared the images",
        "confidence_note": "No measurement was produced, so there is nothing to "
                           "trust or distrust here.",
    }


def _render_low_trust(f: Facts) -> Dict[str, Any]:
    trust = num(f.get("trust"), 0.0)
    band = f.get("trust_band") or "low"
    source = f.get("trust_source") or ""
    via = ("This figure comes from how often this method has actually been right "
           "on held-out data." if source == "measured_band" else
           "This figure combines the tool's own confidence with how reliable "
           "that method has proven to be.")
    if trust < TRUST_FLOOR:
        head = "We cannot stand behind this result."
        why = (f"{via} On that basis the answer is not dependable enough to act "
               f"on, and telling you otherwise would be worse than telling you "
               f"nothing.")
    else:
        head = "Treat this as a lead, not a finding."
        why = (f"{via} It is suggestive but not strong enough to commit money, "
               f"people or equipment to on its own.")
    return {
        "headline": head,
        "why": why,
        "action": ("Check it against something you already know — a field visit, "
                   "a local report, or your own records for the area — before "
                   "acting on it."),
        "who": "Someone who can visit or confirm locally",
        "confidence_note": f"Reliability rating: {band.replace('_', ' ')}.",
    }


def _render_misrouted(f: Facts) -> Dict[str, Any]:
    return {
        "headline": "We are not certain we answered the question you asked.",
        "why": ("Your question could reasonably have meant more than one kind of "
                "analysis, and the one we chose may not be the one you wanted."),
        "action": ("Ask again naming what you want to know — for example "
                   "\"what changed between these two dates\" or \"is there "
                   "building where there was none\"."),
        "who": "You",
        "confidence_note": "This is about the question, not the analysis.",
    }


def _render_below_floor(f: Facts) -> Dict[str, Any]:
    gsd = f.get("gsd_m")
    floor = detection_floor_ha(gsd)
    floor_txt = (f"{floor:.2f} hectares" if floor else "the smallest area these "
                                                       "images can resolve")
    return {
        "headline": "We found nothing large enough to be sure about.",
        "why": (f"Any change we did pick up is smaller than {floor_txt}, which is "
                f"the smallest change these particular images can show reliably. "
                f"At that size, real change and image noise look the same."),
        "action": ("Nothing to act on from these images. If you expect a change "
                   "here, request imagery taken closer to the ground — the rule "
                   "of thumb is that you can trust about a quarter of a hectare "
                   "for every 10 metres of pixel size."),
        "who": "Anyone considering a follow-up image request",
        "confidence_note": "This is a limit of the imagery, not a doubt about the "
                           "analysis.",
    }


# --------------------------------------------------------------------------- #
# Verdicts -- evaluated in order after the refusals pass.
# --------------------------------------------------------------------------- #

def _render_flood_exposure(f: Facts) -> Dict[str, Any]:
    nw = num(f.get("near_water_500m"))
    return {
        "headline": ("Land has changed right beside water, which matters for "
                     "flooding."),
        "why": (f"{_scale_phrase(f).capitalize()} has changed, and {_pct(nw)} of "
                f"that change is within 500 metres of a water body. Change this "
                f"close to water affects how the ground drains, and can mean "
                f"water has spread onto land that was previously dry, or that "
                f"ground beside a channel has been built over or cut into."),
        "action": ("Compare this against your flood records for the same stretch "
                   "of water, and treat the affected stretch as higher risk for "
                   "the coming season. If the change is recent, a ground check "
                   "along the water line is worth more than another satellite "
                   "image."),
        "who": "Local water or irrigation office; district flood or disaster "
               "management cell",
        "confidence_note": "The change itself is measured; the link to flooding "
                           "is proximity, which is strong evidence but not proof.",
    }


def _render_encroachment(f: Facts) -> Dict[str, Any]:
    new_ha = num(f.get("built_up_new_ha"), 0.0)
    nw = f.get("near_water_500m")
    extra = (f" A share of it is near water ({_pct(nw)}), which can also affect "
             f"drainage." if nw and num(nw) > 0.2 else "")
    return {
        "headline": "New construction has appeared where there was none before.",
        "why": (f"About {new_ha:.2f} hectares of built-up ground is new between "
                f"the two dates — that is ground that was vegetation, bare soil "
                f"or water in the earlier image and now reads as built surface. "
                f"This is the pattern that unauthorised construction produces, "
                f"but it is also what permitted building and road work looks "
                f"like.{extra}"),
        "action": ("Put this against your land records for the same boundary "
                   "before acting. If the parcel has no approval on file, pass it "
                   "to the revenue or land-records office with these two dates "
                   "and the highlighted area — that is enough for them to open a "
                   "case."),
        "who": "Revenue or land-records office; village-level land management "
               "committee",
        "confidence_note": "The new built-up area is measured. Whether it is "
                           "authorised is not something an image can tell you.",
    }


def _render_vegetation_loss(f: Facts) -> Dict[str, Any]:
    lost = num(f.get("vegetation_lost_ha"), 0.0)
    return {
        "headline": "Vegetation has been lost in this area.",
        "why": (f"About {lost:.2f} hectares that was green at the earlier date is "
                f"no longer green. At the end of a harvest season this is "
                f"completely normal and needs no attention. Outside a harvest "
                f"window it usually means clearing, cutting or burning."),
        "action": ("Check the date against your cropping calendar first. If the "
                   "two image dates do not sit at the end of a harvest, go and "
                   "look — and if the land is yours, note it before the next "
                   "season."),
        "who": "Landowner or farmer; agriculture extension officer",
        "confidence_note": "Loss of green cover is measured; its cause is not.",
    }


def _render_modest_change(f: Facts) -> Dict[str, Any]:
    return {
        "headline": "Something changed, but nothing that needs urgent action.",
        "why": (f"{_scale_phrase(f).capitalize()} has changed between the two "
                f"dates, which is real but modest. There is no new construction "
                f"or vegetation loss large enough to act on from these images "
                f"alone."),
        "action": ("Note the date and check the same area again after the next "
                   "season. If the change keeps growing, that is worth acting on."),
        "who": "You, or the office responsible for the area",
        "confidence_note": "A modest measured change — worth watching, not worth "
                           "spending on yet.",
    }


def _render_no_action(f: Facts) -> Dict[str, Any]:
    return {
        "headline": "Nothing here needs action.",
        "why": ("The comparison found no change big enough to matter between the "
                "two dates. The land was effectively the same on both days for "
                "anything this imagery can see."),
        "action": ("No follow-up needed. Keep the two dates on file — if you are "
                   "asked whether this area changed in this period, this is the "
                   "evidence."),
        "who": "You",
        "confidence_note": "An absence of findings, based on a measured "
                           "comparison.",
    }


RULES: List[Rule] = [
    # -- refusals (order is significant) ------------------------------------ #
    Rule("R0_no_measurement", "refusal", "insufficient_evidence",
         fires_when_absent=("changed_area_ha", "changed_fraction"),
         blocker=True, render=_render_nothing_found, domain="general",
         description="No measurable change result at all.",
         missing_reason="neither a changed area nor a changed fraction was produced"),
    Rule("R1_routing_uncertain", "refusal", "verify_first",
         requires=("intent_confidence",), blocker=True, render=_render_misrouted,
         when=lambda f: num(f.get("intent_confidence"), 1.0)
         < ROUTING_CONFIDENCE_FLOOR,
         domain="general",
         description="The question could have meant more than one analysis."),
    Rule("R2_unmeasured_method", "refusal", "verify_first",
         requires=("method_unmeasured",), blocker=True, render=_render_low_trust,
         domain="general",
         description="The method behind this number has no measured reliability."),
    Rule("R3_trust_too_low", "refusal", "verify_first",
         requires=("trust",), blocker=True, render=_render_low_trust,
         when=lambda f: num(f.get("trust"), 0.0) < TRUST_ACTION_FLOOR,
         domain="general",
         description="Trust below the floor for acting on the result."),
    Rule("R4_below_detection_floor", "refusal", "no_action",
         requires=("gsd_m", "changed_area_ha"), blocker=True,
         # Area-based only: a fraction cannot be compared against an area floor
         # without knowing the scene size, and inventing one would undo the point.
         # Also requires *some* change: genuinely nothing changed is a different
         # answer ("no action needed") from "something too small to trust".
         when=lambda f: 0.0 < num(f.get("changed_area_ha"), 0.0)
         < (detection_floor_ha(f.get("gsd_m")) or 0.0),
         render=_render_below_floor, domain="general",
         description="Any change present is below what this resolution resolves."),

    # -- verdicts ----------------------------------------------------------- #
    # Order is priority: the first applicable verdict is the headline, and any
    # others that also hold are reported as supporting conclusions. Construction
    # on land that was not built before leads because it is the most concrete,
    # actionable finding; being near water is context that sharpens it.
    Rule("V2_encroachment", "verdict", "act",
         requires=("built_up_new_ha",),
         when=lambda f: num(f.get("built_up_new_ha"), 0.0)
         >= SIGNIFICANT_AREA_HA,
         render=_render_encroachment, domain="encroachment",
         description="New built-up ground on land that was not built before."),
    Rule("V1_flood_exposure", "verdict", "act",
         requires=("near_water_500m",), requires_any=(("changed_fraction",),
                                                      ("changed_area_ha",)),
         when=lambda f: (num(f.get("near_water_500m"))
                         >= WATER_PROXIMITY_STRONG
                         and (num(f.get("changed_fraction"), 0.0)
                              >= SIGNIFICANT_FRACTION
                              or num(f.get("changed_area_ha"), 0.0)
                              >= SIGNIFICANT_AREA_HA)),
         render=_render_flood_exposure, domain="flood",
         description="Significant change concentrated near water."),
    Rule("V3_vegetation_loss", "verdict", "monitor",
         requires=("vegetation_lost_ha",),
         when=lambda f: num(f.get("vegetation_lost_ha"), 0.0)
         >= SIGNIFICANT_AREA_HA,
         render=_render_vegetation_loss, domain="agriculture",
         description="Vegetation cover lost between the two dates."),
    Rule("V4_modest_change", "verdict", "monitor",
         requires_any=(("changed_area_ha",), ("changed_fraction",)),
         when=lambda f: (num(f.get("changed_fraction"), 0.0) >= MINOR_FRACTION
                         or num(f.get("changed_area_ha"), 0.0) >= 0.10),
         fallback=True,
         render=_render_modest_change, domain="general",
         description="A real but modest change, with no specific finding."),
    Rule("V5_no_action", "verdict", "no_action",
         requires_any=(("changed_area_ha",), ("changed_fraction",)),
         when=lambda f: (num(f.get("changed_fraction"), 0.0) < MINOR_FRACTION
                         and num(f.get("changed_area_ha"), 0.0) < 0.10),
         fallback=True,
         render=_render_no_action, domain="general",
         description="No change worth acting on."),
]


def refusal_rules() -> List[Rule]:
    return [r for r in RULES if r.kind == "refusal"]


def verdict_rules() -> List[Rule]:
    """Verdicts a user would recognise as a specific finding (non-fallback)."""
    return [r for r in RULES if r.kind == "verdict" and not r.fallback]


def fallback_rules() -> List[Rule]:
    """Generic tiers consulted only when no specific verdict applied."""
    return [r for r in RULES if r.kind == "verdict" and r.fallback]
