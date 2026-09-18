"""Facts layer -- turn specialist outputs into the facts a decision needs.

The failure this module exists to prevent
-----------------------------------------
An earlier design keyed its rules on ``changed_area_ha``, a key only
``impact_analysis`` emits. On a ``change_analysis`` run every predicate read a
missing key, the rules fell through to their fallback, and the layer reported
``no_action`` -- on a run where area had genuinely changed. Absence of a fact is
not evidence of no change.

So every fact here carries its provenance, and a rule that requires a fact which
is neither reported nor derivable produces ``insufficient_evidence`` rather than
a verdict. ``transitions`` is the sharpest case: ``impact_analysis`` emits
hectares per class while ``change/transitions.py`` emits a fraction-plus-table
shape. They are not interchangeable, so the fact layer resolves which one it is
holding and says so, instead of reading the same key name and hoping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Facts the rules reason over, in the vocabulary the engine uses.
FACT_KEYS = (
    "changed_area_ha", "changed_fraction", "built_up_new_ha", "built_up_lost_ha",
    "vegetation_lost_ha", "vegetation_gained_ha", "near_water_250m",
    # How the new built-up ground is arranged, not merely how much of it there
    # is. Present only for runs whose impact analysis measured it; absent
    # elsewhere, where the rules that need it simply do not apply.
    "built_up_new_regions", "built_up_new_largest_ha",
    "built_up_new_largest_share",
    "near_water_500m", "near_water_1000m", "n_change_regions", "largest_region_box",
    "priority_zone", "gsd_m", "gsd_assumed", "confidence", "trust", "trust_band",
    "trust_source", "calibrated", "dominant_direction", "modality_agreement",
    "transitions_shape", "has_prob_map",
    # Official reference map (ISRO/Bhuvan), present only for runs whose imagery
    # came through the online acquisition lane. Absent is the normal case for an
    # uploaded pair, and the decision then reads exactly as it did before.
    "isro_context_present", "isro_context_layers", "isro_layers", "isro_themes",
    "isro_open_land_themes", "isro_built_themes", "isro_water_themes",
)

# How much area is one pixel? Used only to phrase advice in everyday terms.
HECTARES_PER_FOOTBALL_FIELD = 0.714
HECTARES_PER_ACRE = 0.4047


@dataclass
class Facts:
    """Assembled facts, each with where it came from."""

    values: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)   # fact -> provenance
    notes: List[str] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def has(self, *keys: str) -> bool:
        """True only if every named fact is present *and* not None."""
        return all(self.values.get(k) is not None for k in keys)

    def missing(self, *keys: str) -> List[str]:
        return [k for k in keys if self.values.get(k) is None]

    def _set(self, key: str, value: Any, source: str) -> None:
        if value is None:
            if key not in self.absent:
                self.absent.append(key)
            return
        self.values[key] = value
        self.sources[key] = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "provenance": self.sources,
            "absent": self.absent,
            "notes": self.notes,
        }


def num(value: Any, default: float = 0.0) -> float:
    """Coerce a fact to a float, tolerating whatever came through the pipeline.

    Facts arrive from model outputs and GIS maths. A malformed value must make a
    rule inapplicable, not raise: refusing to decide is an acceptable answer for
    this layer, crashing is not. NaN is treated as absent for the same reason.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _first(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def transitions_is_hectare_shape(transitions: Any) -> Optional[bool]:
    """Which ``transitions`` shape are we holding?

    True  -> ``{built_up_new_ha, vegetation_lost_ha, ...}`` (per-class hectares)
    False -> ``{changed_fraction, date_a, date_b, top[]}`` (a summary table)
    None  -> absent
    """
    if not isinstance(transitions, dict) or not transitions:
        return None
    if any(k.endswith("_ha") for k in transitions):
        return True
    if "changed_fraction" in transitions or "top" in transitions:
        return False
    return None


def resolve_run_outputs(outputs: Optional[Dict[str, Any]]
                        ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]],
                                   Optional[Dict[str, Any]]]:
    """``(primary, impact, change)`` for a finished run, whatever envelope it used.

    Three shapes reach this layer: a flat tool output, a run keyed
    ``impact_analysis``/``change_analysis``, and the ``investigation`` envelope
    that ``AgentController._run_investigation`` builds -- where the impact result
    sits at ``investigation.impact`` and the change result at
    ``investigation.change``.

    Miss that third shape and the facts are all absent, which is not a neutral
    failure: ``R0_no_measurement`` fires and the flagship "full analysis" path
    tells a user "we could not find anything to base a decision on" while the run
    is holding hectares, water proximity and ranked zones. Resolving envelopes in
    one place is what keeps every reader -- facts and the trust gate -- honest.
    """
    out = outputs or {}
    impact: Optional[Dict[str, Any]] = None
    change: Optional[Dict[str, Any]] = None
    if isinstance(out.get("impact"), dict):
        impact = out["impact"]
    if impact is None and isinstance(out.get("impact_analysis"), dict):
        impact = out["impact_analysis"]
    if isinstance(out.get("change_analysis"), dict):
        change = out["change_analysis"]
    inv = out.get("investigation")
    if isinstance(inv, dict):
        if impact is None and isinstance(inv.get("impact"), dict):
            impact = inv["impact"]
        if change is None and isinstance(inv.get("change"), dict):
            change = inv["change"]
    return (impact or change or out), impact, change


def assemble(outputs: Dict[str, Any], task: str = "",
             visuals: Optional[Dict[str, Any]] = None,
             images: Optional[List[Any]] = None) -> Facts:
    """Build the canonical fact set from whatever the run actually produced.

    Deliberately tolerant about *shape* and strict about *claims*: it will look
    in several places for a fact, but if it cannot find or derive one it records
    it as absent rather than substituting a default.
    """
    out = outputs or {}
    vis = visuals or {}
    facts = Facts()

    src, impact, change = resolve_run_outputs(out)

    # ---- area ------------------------------------------------------------ #
    ha = _first(src, "changed_area_ha")
    if ha is None and impact:
        ha = _first(impact, "changed_area_ha")
    facts._set("changed_area_ha", ha, "reported by impact_analysis" if ha is not None
               else "")
    frac = _first(src, "changed_fraction", "changed_area_fraction")
    if frac is None and isinstance(change, dict):
        frac = _first(change, "changed_area_fraction")
    # ``changed_area_fraction`` is a documented alias in change_analysis.
    facts._set("changed_fraction", frac,
               "reported" if frac is not None else "")

    # ---- transitions ----------------------------------------------------- #
    trans = _first(src, "transitions")
    shape = transitions_is_hectare_shape(trans)
    if shape is not None:
        facts._set("transitions_shape",
                   "per-class hectares" if shape else "summary table",
                   "inspected")
    if shape:
        for k in ("built_up_new_ha", "built_up_lost_ha",
                  "vegetation_lost_ha", "vegetation_gained_ha",
                  "built_up_new_regions", "built_up_new_largest_ha",
                  "built_up_new_largest_share"):
            facts._set(k, trans.get(k), "reported in transitions")
    elif trans:
        facts.notes.append(
            "transitions arrived in the summary-table shape, which does not carry "
            "per-class hectares, so per-class rules cannot be applied to this run")

    # ---- proximity to water --------------------------------------------- #
    nw = _first(src, "near_water") or {}
    if isinstance(nw, dict):
        for key, fact in (("within_250m", "near_water_250m"),
                          ("within_500m", "near_water_500m"),
                          ("within_1000m", "near_water_1000m")):
            facts._set(fact, nw.get(key), "reported by impact_analysis")

    # ---- spatial detail -------------------------------------------------- #
    facts._set("n_change_regions", _first(src, "num_change_regions"),
               "reported")
    box = _first(src, "largest_region_box")
    if box is not None:
        facts._set("largest_region_box", list(box) if hasattr(box, "__iter__") else box,
                   "reported")
    facts._set("priority_zone", _first(src, "priority_zone") or None, "reported")
    facts._set("dominant_direction", _first(src, "dominant_direction") or None,
               "reported")

    # ---- acquisition quality -------------------------------------------- #
    facts._set("gsd_m", _first(src, "gsd_m"), "reported by impact_analysis")
    if "gsd_assumed" in (src or {}):
        facts._set("gsd_assumed", bool(src.get("gsd_assumed")),
                   "reported by impact_analysis")
    elif images:
        assumed = [getattr(im, "transform_bounds", None) is None for im in images]
        facts._set("gsd_assumed", any(assumed),
                   "derived: georeferencing missing on at least one input")

    # ---- confidence and trust ------------------------------------------- #
    meta = out.get("confidence_meta") or (src or {}).get("confidence_meta") or {}
    conf = _first(src, "confidence")
    facts._set("confidence", conf, "reported")
    facts._set("trust", meta.get("trust"), "reported by the confidence layer")
    facts._set("trust_band", meta.get("trust_band"), "reported by the confidence layer")
    facts._set("trust_source", meta.get("trust_source"), "reported by the confidence layer")
    if "calibrated" in meta:
        facts._set("calibrated", bool(meta.get("calibrated")), "reported by the confidence layer")
    if meta.get("band_evidence"):
        facts._set("band_evidence", meta["band_evidence"], "measured on held-out data")

    # ---- evidence availability ------------------------------------------ #
    facts._set("modality_agreement", _first(src, "modality_agreement")
               or _first(out, "modality"), "reported")
    facts._set("has_prob_map", bool((vis.get("prob_map") is not None)
                                    or ((src or {}).get("_visual", {}) or {}).get("prob_map")
                                    is not None),
               "inspected: whether a raw probability field is available")

    # ---- official reference map (ISRO/Bhuvan) ---------------------------- #
    # Read from the run envelope, not from a tool output: the acquisition lane
    # keeps this in the run's provenance, and the server attaches it to the run
    # for imagery that came through the online lane. Verification (that the layer
    # really has data in this window) already happened there; this layer only
    # decides what the record means for the decision.
    from .authority import facts_for as _isro_facts
    for key, value in _isro_facts(out.get("isro_context")).items():
        facts._set(key, value, "ISRO context verified by the acquisition lane")

    # Anything declared but never found is recorded, so the engine can say what
    # it could not establish rather than quietly deciding on partial input.
    for key in FACT_KEYS:
        if key not in facts.values and key not in facts.absent:
            facts.absent.append(key)
    facts.absent.sort()
    return facts


def human_area(hectares: float) -> str:
    """Area in everyday units a non-expert can picture."""
    if hectares is None:
        return "an unknown area"
    if hectares < 0.5:
        acres = hectares / HECTARES_PER_ACRE
        return f"about {acres:.1f} acres"
    fields = hectares / HECTARES_PER_FOOTBALL_FIELD
    return (f"about {hectares:.1f} hectares "
            f"(roughly {fields:.0f} football fields)")
