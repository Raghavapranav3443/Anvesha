"""The authority lane inside the decision layer: what ISRO's own map says.

The acquisition layer already answers *"where do I get the data?"*, and it keeps
its provenance on disk next to the imagery it fetched. This module is the other
half: it takes the ISRO context that was **verified present** for an area and
turns it into (a) facts the decision rules can reason over and (b) plain English
a non-expert can act on.

Why this is the difference between an observation and advice
-----------------------------------------------------------
"Our analysis found new construction" is an observation. "Our analysis found new
construction, and ISRO's own land-use map records this parcel as agriculture" is
something a revenue officer can open a file on. The second sentence is only
possible because the change detection and the Government of India's own map are
both in the room, and it is the reason the two lanes exist.

Four honesty rules, each of which a test pins
---------------------------------------------
1. **Only verified layers count.** A layer is used only when its rendered
   evidence cleared the acquisition lane's own ``PRESENCE_MARGIN`` above a
   control window. Bhuvan answers HTTP 200 with a valid PNG for a layer with no
   data in the area, so counting rendered tiles would manufacture official
   corroboration out of blank images. That threshold is imported, not re-guessed,
   so the two layers cannot drift apart.
2. **Presence is not agreement.** A map recording open land under new
   construction is *consistent with* the construction being off-record. It is
   not proof of anything, and it never turns a weak detection into a confident
   one -- it sharpens who to tell, not how sure we are.
3. **Only one thing overrides our own finding, and only downward.** Building on
   ground the authority maps as water is a genuine conflict between two
   credible sources, so that case stops at "check before acting". Everything
   else adds context to the verdict rather than replacing it.
4. **Nothing is invented.** No layer name, theme or state is guessed here. An
   area with no verified context produces no authority block at all, and the
   decision reads exactly as it did before this module existed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Theme -> what the record actually means for a parcel. These read the *theme*
# strings the acquisition lane stores, not layer names, which are unpredictable
# (`nuis:AS_DI_*`, `basemap:AN_LULC`, ...).
OPEN_LAND_THEMES = ("agriculture", "cropland", "forest", "wasteland", "wetland")
BUILT_THEMES = ("builtup", "builtup_urban", "builtup_rural")
# Themes where a new building is a conflict between two credible sources rather
# than a corroboration: water is mapped there, and construction is not ambiguous
# about land the state says is a water body.
WATER_SENSITIVE_THEMES = ("waterbody", "river", "wetland")

# How much new built-up ground makes building on mapped water a *conflict* worth
# stopping for, rather than a single structure worth a look. Kept equal to the
# decision layer's significance floor on purpose, and a test asserts they match so
# the two cannot drift apart into contradictory answers.
MIN_CONFLICT_AREA_HA = 0.5

_FALLBACK_PRESENCE_MARGIN = 0.02


def _presence_margin() -> float:
    """The acquisition lane's own presence threshold, reused rather than re-guessed."""
    try:
        from ..acquire.providers.bhuvan import PRESENCE_MARGIN

        return float(PRESENCE_MARGIN)
    except Exception:                      # acquisition layer absent or broken
        return _FALLBACK_PRESENCE_MARGIN


@dataclass(frozen=True)
class VerifiedLayer:
    """One thematic layer, with the evidence that it is really there."""

    name: str
    theme: str
    theme_label: str
    state: str
    evidence: float
    coverage: float
    control: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.name,
            "theme": self.theme,
            "theme_label": self.theme_label,
            "state": self.state,
            "evidence": round(self.evidence, 4),
            "coverage": round(self.coverage, 4),
            "control_coverage": round(self.control, 4),
        }


def _raw_layers(raw: Any) -> List[Dict[str, Any]]:
    """Accept the shapes the acquisition lane actually writes, and nothing else."""
    if isinstance(raw, dict):
        inner = raw.get("layers") or raw.get("context") or raw.get("isro_context")
        raw = inner
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, dict)]


def verified_layers(raw: Any) -> List[VerifiedLayer]:
    """Layers whose own evidence clears the presence margin. Order preserved.

    The evidence field is written by the acquisition lane from
    ``coverage - control_coverage``. It is recomputed here for data that only
    carries the two coverage numbers, so a caller cannot accidentally launder a
    blank render into a present layer by omitting ``evidence``.
    """
    margin = _presence_margin()
    out: List[VerifiedLayer] = []
    for item in _raw_layers(raw):
        coverage = _as_float(item.get("coverage"))
        control = _as_float(item.get("control_coverage"))
        evidence = _as_float(item.get("evidence"))
        if evidence is None:
            evidence = max(0.0, (coverage or 0.0) - (control or 0.0))
        theme = str(item.get("theme") or "").strip()
        if evidence < margin:
            continue
        out.append(VerifiedLayer(
            name=str(item.get("layer") or item.get("name") or "").strip(),
            theme=theme,
            theme_label=str(item.get("theme_label") or theme or "layer").strip(),
            state=str(item.get("state") or "").strip(),
            evidence=evidence,
            coverage=coverage or 0.0,
            control=control or 0.0,
        ))
    return out


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out                 # NaN reads as absent


def _themes_in(layers: Sequence[VerifiedLayer], wanted: Iterable[str]) -> List[str]:
    """Plain-English labels, deduplicated, in the order they were asked for."""
    wants = set(wanted)
    labels: List[str] = []
    for theme in wanted:
        for layer in layers:
            if layer.theme in wants and layer.theme == theme:
                if layer.theme_label not in labels:
                    labels.append(layer.theme_label)
    # Any theme that matches the family but is not in the declared order (the
    # catalogue adds theme names over time) is still reported, never dropped.
    for layer in layers:
        if layer.theme in wants and layer.theme_label not in labels:
            labels.append(layer.theme_label)
    return labels


def facts_for(raw: Any) -> Dict[str, Any]:
    """The fact values this context contributes. Empty dict when nothing verified."""
    layers = verified_layers(raw)
    if not layers:
        return {}
    return {
        "isro_context_present": True,
        "isro_context_layers": len(layers),
        "isro_layers": [layer.to_dict() for layer in layers],
        "isro_themes": [layer.theme_label for layer in layers],
        "isro_open_land_themes": _themes_in(layers, OPEN_LAND_THEMES),
        "isro_built_themes": _themes_in(layers, BUILT_THEMES),
        "isro_water_themes": _themes_in(layers, WATER_SENSITIVE_THEMES),
    }


def block(raw: Any, facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``authority`` record attached to a decision, plus its sentences.

    ``facts`` is the assembled fact dict, used only to decide which sentences are
    relevant to *this* run -- an agriculture layer is worth mentioning next to a
    construction finding and noise next to a question about water.
    """
    layers = verified_layers(raw)
    if not layers:
        return {"available": False, "layers": [], "themes": [], "sentences": []}
    values = facts or {}
    built = values.get("isro_built_themes") or []
    open_land = values.get("isro_open_land_themes") or []
    water = values.get("isro_water_themes") or []
    record: Dict[str, Any] = {
        "available": True,
        "layers": [layer.to_dict() for layer in layers],
        "themes": [layer.theme_label for layer in layers],
        "evidence_margin": _presence_margin(),
        "sentences": [],
        "note": ("ISRO thematic layers, confirmed present in this window against a "
                 "control render. Presence is evidence of what the map records, "
                 "not of agreement with this analysis."),
    }
    record["sentences"] = _sentences(layers, open_land, built, water)
    return record


def _share(layer: VerifiedLayer) -> str:
    """Share of the analysed window this layer covers. Never a bare 0%."""
    pct = int(round(layer.coverage * 100))
    return f"{max(1, pct)}%" if layer.coverage > 0 else "under 1%"


def _sentences(layers: Sequence[VerifiedLayer], open_land: List[str],
               built: List[str], water: List[str]) -> List[str]:
    """Plain English for a reader who has never seen a map overlay.

    Written as one factual summary plus at most two readings of it. The earlier
    version emitted one sentence per layer, which on a real 12 km window read as
    self-contradiction -- "the record shows agriculture, not built-up ground"
    directly above "the record already shows built-up". Both were true of
    *different parts* of the window, and a reader has no way to know that. A
    window almost always mixes themes, so the mixed case is stated as the mixed
    case rather than as two confident opposites.
    """
    shares = ", ".join(f"{layer.theme_label} {_share(layer)}" for layer in layers)
    n = len(layers)
    out = [f"ISRO's own map records this same window as: {shares}."]

    if open_land and built:
        out.append(
            "The window contains both open ground and ground already mapped as "
            "built-up, so the official record alone cannot say whether any one "
            "new building sits on open land or inside an area already "
            "recognised as built up. Check the parcel against your land records.")
    elif built:
        out.append(
            "This window is already mapped as built-up ground, so change here is "
            "development on land the official map already recognises as built up.")
    elif open_land:
        out.append(
            f"ISRO's own record for this window shows {', '.join(open_land)}, not "
            f"built-up ground, so new construction is not something the official "
            f"map shows.")
    if water:
        joined = ", ".join(water)
        if len(water) == 1:
            out.append(
                f"{joined.capitalize()} is mapped inside this window, so new "
                f"building on or beside it matters for drainage and flood "
                f"exposure.")
        else:
            out.append(
                f"Water and wetland are mapped inside this window ({joined}), so "
                f"new building near those features matters for drainage and flood "
                f"exposure.")
    if n:
        out.append("You can check that yourself on ISRO's Bhuvan portal.")
    return out


def water_conflict(facts: Dict[str, Any]) -> bool:
    """True when new building is reported on ground the authority maps as water.

    Deliberately narrow: it needs *both* a meaningful amount of new built-up
    ground and a verified water-family layer. A single small structure on a
    mapped wetland is worth a look but not worth overriding a verdict for.
    """
    from .facts import num                          # local import: no cycle

    themes = facts.get("isro_water_themes") or []
    if not themes:
        return False
    return num(facts.get("built_up_new_ha"), 0.0) >= MIN_CONFLICT_AREA_HA
