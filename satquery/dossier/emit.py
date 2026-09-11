"""C2 — dossier V2 renderer (additive over report.json; never rewrites it).

Maps a finished AgentResult into the head -> key_facts -> before/after ->
timeline -> hypotheses -> sources -> trace appendix schema the judges
evaluate (R1 frozen contract). Ricles come from grounding/change boxes with
confirmed/uncertain/refuted status.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class DossierDoc:
    run_id: str
    head: dict = field(default_factory=dict)
    key_facts: List[str] = field(default_factory=list)
    before_after: dict | None = None
    timeline: List[dict] = field(default_factory=list)
    hypotheses: List[dict] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    trace: List[dict] = field(default_factory=list)
    reticles: List[dict] = field(default_factory=list)   # {x_pct,y_pct,verdict,...}
    freshness: dict | None = None
    aoi: dict = field(default_factory=lambda: {"kind": "pixel-space",
                                                "bbox": None})

    def to_dict(self) -> dict:
        return asdict(self)


def _box_to_reticle(box, img, verdict: str) -> dict:
    x0, y0, x1, y1 = box
    x_pct = round((x0 + x1) / 2.0 / max(getattr(img, "width", 1), 1) * 100, 2)
    y_pct = round((y0 + y1) / 2.0 / max(getattr(img, "height", 1), 1) * 100, 2)
    from ..geodate import quadrant_of
    return {"box": list(box), "x_pct": x_pct, "y_pct": y_pct,
            "verdict": verdict, "quadrant": quadrant_of(box,
                                                        img.width, img.height)}


def emit(result, imgs) -> DossierDoc:
    """Build the dossier for a finished result (pure function of result)."""
    run_id = getattr(result, "run_id", "")
    outputs: dict = getattr(result, "outputs", {}) or {}
    trace: list = getattr(result, "trace", []) or []

    doc = DossierDoc(run_id=run_id)

    # head <- first answer sentence + selected task + configuration
    answer = str(outputs.get("answer") or "").strip()
    doc.head = {
        "answer": answer,
        "answer_raw": str(outputs.get("answer_raw") or answer).strip(),
        "selected_task": getattr(result, "selected_task", ""),
        "configuration": getattr(result, "configuration", {}) or {}}

    # key_facts <- answer / confidence / area / class
    facts = []
    if answer:
        facts.append(answer)
    conf = outputs.get("confidence")
    if conf is not None:
        meta = outputs.get("confidence_meta") or {}
        facts.append(f"Confidence {float(conf):.1%} (method: "
                     f"{meta.get('method', 'formula')}).")
    for label in (outputs.get("labels") or [])[:3]:
        if isinstance(label, (list, tuple)) and len(label) == 2:
            facts.append(f"Detected: {label[0]} ({float(label[1]):.1%})")
    doc.key_facts = facts[:4]

    # before/after for bitemporal change tasks
    changed_ha = outputs.get("changed_area_ha")
    if changed_ha is not None:
        doc.before_after = {
            "changed_area_ha": changed_ha,
            "transitions": (outputs.get("transitions") or {}).get("top", [])}

    # timeline <- trace steps with duration
    doc.timeline = [{"name": s.get("name", ""), "status": s.get("status", "ok"),
                     "duration_ms": s.get("duration_ms")}
                    for s in trace]

    # hypotheses: grounding/change boxes -> reticles
    ranked = outputs.get("ranked")
    boxes = []
    if ranked:
        primary = getattr(ranked, "primary", None)
        if primary and getattr(primary, "box", None):
            doc.reticles.append(_box_to_reticle(primary.box, imgs[0],
                                                "confirmed"))
            boxes.append(primary.box)
        for alt in getattr(ranked, "alternates", []) or []:
            if getattr(alt, "box", None):
                doc.reticles.append(_box_to_reticle(alt.box, imgs[0],
                                                    "uncertain"))
                boxes.append(alt.box)
    for box in outputs.get("boxes") or []:
        if box not in boxes:
            doc.reticles.append(_box_to_reticle(box, imgs[0], "confirmed"))

    if doc.reticles:
        doc.hypotheses.append({"summary": f"{len(doc.reticles)} region(s) "
                               f"localised", "reticles": len(doc.reticles)})

    # sources <- inputs + models
    for im in imgs or []:
        doc.sources.append({"file": getattr(getattr(im, "path", None), "name", ""),
                            "modality": getattr(im, "modality", ""),
                            "acquired": getattr(im, "acquired", "") or None,
                            "crs": getattr(im, "crs", None)})
    doc.sources.append({"model": outputs.get("source_model", "SatQuery AI"),
                        "source": "agentic EO specialist orchestration"})

    doc.trace = trace

    # aoi: geo if any box projects to lon/lat, else honest pixel-space
    if doc.reticles and imgs:
        try:
            from ..geodate import pixel_to_lonlat
            gb = pixel_to_lonlat(imgs[0], doc.reticles[0]["box"])
            if gb.lonlat_polygon:
                doc.aoi = {"kind": "geo", "bbox": gb.to_dict()}
        except Exception:
            pass

    # freshness clock (C3)
    try:
        from ..freshness import clocks_for
        doc.freshness = clocks_for(result, imgs).to_dict()
    except Exception:
        pass

    return doc
