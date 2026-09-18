"""Concrete tool implementations invoked by the agent controller.

Every tool receives a uniform context:
    ctx = {"images": [RSImage, ...], "query": str, "config": dict, "params": dict}
and returns a JSON-serialisable result dict (visual evidence arrays are kept
under the '_visual' key and stripped from reports).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .io_utils import rgb_composite

# --- tunable thresholds -------------------------------------------------- #
# Minimum probability from the change detector's prob_map for a pixel to be
# considered "changed" when building the impact-analysis mask.
CHANGE_MASK_PROB_THRESHOLD = 0.85
# Minimum fraction of changed pixels lying within 500 m of water before the
# "new built-up near water" mention is surfaced in the impact answer.
NEAR_WATER_MENTION_THRESHOLD = 0.05


def single_vqa_tool(ctx: Dict) -> Dict:
    from .models import get_vqa_model
    img = ctx["images"][0]
    question = ctx["query"]
    out = get_vqa_model().answer(img, question)
    return {
        "answer": out["answer"],
        "confidence": out.get("confidence", 0.0),
        "candidates": out.get("candidates"),
        "evidence": out.get("evidence"),
        "source_model": out.get("source", "RS-VQA specialist"),
    }


def caption_tool(ctx: Dict) -> Dict:
    from .models import describe
    img = ctx["images"][0]
    out = describe(img, query_hint=ctx.get("query", ""))
    return {
        "caption": out["caption"],
        "confidence": out["confidence"],
        "labels": out["labels"],
        "layout": out["layout"],
        "generative": out.get("generative", False),
        "source_model": out["source"],
    }


def grounding_tool(ctx: Dict) -> Dict:
    from .models import ground
    from .io_utils import rgb_composite
    img = ctx["images"][0]
    forced = (ctx.get("_forced_concept")
              or ctx.get("params", {}).get("concept"))
    res = ground(img, forced) if forced else \
        ground(img, ctx.get("query") or "water body")
    vis = _overlay_mask(rgb_composite(img), res.mask, color=(0.9, 0.2, 0.1))
    for i, box in enumerate(res.boxes[:3]):
        _draw_box(vis, box)
    return {
        "answer": (f"Grounded {len(res.boxes)} '{res.concept}' region(s) covering "
                   f"{res.area_fraction * 100:.1f}% of the scene."),
        "concept": res.concept,
        "boxes": res.boxes,
        "num_regions": len(res.boxes),
        "area_fraction": round(res.area_fraction, 4),
        "confidence": res.confidence,
        "note": ("Regions localised by calibrated spectral-index response to "
                 f"'{res.concept}'."),
        "_visual": {"overlay": vis},
    }


def change_analysis_tool(ctx: Dict) -> Dict:
    a, b = ctx["images"][0], ctx["images"][1]
    from .models.change import analyse_pair
    params = ctx.get("params", {})
    out = analyse_pair(a, b, query="",
                       date_a=params.get("date_a", "T1"),
                       date_b=params.get("date_b", "T2"))
    prob = out.pop("prob_map")
    mask = out.pop("change_map")
    vis = _overlay_mask(rgb_composite(b), mask.astype(bool), color=(0.95, 0.75, 0.0))
    if out.get("largest_region_box"):
        _draw_box(vis, out["largest_region_box"], color=(1.0, 0.3, 0.3))
    return {
        **out,
        "_visual": {"change_overlay": vis, "prob_map": prob,
                    "mask": mask},
    }


def change_vqa_tool(ctx: Dict) -> Dict:
    a, b = ctx["images"][0], ctx["images"][1]
    from .models.change import analyse_pair
    params = ctx.get("params", {})
    out = analyse_pair(a, b, query=ctx.get("query", ""),
                       date_a=params.get("date_a", "T1"),
                       date_b=params.get("date_b", "T2"))
    answer = out.get("answer")
    prob, mask = out.pop("prob_map"), out.pop("change_map")
    vis = _overlay_mask(rgb_composite(b), mask.astype(bool), color=(0.95, 0.75, 0.0))
    return {
        "answer": answer,
        "description": out["description"],
        "changed_area_fraction": out["changed_area_fraction"],
        "increased": out["increased"],
        "decreased": out["decreased"],
        "dominant_direction": out["dominant_direction"],
        "largest_region_box": out["largest_region_box"],
        "confidence": out["confidence"],
        "method": out["method"],
        "_visual": {"change_overlay": vis, "prob_map": prob, "mask": mask},
    }


def impact_analysis_tool(ctx: Dict) -> Dict:
    from .impact import analyse_impact, impact_confidence
    from .models.change import ChangeDetectorNet
    a, b = ctx["images"][0], ctx["images"][1]
    det = ChangeDetectorNet()
    cm = det.map(a, b)
    mask = cm["prob_map"] >= CHANGE_MASK_PROB_THRESHOLD
    impact = analyse_impact(a, b, mask, query=ctx.get("query", ""))
    vis = _overlay_mask(rgb_composite(b), mask.astype(bool),
                        color=(0.95, 0.75, 0.0))

    # One shared, auditable formula -- see anvesha.impact.impact_confidence.
    _conf = impact_confidence(impact)
    _confidence = _conf["value"]

    return {
        "findings": impact["findings"],
        "changed_area_ha": impact["changed_area_ha"],
        # Exposed so callers downstream (the investigation path, the decision
        # layer) can re-derive confidence from the same evidence. Without it,
        # anything reading only this tool's output had a dead signal term.
        "changed_fraction": impact["changed_fraction"],
        "confidence_breakdown": _conf["terms"],
        "near_water": impact["near_water"],
        "transitions": impact["transitions"],
        "zones_top": impact["zones_top"],
        "priority_zone": impact["priority_zone"],
        "gsd_m": impact["gsd_m"],
        "gsd_assumed": impact["gsd_assumed"],
        "answer": _impact_answer(impact),
        "confidence": _confidence,
        "source_model": "Impact Analysis engine (change detector + index "
                        "masks + chamfer distance)",
        # prob_map is exposed so the decision layer's sensitivity analysis can
        # re-threshold the raw probability field. Without it, impact_analysis --
        # the task whose decision matters most -- was the one task that could
        # not report how much its conclusion depends on the threshold.
        "_visual": {"change_overlay": vis, "prob_map": cm["prob_map"]},
    }


def _impact_answer(imp: Dict) -> str:
    parts = [f"Changed area ≈ {imp['changed_area_ha']} ha"]
    t = imp["transitions"]
    if t["built_up_new_ha"] > 0:
        nw = imp["near_water"]["within_500m"]
        parts.append(f"new built-up ~{t['built_up_new_ha']} ha"
                     + (f" ({nw * 100:.0f}% within 500 m of water)"
                        if nw > NEAR_WATER_MENTION_THRESHOLD else ""))
    if t["vegetation_lost_ha"] > 0:
        parts.append(f"vegetation lost ~{t['vegetation_lost_ha']} ha")
    if imp.get("priority_zone"):
        parts.append(f"priority zone {imp['priority_zone']}")
    f0 = (imp["findings"] or [{}])[0]
    if f0.get("action"):
        parts.append(f"Action: {f0['action']}")
    return ". ".join(p.rstrip(".") for p in parts) + "."


def optical_sar_tool(ctx: Dict) -> Dict:
    from .io_utils import rgb_composite
    from .models.optical_sar import FusionNet
    imgs = ctx["images"]
    optical = next((i for i in imgs if i.modality != "sar"), imgs[0])
    sar = next((i for i in imgs if i.modality == "sar"), imgs[-1])
    out = FusionNet().analyse(optical, sar)
    vis_opt = np.asarray(rgb_composite(optical), np.float32)
    vis_sar = np.asarray(rgb_composite(sar), np.float32)
    return {**out, "_visual": {"optical": vis_opt, "sar": vis_sar}}


# --------------------------------------------------------------------------- #
# Visual helpers
# --------------------------------------------------------------------------- #

def _overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                  color=(0.95, 0.3, 0.1), alpha: float = 0.45) -> np.ndarray:
    vis = rgb.copy()
    m = mask.astype(bool)
    if vis.shape[2] >= 3:
        for c_i, cv in enumerate(color[:3]):
            vis[..., c_i] = np.where(m, vis[..., c_i] * (1 - alpha) + cv * alpha,
                                     vis[..., c_i])
    return np.clip(vis, 0, 1)


def _draw_box(vis: np.ndarray, box, color=(1.0, 0.35, 0.35), thickness: int = 3):
    x0, y0, x1, y1 = box
    h, w = vis.shape[:2]
    t = max(1, thickness)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w - 1, x1 - 1), min(h - 1, y1 - 1)
    vis[y0c:y0c + t, x0c:x1c + 1] = color
    vis[y1c - t + 1:y1c + 1, x0c:x1c + 1] = color
    vis[y0c:y1c + 1, x0c:x0c + t] = color
    vis[y0c:y1c + 1, x1c - t + 1:x1c + 1] = color
