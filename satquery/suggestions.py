"""'Ask the data back' — generates contextual follow-up queries after a
result, creating the question → evidence → deeper-question loop."""
from __future__ import annotations

from typing import Dict, List

FOLLOW_UPS: Dict[str, List[str]] = {
    "change_vqa": [
        "Where did most of the change occur?",
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Describe the land-cover in the later image.",
    ],
    "change_analysis": [
        "Has the built-up area increased, decreased, or remained unchanged?",
        "What is the total changed area?",
        "Was vegetation lost near water?",
    ],
    "impact_analysis": [
        "Which zone changed the most?",
        "Was vegetation lost near water?",
        "Should this be prioritised for review?",
    ],
    "investigation": [
        "Which zone changed the most?",
        "Was vegetation lost near water?",
        "Describe the land-cover in the later image.",
        "Highlight the water body in the later image.",
    ],
    "optical_sar": [
        "Which classes does SAR confirm but optical miss?",
        "Is there any cloud-affected region in the optical image?",
    ],
    "captioning": [
        "Highlight the water body referred to in the query.",
        "Is there any built-up area in this image?",
    ],
    "grounding": [
        "What is the area of the highlighted region?",
        "Is there another region like this in the image?",
    ],
    "single_vqa": [
        "Describe the land-cover and major objects visible in this image.",
        "Highlight the water body referred to in the query.",
    ],
}


def suggest(result: Dict) -> List[str]:
    task = result.get("selected_task", "")
    base = list(FOLLOW_UPS.get(task, FOLLOW_UPS["single_vqa"]))
    out = result.get("outputs", {}) or {}

    # contextual extras from actual outputs
    inv = out.get("investigation") or {}
    impact = inv.get("impact") or out
    if impact.get("near_water", {}).get("within_500m", 0) > 0.2:
        base.insert(0, "How much of the change lies within 500 m of water?")
    if impact.get("transitions", {}).get("vegetation_lost_ha", 0) > 0:
        base.insert(1, "Was vegetation lost near water?")
    if out.get("labels"):
        top = out["labels"][0][0] if out["labels"] else ""
        if top:
            base.append(f"Is there any {top.lower()} near the image border?")

    seen, uniq = set(), []
    for q in base:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:5]
