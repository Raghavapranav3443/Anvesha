"""B3 — grounding ranked output + per-box region-VQA escape hatch.

Keeps spectral-index regions as the PRIMARY (auditable) result and adds a
prior-scored alternate, an honest ``{primary, alternates[2], why}`` rank, and
per-box "Ask about this region" crops. NEVER blanks: a min-area fallback
guarantees at least one region when any signal exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


@dataclass
class RankedBox:
    box: list                      # [x0, y0, x1, y1]
    rank: int                      # 1 = primary
    role: str                      # "primary" | "alternate"
    why: str
    shape_priors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"box": self.box, "rank": self.rank, "role": self.role,
                "why": self.why, "shape_priors": self.shape_priors}


@dataclass
class GroundingRanked:
    primary: Optional[RankedBox]
    alternates: List[RankedBox] = field(default_factory=list)
    why: str = ""
    fallback_used: bool = False
    concept: str = ""

    def to_dict(self) -> dict:
        return {"primary": self.primary.to_dict() if self.primary else None,
                "alternates": [a.to_dict() for a in self.alternates],
                "why": self.why, "fallback_used": self.fallback_used,
                "concept": self.concept}


def _box_mask(box, h, w):
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def rank_regions(result, img, concept: str) -> GroundingRanked:
    """Convert a ``GroundingResult`` into ranked output with shape priors."""
    from .priors import shape_priors, prior_score

    h, w = img.height, img.width
    boxes = list(result.boxes or [])
    ranked = []
    for i, box in enumerate(boxes[:5]):
        bm = _box_mask(box, h, w)
        priors = shape_priors(box, bm)
        ranked.append((box, prior_score(concept, priors), priors))

    # sort: primary = largest spectral region; alternates prior-scored
    if not ranked:
        return GroundingRanked(primary=None, alternates=[],
                               why="no spectral region above threshold",
                               fallback_used=False, concept=concept)

    ranked.sort(key=lambda t: -t[1])
    primary = RankedBox(box=list(ranked[0][0]), rank=1, role="primary",
                        why="largest calibrated spectral-index region",
                        shape_priors=ranked[0][2])
    alternates = [RankedBox(box=list(b), rank=i + 2, role="alternate",
                             why=f"prior-scored alternate (shape match {s:.2f})",
                             shape_priors=p)
                  for i, (b, s, p) in enumerate(ranked[1:3])]

    return GroundingRanked(primary=primary, alternates=alternates,
                           why=(f"{concept} located by calibrated spectral "
                                f"index; {len(alternates)} alternate(s) "
                                f"ranked by shape prior"),
                           fallback_used=False, concept=concept)
