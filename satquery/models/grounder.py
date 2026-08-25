"""Text-guided region grounding specialist.

Maps a referring phrase (e.g. 'the water body') to a spatial mask and
bounding box. Uses remote-sensing spectral indices (NDVI/NDWI/ExG for
multispectral input; colour proxies for RGB) with connected-component
analysis. Thresholds are calibrated on the RS-adapted training track.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import CONFIG
from ..io_utils import RSImage
from ..text import concept_of_text


@dataclass
class GroundingResult:
    concept: str
    mask: np.ndarray                 # HxW bool
    boxes: List[List[int]]           # [x0,y0,x1,y1] pixel coords
    score_map: np.ndarray            # HxW float in [0,1]
    area_fraction: float
    confidence: float


def _score_map(img: RSImage, concept: str) -> np.ndarray:
    a = img.array.astype(np.float32)
    c = a.shape[2]

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    if concept == "water":
        if img.modality == "sar":
            # smooth low-backscatter regions
            g = a.mean(axis=2)
            sm = _box_blur(g, 9)
            score = 1.0 - norm(sm)
        elif c >= 4:  # NDWI = (Green - NIR)/(Green + NIR)
            green, nir = a[..., 1], a[..., 3]
            ndwi = (green - nir) / (green + nir + 1e-6)
            score = norm(ndwi)
        else:
            r, g = a[..., 0], a[..., 1]
            b = a[..., 2] if c > 2 else (r + g) / 2.0
            ndwi_like = (g - r) / (g + r + 1e-6)
            blue_dom = (b - (r + g) / 2.0)
            score = norm(0.6 * ndwi_like + 0.4 * blue_dom)
    elif concept in ("vegetation", "agriculture"):
        if c >= 4:  # NDVI
            red, nir = a[..., 2], a[..., 3]
            idx = (nir - red) / (nir + red + 1e-6)
        else:
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            idx = (2 * g - r - b)
        score = norm(idx)
    elif concept == "built-up":
        bright = a.mean(axis=2)
        sat = a.max(axis=2) - a.min(axis=2)
        score = norm(bright) * (1.0 - norm(sat))
        if img.modality == "sar":
            score = norm(_box_blur(a.mean(axis=2), 5))  # strong double-bounce
    elif concept == "road":
        bright = a.mean(axis=2)
        score = norm(bright) * (1.0 - norm(np.abs(np.gradient(bright, axis=1))[..., 0]))
    else:  # bare
        r = a[..., 0]
        b = a[..., 2 % c] if c > 1 else np.zeros_like(r)
        score = norm(r - b) * norm(a.mean(axis=2))
    return np.clip(score, 0.0, 1.0)


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = x.shape
    return (cs[k:k + h, k:k + w] - cs[:-k, k:k + w] - cs[k:k + h, :-k]
            + cs[:-k, :-k]) / float(k * k)


def _connected_components(mask: np.ndarray):
    """Simple two-pass labelling (8-connectivity)."""
    from collections import deque
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        row = mask[i]
        for j in range(w):
            if row[j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                labels[i, j] = cur
                while q:
                    y, x = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                    and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                q.append((ny, nx))
    return labels, cur


def _learned_box(img: RSImage, concept: str):
    """Experimental learned head (BigEarthNet.txt reference boxes).

    Disabled by default: measured IoU>0.5 hit-rate ~0.12 on held-out refs
    (region-style expressions don't transfer to free user queries). Enable
    with SATQUERY_LEARNED_GROUNDING=1.
    """
    import os
    if os.environ.get("SATQUERY_LEARNED_GROUNDING") != "1":
        return None
    path = CONFIG.weights_dir / "grounding.pt"
    if not path.exists():
        return None
    global _GROUNDED_HEAD
    try:
        import torch
        if _GROUNDED_HEAD is None:
            from scripts.train_grounding import GroundingHead, bow as gbow, \
                heat_to_box
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            head = GroundingHead()
            head.load_state_dict(ckpt["head"])
            device = CONFIG.resolve_device()
            head = head.eval().to(device)
            _GROUNDED_HEAD = (head, gbow, heat_to_box, device,
                              float(ckpt.get("val_hit", 0.0)))
        head, gbow, heat_to_box, device, val_hit = _GROUNDED_HEAD
        from .backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor

        enc = _GROUND_ENC.get("enc")
        if enc is None:
            enc = SceneEncoder(3).to(device).eval()
            ck2 = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                             weights_only=False)
            enc.load_state_dict(ck2["encoder"])
            _GROUND_ENC["enc"] = enc
        rgb3 = img.array
        c = rgb3.shape[2]
        if c >= 4:
            rgb3 = rgb3[..., [2, 1, 0]]
        elif c == 1 or c == 2:
            rgb3 = np.repeat(rgb3.mean(axis=2, keepdims=True), 3, axis=2)
        x = to_tensor(resize_np(normalise_for_encoder(rgb3, "rgb"), 120)).to(device)
        text = f"where is a connected region of {concept} located? largest smallest"
        q = torch.from_numpy(gbow(text)).unsqueeze(0).to(device)
        with torch.no_grad():
            fmap = enc.feature_map(x, stride=8)
            heat = torch.sigmoid(head(fmap, q)[0, 0]).cpu().numpy()
        box = heat_to_box(heat)                        # cx,cy,w,h normalized
        h, w = img.height, img.width
        x0 = max(0, int((box[0] - box[2] / 2) * w)); x1 = min(w, int((box[0] + box[2] / 2) * w))
        y0 = max(0, int((box[1] - box[3] / 2) * h)); y1 = min(h, int((box[1] + box[3] / 2) * h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return {"boxes": [[x0, y0, x1, y1]],
                "val_hit": val_hit}
    except Exception:
        return None


_GROUNDED_HEAD = None
_GROUND_ENC: dict = {}


def ground(img: RSImage, query: str, max_regions: int = 5,
           min_area_frac: float = 0.004) -> GroundingResult:
    concept, lex_score = concept_of_text(query)
    if concept is None:
        concept = "water"
        lex_score = 0.15

    sm = _score_map(img, concept)
    thr = float(np.percentile(sm, 88))          # adaptive threshold
    mask = sm >= max(thr, 0.55)

    # despeckle: remove tiny components
    labels, n = _connected_components(mask)
    total = mask.size
    keep = np.zeros_like(mask)
    boxes = []
    sizes = []
    if n > 0:
        counts = np.bincount(labels.ravel())
        order = np.argsort(-counts[1:])[:max_regions] + 1
        for lb in order:
            comp = labels == lb
            frac = comp.sum() / total
            if frac < min_area_frac:
                continue
            keep |= comp
            ys, xs = np.where(comp)
            boxes.append([int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1])
            sizes.append(frac)

    purity = float(sm[keep].mean()) if keep.any() else 0.0
    area_fraction = float(keep.sum() / total)

    # learned referring-expression head (BigEarthNet.txt reference boxes)
    learned = _learned_box(img, concept)
    boxes = list(boxes)
    source = "calibrated spectral-index response"
    if learned and not boxes:
        boxes = learned["boxes"]
        source = "learned referring-expression head (BigEarthNet.txt)"
    elif learned and boxes:
        # ensemble: keep index regions as primary, append model suggestion if
        # it does not duplicate the primary box
        lb = learned["boxes"][0]
        x0, y0, x1, y1 = boxes[0]
        lx, ly = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
        inside = x0 <= lx <= x1 and y0 <= ly <= y1
        if not inside:
            boxes.append(lb)
        source = ("spectral-index regions + learned referring-expression "
                  "suggestion")

    conf = float(np.clip(0.35 * lex_score + 0.45 * purity + 0.2 * min(area_fraction * 12, 1),
                         0.05, 0.97))
    if learned and not boxes:
        conf = max(conf, min(0.6, 0.3 + learned.get("val_hit", 0.0)))

    return GroundingResult(concept=concept, mask=keep, boxes=boxes,
                           score_map=sm, area_fraction=area_fraction,
                           confidence=round(conf, 3))
