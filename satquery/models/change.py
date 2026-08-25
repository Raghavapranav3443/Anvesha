"""Bi-temporal change analysis specialist.

* ChangeDetectorNet - Siamese RS-adapted encoder + difference head, trainable
  on LEVIR-CD (scripts/train_change.py).
* Fallback differencing with morphological cleanup.
* Change description & change-VQA reasoner built on the change map plus
  per-date scene classification deltas.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import CONFIG
from ..io_utils import RSImage, rgb_composite
from .backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor, _rgb3_compat
from .scene import get_scene_classifier


class ChangeDetectorNet:
    """Siamese change detector producing a pixel-level change probability map."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        self.device = device or CONFIG.resolve_device()
        self.torch = torch
        self.encoder: Optional[SceneEncoder] = None
        self.head = None
        self.trained = False
        if CONFIG.change_weights.exists():
            try:
                ckpt = torch.load(CONFIG.change_weights, map_location="cpu",
                                  weights_only=False)
                enc = SceneEncoder(3)
                enc.load_state_dict(ckpt["encoder"])
                head = _DiffHead()
                head.load_state_dict(ckpt["head"])
                self.encoder, self.head = enc.eval(), head.eval()
                self.encoder.to(self.device); self.head.to(self.device)
                self.trained = True
            except Exception:
                self.trained = False

    def map(self, a: RSImage, b: RSImage) -> Dict:
        """Returns {'prob_map': HxW float [0,1], 'method': str}."""
        t = self.torch
        fa_rgb, fb_rgb = _rgb3_compat(a), _rgb3_compat(b)
        h, w = a.height, a.width

        if self.trained:
            prob = self._tiled_infer(fa_rgb, fb_rgb, h, w)
            if float(prob.max()) < 0.7:
                # distribution shift safeguard: learned map is flat, so the
                # scene is likely out of the training distribution - fall back
                # to direct bi-temporal differencing evidence
                prob = np.maximum(prob, _differencing_map(
                    rgb_composite(a), rgb_composite(b)))
                method = ("fine-tuned Siamese network + differencing support "
                          "(low-confidence region)")
            else:
                method = "fine-tuned Siamese change network (LEVIR-CD)"
        else:
            fb = _differencing_map(rgb_composite(a), rgb_composite(b))
            prob, method = fb, ("smoothed bi-temporal differencing "
                                "(train scripts/train_change.py to enable the "
                                "learned model)")

        return {"prob_map": prob.astype(np.float32), "method": method}

    def _tiled_infer(self, fa_rgb, fb_rgb, h: int, w: int,
                     tile: int = 128) -> np.ndarray:
        """Sliding-window inference at the network's training resolution."""
        t = self.torch
        acc = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        step = tile // 2
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        if ys[-1] + tile < h:
            ys.append(h - tile)
        if xs[-1] + tile < w:
            xs.append(w - tile)
        with t.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    ca = fa_rgb[y0:y1, x0:x1]
                    cb = fb_rgb[y0:y1, x0:x1]
                    # pad small edge tiles back to full size
                    pad_y, pad_x = tile - ca.shape[0], tile - ca.shape[1]
                    if pad_y or pad_x:
                        ca = np.pad(ca, ((0, pad_y), (0, pad_x), (0, 0)))
                        cb = np.pad(cb, ((0, pad_y), (0, pad_x), (0, 0)))
                    xa = to_tensor(resize_np(normalise_for_encoder(ca, "rgb"), tile)).to(self.device)
                    xb = to_tensor(resize_np(normalise_for_encoder(cb, "rgb"), tile)).to(self.device)
                    e1 = self.encoder.feature_map(xa, stride=8)
                    e2 = self.encoder.feature_map(xb, stride=8)
                    logits = self.head(t.cat([e1, e2], dim=1))
                    pm = t.sigmoid(logits)[0, 0].cpu().numpy()
                    ph, pw = min(y1, h) - y0, min(x1, w) - x0
                    pm_full = _resize_prob(pm, pw, ph)
                    acc[y0:y0 + ph, x0:x0 + pw] += pm_full
                    weight[y0:y0 + ph, x0:x0 + pw] += 1.0
        return acc / np.maximum(weight, 1e-6)


class _DiffHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# High-level analysis: description + change-VQA + optional map stats
# --------------------------------------------------------------------------- #

def analyse_pair(a: RSImage, b: RSImage, query: str = "",
                 date_a: str = "T1", date_b: str = "T2") -> Dict:
    det = ChangeDetectorNet()
    cm = det.map(a, b)
    prob = cm["prob_map"]

    mask = prob >= 0.5
    area_frac = float(mask.mean())
    conf_change = float(np.clip(prob[mask].mean() if mask.any() else 0.0,
                                0.05, 0.99)) if mask.any() else 0.05

    # largest region + direction
    box, centroid_dir, n_regions = _region_stats(mask)

    scene = get_scene_classifier()
    pa = scene.concept_presence(a)
    pb = scene.concept_presence(b)

    deltas = {k: round(pb[k] - pa[k], 3) for k in pa}
    increased = sorted([k for k, v in deltas.items() if v > 0.05],
                       key=lambda k: -deltas[k])
    decreased = sorted([k for k, v in deltas.items() if v < -0.05],
                       key=lambda k: deltas[k])

    desc_parts = []
    if area_frac < 0.005:
        desc_parts.append(
            f"Between {date_a} and {date_b}, no significant surface change was detected.")
        overall = "no significant change"
    else:
        what = []
        for c in increased[:2]:
            what.append(f"increase in {c}")
        for c in decreased[:2]:
            what.append(f"decrease in {c}")
        what_txt = ", ".join(what) if what else "surface alterations of mixed character"
        pct = area_frac * 100.0
        desc_parts.append(
            f"Between {date_a} and {date_b}, the dominant changes are {what_txt}, "
            f"affecting about {pct:.1f}% of the scene.")
        if centroid_dir:
            desc_parts.append(f"The main changed region lies towards the {centroid_dir}.")
        if box:
            x0, y0, x1, y1 = box
            desc_parts.append(
                f"Largest connected changed area spans pixels ({x0},{y0})-({x1},{y1}).")
        overall = "; ".join(what) if what else "mixed surface change"

    answer = None
    if query:
        q = query.lower()
        if any(w in q for w in ("has", "increased", "decreased", "remained")):
            bu_delta = deltas.get("built-up", 0.0)
            if abs(bu_delta) <= 0.05:
                answer = f"Built-up area has remained largely unchanged (delta {bu_delta:+.2f})."
            elif bu_delta > 0:
                answer = (f"Yes - built-up area has increased (presence delta "
                          f"{bu_delta:+.2f}); change concentrated towards the {centroid_dir or 'centre'}.")
            else:
                answer = f"Built-up area has decreased (presence delta {bu_delta:+.2f})."
        elif "what changed" in q or "change" in q:
            answer = overall.replace(";", ",") .capitalize() + "."
        elif "where" in q:
            if box:
                x0, y0, x1, y1 = box
                answer = (f"Change is concentrated towards the {centroid_dir or 'centre'}; "
                          f"largest region at pixels ({x0},{y0})-({x1},{y1}).")
            else:
                answer = "No spatially coherent change region was detected."

    conf = 0.55 * conf_change + 0.25 * min(area_frac * 20, 1.0) + \
        0.10 * (len(increased) + len(decreased)) / 4.0
    return {
        "description": " ".join(desc_parts),
        "answer": answer or desc_parts[0],
        "changed_area_fraction": round(area_frac, 4),
        "num_change_regions": n_regions,
        "largest_region_box": box,
        "dominant_direction": centroid_dir,
        "concept_deltas": deltas,
        "increased": increased,
        "decreased": decreased,
        "confidence": round(float(np.clip(conf, 0.05, 0.97)), 3),
        "change_map": mask.astype(np.uint8),
        "prob_map": prob,
        "method": cm["method"],
    }


def _differencing_map(comp_a: np.ndarray, comp_b: np.ndarray) -> np.ndarray:
    ga = resize_np(np.asarray(comp_a, np.float32), 256).mean(axis=2)
    gb = resize_np(np.asarray(comp_b, np.float32), 256).mean(axis=2)
    d = np.abs(ga - gb)
    d = _box_blur(d, 7)
    thr = max(float(d.mean() + 1.5 * d.std()), 0.06)
    return np.clip((d - thr) / (d.max() - thr + 1e-6), 0, 1)


def _region_stats(mask: np.ndarray):
    h, w = mask.shape
    step = max(1, (h * w) // 400000)
    sub = mask[::step, ::step]
    labels_s, n = _label(sub)
    scale = step
    best_box = None
    if n > 0:
        counts = np.bincount(labels_s.ravel())
        lb = int(np.argmax(counts[1:]) + 1)
        comp = labels_s == lb
        ys, xs = np.where(comp)
        if len(xs):
            best_box = [int(xs.min()) * scale, int(ys.min()) * scale,
                        int((xs.max()) + 1) * scale, int((ys.max()) + 1) * scale]
    cy, cx = np.where(mask)
    direction = ""
    if len(cx):
        dy = cy.mean() / h - 0.5
        dx = cx.mean() / w - 0.5
        ns = "north" if dy < -0.05 else ("south" if dy > 0.05 else "")
        ew = "west" if dx < -0.05 else ("east" if dx > 0.05 else "")
        direction = "-".join(p for p in (ns, ew) if p) or "centre"
    return best_box, direction, int(n)


def _label(mask: np.ndarray):
    from collections import deque
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                labels[i, j] = cur
                while q:
                    y, x = q.popleft()
                    for dyi in (-1, 0, 1):
                        for dxi in (-1, 0, 1):
                            ny, nx = y + dyi, x + dxi
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                    and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                q.append((ny, nx))
    return labels, cur


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = x.shape
    return (cs[k:k + h, k:k + w] - cs[:-k, k:k + w] - cs[k:k + h, :-k]
            + cs[:-k, :-k]) / float(k * k)


def _resize_prob(prob: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8))
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0
