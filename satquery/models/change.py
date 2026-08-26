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
        self.arch = "v1"
        if CONFIG.change_weights.exists():
            try:
                ckpt = torch.load(CONFIG.change_weights, map_location="cpu",
                                  weights_only=False)
                enc = SceneEncoder(3)
                enc.load_state_dict(ckpt["encoder"])
                if ckpt.get("arch") == "v2":
                    head = ChangeHeadV2()
                    head.load_state_dict(ckpt["head"])
                    self.arch = "v2"
                else:
                    head = _DiffHead()
                    head.load_state_dict(ckpt["head"])
                self.encoder, self.head = enc.eval(), head.eval()
                self.encoder.to(self.device); self.head.to(self.device)
                self.trained = True
            except Exception:
                self.trained = False

    def map(self, a: RSImage, b: RSImage, tta: bool = False) -> Dict:
        """Returns {'prob_map': HxW float [0,1], 'method': str}.

        When *tta* is True, predictions are averaged over the original image
        plus horizontal-flip, vertical-flip and both-flip variants (typically
        gives +2-3 F1 points at 4× inference cost).
        """
        t = self.torch
        fa_rgb, fb_rgb = _rgb3_compat(a), _rgb3_compat(b)
        h, w = a.height, a.width

        def _infer_once(fa, fb):
            if self.trained and self.arch == "v2":
                return self._tiled_infer_v2(fa, fb, h, w)
            elif self.trained:
                return self._tiled_infer(fa, fb, h, w)
            else:
                return _differencing_map(rgb_composite(a), rgb_composite(b))

        if self.trained and tta:
            # Test-time augmentation: average over 4 transforms
            prob_orig = _infer_once(fa_rgb, fb_rgb)
            prob_hflip = _infer_once(fa_rgb[:, ::-1].copy(), fb_rgb[:, ::-1].copy())[:, ::-1].copy()
            prob_vflip = _infer_once(fa_rgb[::-1].copy(), fb_rgb[::-1].copy())[::-1].copy()
            prob_both = _infer_once(
                fa_rgb[::-1, ::-1].copy(), fb_rgb[::-1, ::-1].copy())[::-1, ::-1].copy()
            prob = (prob_orig + prob_hflip + prob_vflip + prob_both) / 4.0
            prob = _clean_mask_prob(prob)
            method = "FPN-lite Siamese + TTA (4-way average)"
        elif self.trained and self.arch == "v2":
            prob = self._tiled_infer_v2(fa_rgb, fb_rgb, h, w)
            prob = _clean_mask_prob(prob)
            if float(prob.max()) < 0.35:
                prob = np.maximum(prob, _differencing_map(
                    rgb_composite(a), rgb_composite(b)))
                method = ("FPN-lite Siamese network + differencing support "
                          "(low-confidence region)")
            else:
                method = "FPN-lite Siamese change network (LEVIR-CD, v2)"
        elif self.trained:
                method = "fine-tuned Siamese change network (LEVIR-CD)"
        else:
            fb = _differencing_map(rgb_composite(a), rgb_composite(b))
            prob, method = fb, ("smoothed bi-temporal differencing "
                                "(train scripts/train_change.py to enable the "
                                "learned model)")

        return {"prob_map": prob.astype(np.float32), "method": method}

    def _tiled_infer_v2(self, fa_rgb, fb_rgb, h: int, w: int,
                        tile: int = 192) -> np.ndarray:
        """Sliding-window inference for the v2 FPN-lite head. Tiles of 192px,
        output at stride 2, upsampled per tile and averaged in overlaps."""
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
                    pad_y, pad_x = tile - ca.shape[0], tile - ca.shape[1]
                    if pad_y or pad_x:
                        ca = np.pad(ca, ((0, pad_y), (0, pad_x), (0, 0)))
                        cb = np.pad(cb, ((0, pad_y), (0, pad_x), (0, 0)))
                    xa = to_tensor(resize_np(ca, tile)).to(self.device)
                    xb = to_tensor(resize_np(cb, tile)).to(self.device)
                    f8a = self.encoder.feature_map(xa, stride=8)
                    f16a = self.encoder.feature_map(xa, stride=16)
                    f8b = self.encoder.feature_map(xb, stride=8)
                    f16b = self.encoder.feature_map(xb, stride=16)
                    logits = self.head(f8a, f16a, f8b, f16b)
                    pm = t.sigmoid(logits)[0, 0].cpu().numpy()
                    ph, pw = min(y1, h) - y0, min(x1, w) - x0
                    pm_full = _resize_prob(pm, pw, ph)
                    acc[y0:y0 + ph, x0:x0 + pw] += pm_full
                    weight[y0:y0 + ph, x0:x0 + pw] += 1.0
        return acc / np.maximum(weight, 1e-6)

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


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention on the difference features —
    suppresses pseudo-change channels before decoding."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]


class ChangeHeadV2(nn.Module):
    """FPN-lite siamese decoder.

    Consumes stride-8 (128ch) and stride-16 (256ch) feature maps from BOTH
    dates, forms multi-scale differences, fuses with channel attention, and
    decodes to full-resolution logits (output stride 2).
    forward(f8a, f16a, f8b, f16b) -> B x 1 x H/2 x W/2
    """

    def __init__(self, ch8: int = 128, ch16: int = 256, d: int = 128):
        super().__init__()
        self.reduce8 = nn.Sequential(nn.Conv2d(ch8 * 2, d, 1), nn.ReLU(inplace=True))
        self.reduce16 = nn.Sequential(nn.Conv2d(ch16 * 2, d, 1), nn.ReLU(inplace=True))
        self.se = SEBlock(d * 2)
        self.dec1 = nn.Sequential(nn.Conv2d(d * 2, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, f8a, f16a, f8b, f16b):
        d8 = self.reduce8(torch.cat([f8a, f8b], dim=1))
        d16 = self.reduce16(torch.cat([f16a, f16b], dim=1))
        d16u = nn.functional.interpolate(d16, size=d8.shape[-2:],
                                         mode="bilinear", align_corners=False)
        d = self.se(torch.cat([d8, d16u], dim=1))
        h = self.dec1(d)
        h = nn.functional.interpolate(h, scale_factor=2, mode="bilinear",
                                      align_corners=False)
        h = self.dec2(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="bilinear",
                                      align_corners=False)
        return self.final(h)


class _DiffHead(nn.Module):
    """Legacy v1 head (stride-16 box regression era). Kept for rollback."""

    def __init__(self):
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


def _clean_mask_prob(prob: np.ndarray) -> np.ndarray:
    """Morphological open/close + small-component removal on the thresholded
    mask (applied to the probability field so downstream thresholds stay)."""
    thr = 0.5
    mask = (prob >= thr)
    if not mask.any():
        return prob
    from PIL import Image, ImageFilter
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MinFilter(3))     # erosion: kill specks
    im = im.filter(ImageFilter.MaxFilter(3))     # dilation: restore shape
    cleaned = np.asarray(im) > 127
    if cleaned.sum() == 0:
        return np.zeros_like(prob)
    # remove small components (only for manageable image sizes)
    if max(cleaned.shape) <= 1024:
        labels, n = _label(cleaned)
        if n:
            counts = np.bincount(labels.ravel())
            for lb in range(1, n + 1):
                if counts[lb] < 40:              # < ~40 px is noise
                    cleaned[labels == lb] = False
    out = prob.copy()
    out[~cleaned] = 0.0
    return out


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
    """Connected component labeling.  Uses ``scipy.ndimage.label`` when
    available (C-compiled union-find); falls back to a pure-Python BFS.
    """
    try:
        from scipy.ndimage import label as _scipy_label
        labels, n = _scipy_label(mask.astype(np.int32))
        return labels.astype(np.int32), int(n)
    except ImportError:
        pass
    # Fallback: BFS-based labeling (no path compression — slow on large masks)
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
