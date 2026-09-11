# -*- coding: utf-8 -*-
"""B4 — fusion spatial product: per-pixel agreement map (optical vs SAR)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class AgreementArtifact:
    overlay: np.ndarray
    fractions: Dict[str, float]
    quadrants: Dict[str, float]
    notes: List[str]
    sar_water_pixel_count: int = 0


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = x.shape
    return (cs[k:k + h, k:k + w] - cs[:-k, k:k + w] - cs[k:k + h, :-k]
            + cs[:-k, :-k]) / float(k * k)


def _optical_evidence(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    ndwi_like = (g - r) / (g + r + 1e-6)
    bright = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "water": np.clip((ndwi_like - 0.05) * 6.0, 0, 1)
                   * np.clip((b - g * 0.9) * 4.0 + 0.5, 0, 1),
        "vegetation": np.clip((exg - 0.08) * 5.0, 0, 1),
        "built-up": np.clip((bright - 0.45) * 8.0, 0, 1)
                    * np.clip(0.18 - sat, 0, 1) / 0.18,
        "bare": np.clip((r - g) * 5.0, 0, 1) * np.clip(bright - 0.3, 0, 1),
    }


def _sar_evidence(sar: np.ndarray) -> Dict[str, np.ndarray]:
    vv = sar[..., 0].astype(np.float32)
    sm = _box_blur(vv, 9)
    texture = np.abs(vv - sm)
    return {
        "water": np.clip((1.0 - _norm(sm)) * 2.2, 0, 1),
        "built-up": np.clip(_norm(sm) * 2.4, 0, 1),
        "vegetation": np.clip(_norm(texture) * 6.0, 0, 1),
        "bare": np.clip(1.0 - np.abs(_norm(texture) - 0.08) / 0.15, 0, 1) * 0.5,
    }


def _cloud_mask(rgb: np.ndarray) -> np.ndarray:
    bright = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    return ((bright > 0.75) & (sat < 0.06)).astype(np.float32)


def build_agreement(optical, sar,
                    classes=("water", "vegetation", "built-up", "bare"),
                    agree_tol: float = 0.10) -> AgreementArtifact:
    from ..io_utils import rgb_composite
    arr_o = getattr(optical, "array", optical)
    rgb_o = rgb_composite(optical) if arr_o.ndim == 3 else optical
    sar_arr = getattr(sar, "array", sar).astype(np.float32)
    if sar_arr.ndim == 2:
        sar_arr = sar_arr[..., None]
    if sar_arr.shape[2] >= 2:
        sar_arr = sar_arr[..., :2]
    h, w = rgb_o.shape[:2]
    o = _optical_evidence(rgb_o)
    s = _sar_evidence(sar_arr)
    cloud = _cloud_mask(rgb_o)
    keys = list(classes)
    o_stack = np.stack([o[k] for k in keys], axis=0)
    s_stack = np.stack([s[k] for k in keys], axis=0)
    agree = np.abs(o_stack - s_stack).mean(axis=0)
    overlay = np.zeros((h, w), np.uint8)
    overlay[agree < agree_tol] = 0
    o_dominant = o_stack.max(axis=0) > s_stack.max(axis=0)
    overlay[(agree >= agree_tol) & o_dominant & (cloud < 0.5)] = 1
    overlay[(agree >= agree_tol) & (~o_dominant) & (cloud < 0.5)] = 2
    overlay[cloud >= 0.5] = 3
    sar_water = (s["water"] > 0.4) & (o["water"] < 0.2) & (cloud < 0.5)
    overlay[sar_water] = 4
    valid = overlay != 3
    total = max(int(valid.sum()), 1)
    fracs = {k: float((overlay == i).sum()) / total
             for i, k in enumerate(["agree", "optical-wins", "sar-wins",
                                    "cloud", "sar-only-water"])}
    fracs["sar-only-water"] = float(sar_water.sum()) / total
    quadrants = {}
    for qi, qn in enumerate(["NW", "NE", "SW", "SE"]):
        r0, r1 = (0, h // 2) if qi < 2 else (h // 2, h)
        c0, c1 = (0, w // 2) if qi % 2 == 0 else (w // 2, w)
        cell = overlay[r0:r1, c0:c1]
        vc = np.bincount(cell.ravel(), minlength=5)[:5]
        quadrants[qn] = round(float(vc[int(np.argmax(vc))])
                              / max(int(cell.size), 1), 3)
    notes = []
    sw_pct = fracs["sar-only-water"] * 100
    if sw_pct > 0.5:
        notes.append(f"SAR-only water covers {sw_pct:.1f}% of scene, "
                     f"concentrated {max(quadrants, key=quadrants.get)}.")
    cloud_pct = fracs["cloud"] * 100
    if cloud_pct > 5:
        notes.append(f"Optical {cloud_pct:.1f}% cloud/haze-occluded; "
                     f"SAR reliable there.")
    if not notes:
        notes.append("Modalities broadly complementary; no occlusion artefacts.")
    return AgreementArtifact(overlay=overlay, fractions=fracs, quadrants=quadrants,
                             notes=notes, sar_water_pixel_count=int(sar_water.sum()))


def write_geotiff(classes_map: np.ndarray, reference, path) -> None:
    import rasterio
    from rasterio.transform import from_bounds
    h, w = classes_map.shape
    if reference is not None and getattr(reference, "crs", None) and \
            getattr(reference, "transform_bounds", None):
        orig_h = getattr(reference, "original_height", h) or h
        orig_w = getattr(reference, "original_width", w) or w
        transform = from_bounds(*reference.transform_bounds, orig_w, orig_h)
        crs = reference.crs
    else:
        transform, crs = from_bounds(0, 0, w * 10, h * 10, w, h), None
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=0) as dst:
        dst.write(classes_map, 1)
        dst.descriptions = ("agreement_map",)
