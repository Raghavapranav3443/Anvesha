"""Optical-SAR cross-modal analysis specialist.

* FusionNet - dual-branch RS encoders (optical + SAR) with joint multi-label
  head, trainable on co-registered BigEarthNet v2 S1+S2 pairs
  (scripts/train_optical_sar.py).
* Fallback complementarity analyser combining per-modality evidence:
  spectral indices from optical and texture/backscatter structure from SAR.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import CONFIG
from ..io_utils import RSImage
from .backbone import SceneEncoder, _rgb3_compat as _rgb3, normalise_for_encoder, resize_np, to_tensor


class FusionNet:
    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        self.device = device or CONFIG.resolve_device()
        self.torch = torch
        self.opt_encoder = None
        self.sar_encoder = None
        self.head = None
        self.classes: List[str] = []
        self.trained = False
        if CONFIG.fusion_weights.exists():
            try:
                ckpt = torch.load(CONFIG.fusion_weights, map_location="cpu",
                                  weights_only=False)
                if ckpt.get("synthetic"):
                    # plumbing-test checkpoint trained on random pairs - keep
                    # the heuristic analyser active on real imagery
                    print("FusionNet: synthetic-mode weights found; "
                          "using heuristic analyser until the network is "
                          "trained on real BigEarthNet v2 S1+S2 pairs")
                else:
                    oe, se = SceneEncoder(3), SceneEncoder(2)
                    oe.load_state_dict(ckpt["opt_encoder"])
                    se.load_state_dict(ckpt["sar_encoder"])
                    head = torch.nn.Linear(2 * SceneEncoder.FEATURE_DIM,
                                           len(ckpt["classes"]))
                    head.load_state_dict(ckpt["head"])
                    self.opt_encoder, self.sar_encoder, self.head = \
                        oe.eval().to(self.device), se.eval().to(self.device), \
                        head.eval().to(self.device)
                    self.classes = ckpt["classes"]
                    self.trained = True
            except Exception:
                self.trained = False

    def _fused_scores(self, optical: RSImage, sar: RSImage) -> Dict[str, float]:
        """Forward pass of the dual-branch network (requires trained weights)."""
        o = _rgb3(optical)
        s = sar.array.astype(np.float32)
        if s.shape[2] >= 2:
            s = s[..., :2]
        else:
            s = np.repeat(s, 2, axis=2)
        xo = to_tensor(resize_np(normalise_for_encoder(o, "rgb"), 224)).to(self.device)
        xs = to_tensor(resize_np(normalise_for_encoder(s, "sar"), 224)).to(self.device)
        with self.torch.no_grad():
            feat = self.torch.cat([self.sar_encoder(xs), self.opt_encoder(xo)], dim=1)
            probs = self.torch.sigmoid(self.head(feat))[0].cpu().numpy()
        return {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

    def analyse(self, optical: RSImage, sar: RSImage) -> Dict:
        if self.trained:
            fused = self._fused_scores(optical, sar)
            source = "fine-tuned dual-branch fusion network (BigEarthNet S1+S2)"
        else:
            fused = _heuristic_fused_scores(optical, sar)
            source = "per-modality spectral/backscatter heuristics " \
                     "(train scripts/train_optical_sar.py for the learned model)"

        opt_presence = {k: round(v, 3) for k, v in
                        _modality_scores(optical).items()}
        sar_presence = {k: round(v, 3) for k, v in
                        _sar_scores(sar).items()}

        notes = _complementarity_notes(optical, sar, opt_presence, sar_presence)
        agreement = {
            k: ("agree" if abs(opt_presence[k] - sar_presence[k]) < 0.25 else
                ("optical-only" if opt_presence[k] > sar_presence[k] else "SAR-only"))
            for k in opt_presence
        }
        conf = float(np.clip(0.5 * max(fused.values()) +
                             0.25 * (1 - np.mean([abs(opt_presence[k] - sar_presence[k])
                                                  for k in opt_presence])) + 0.25,
                             0.05, 0.97))

        return {
            "fused_classes": {k: round(v, 3) for k, v in sorted(
                fused.items(), key=lambda kv: -kv[1])[:6]},
            "dominant": max(fused, key=fused.get),
            "optical_evidence": opt_presence,
            "sar_evidence": sar_presence,
            "agreement": agreement,
            "notes": notes,
            "confidence": round(conf, 3),
            "source": source,
        }


# --------------------------------------------------------------------------- #
# Evidence extraction
# --------------------------------------------------------------------------- #

def _modality_scores(img: RSImage) -> Dict[str, float]:
    rgb = _rgb3(img)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b
    ndwi_like = (g - r) / (g + r + 1e-6)
    bright = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "water": min(1.0, ((ndwi_like > 0.05) & (b > g)).mean() * 6),
        "vegetation": min(1.0, (exg > 0.08).mean() * 5),
        "built-up": min(1.0, ((bright > 0.45) & (sat < 0.18)).mean() * 8),
        "bare": min(1.0, ((r > g) & (g > b)).mean() * 5),
    }


def _sar_scores(img: RSImage) -> Dict[str, float]:
    a = img.array.astype(np.float32)
    if a.shape[2] > 2:
        a = a[..., :2] if a.shape[2] >= 2 else np.repeat(a, 2, axis=2)[..., :2]
    if img.modality != "sar":
        # treat as intensity image anyway
        pass
    amp = a[..., 0]
    sm = _blur(amp, 9)
    texture = np.abs(amp - sm).mean()
    dark = float((sm < np.percentile(sm, 35)).mean())
    bright_frac = float((sm > np.percentile(sm, 80)).mean())
    return {
        "water": min(1.0, dark * 2.2),                 # specular: low backscatter
        "built-up": min(1.0, bright_frac * 2.4),       # double-bounce: strong returns
        "vegetation": min(1.0, texture * 6.0),         # volume scattering: high texture
        "bare": min(1.0, (1 - abs(texture - 0.08) / 0.15) * 0.5),
    }


def _heuristic_fused_scores(optical: RSImage, sar: RSImage) -> Dict[str, float]:
    o, s = _modality_scores(optical), _sar_scores(sar)
    return {k: float(np.clip(0.6 * o[k] + 0.4 * s[k], 0, 1)) for k in o}


def _complementarity_notes(optical: RSImage, sar: RSImage,
                           o: Dict[str, float], s: Dict[str, float]) -> List[str]:
    notes = []
    rgb = _rgb3(optical)
    cloudy = bool((rgb.min(axis=2) > 0.75).mean() > 0.10 or
                  (np.abs(rgb[..., 0] - rgb[..., 2]) < 0.04).mean() > 0.45)
    if cloudy:
        notes.append("Optical scene shows bright/low-chroma areas consistent with "
                     "cloud or haze; SAR penetrates clouds and remains reliable here.")
    if s.get("water", 0) > 0.4 and o.get("water", 0) < 0.2:
        notes.append("SAR indicates smooth low-backscatter water surfaces that are "
                     "not obvious in the optical composite.")
    if s.get("built-up", 0) > 0.35:
        notes.append("Strong coherent backscatter (double-bounce) confirms built-up "
                     "structures in the SAR channel.")
    if o.get("vegetation", 0) > 0.4 and s.get("vegetation", 0) > 0.3:
        notes.append("Both modalities agree on vegetation: green/exG response in "
                     "optical and volume-scattering texture in SAR.")
    if not notes:
        notes.append("Modalities are broadly complementary; no occlusion artefacts detected.")
    return notes


def _blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = x.shape
    return (cs[k:k + h, k:k + w] - cs[:-k, k:k + w] - cs[k:k + h, :-k]
            + cs[:-k, :-k]) / float(k * k)
