"""Scene classification specialist: RS-adapted encoder + label heads.

Two operating paths:
* trained   - weights produced by scripts/train_scene_encoder.py
              (EuroSAT quick track) or the BigEarthNet track are loaded;
* heuristic - spectral-index fallback so the assistant remains usable before
              any fine-tuning has been run.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import CONFIG, BEN19_CLASSES, EUROSAT_CLASSES, CONCEPT_TO_EUROSAT, CONCEPT_TO_BEN19
from ..io_utils import RSImage, rgb_composite
from .backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor


class SceneClassifier:
    """Multi-label / single-label land-cover scene understanding."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or CONFIG.resolve_device()
        self.label_space = "eurosat"
        self.classes: List[str] = EUROSAT_CLASSES
        self.encoder: Optional[SceneEncoder] = None
        self.head = None
        self.trained = False
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        import torch
        path = CONFIG.scene_encoder_weights
        if path.exists():
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                in_ch = int(ckpt.get("in_channels", 3))
                self.label_space = ckpt.get("label_space", "eurosat")
                self.classes = ckpt.get(
                    "classes", BEN19_CLASSES if self.label_space == "ben19" else EUROSAT_CLASSES)
                enc = SceneEncoder(in_channels=in_ch)
                enc.load_state_dict(ckpt["encoder"])
                head_w = ckpt["head"]
                head = torch.nn.Linear(enc.FEATURE_DIM, head_w.shape[0])
                head.load_state_dict({"weight": head_w, "bias": ckpt["head_bias"]})
                self.encoder, self.head = enc.eval(), head.eval()
                self.encoder.to(self.device); self.head.to(self.device)
                self.in_channels = in_ch
                self.trained = True
            except Exception:
                self.encoder, self.head, self.trained = None, None, False

    def _forward(self, img: RSImage):
        import torch
        rgb3 = _as_3band(img)
        x = to_tensor(resize_np(normalise_for_encoder(rgb3, img.modality), 224)).to(self.device)
        with torch.no_grad():
            feat = self.encoder(x)
            logits = self.head(feat)[0]
            if self.label_space == "eurosat":
                return torch.softmax(logits, -1).cpu().numpy()
            return torch.sigmoid(logits).cpu().numpy()

    # ------------------------------------------------------------------ #
    def predict(self, img: RSImage, top_k: int = 5) -> Dict:
        probs = self._forward(img) if self.trained else None
        if probs is not None:
            idx = np.argsort(-probs)[:top_k]
            labels = [(self.classes[i], float(probs[i])) for i in idx if probs[i] > 0.03]
            conf = float(np.clip(np.max(probs) * (1.0 + 0.25 * len(labels)), 0.05, 0.99))
            return {"labels": labels, "confidence": round(conf, 3),
                    "source": "fine-tuned scene encoder (RS-adapted)"}
        stats = spectral_stats(img)
        scores = heuristic_class_scores(stats)
        return {"labels": scores[:top_k], "confidence": round(scores[0][1], 3),
                "source": "spectral heuristics (run training to enable the adapted model)"}

    def concept_presence(self, img: RSImage) -> Dict[str, float]:
        """Coarse concept probabilities used by VQA/caption/fusion specialists."""
        pred = self.predict(img, top_k=len(self.classes))
        present = {name for name, _ in pred["labels"]}
        out = {}
        table = CONCEPT_TO_EUROSAT if self.label_space == "eurosat" else CONCEPT_TO_BEN19
        stats = spectral_stats(img)
        for concept, related in table.items():
            p = max([s for n, s in pred["labels"] if n in related], default=0.0)
            out[concept] = float(max(p, _heuristic_concept(concept, stats)))
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_3band(img: RSImage) -> np.ndarray:
    a = img.array
    c = a.shape[2]
    if c >= 4:
        return a[..., [2, 1, 0]]
    if c == 3:
        return a
    if c == 2:
        return np.stack([a[..., 0], a[..., 1], a.mean(axis=2)], axis=2)
    return np.repeat(a, 3, axis=2)


def spectral_stats(img: RSImage) -> Dict[str, float]:
    """Cheap per-concept evidence from colour/spectral statistics."""
    rgb = _as_3band(img)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b                       # excess green (vegetation proxy)
    ndwi_like = (g - r) / (g + r + 1e-6)        # water proxy on visible bands
    bright = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    return {
        "veg_frac": float((exg > 0.08).mean()),
        "water_frac": float(((ndwi_like > 0.05) & (b > g * 0.9)).mean()),
        "built_frac": float(((bright > 0.45) & (sat < 0.18)).mean()),
        "bare_frac": float(((r > g) & (g > b) & (bright > 0.3)).mean()),
        "brightness": float(bright.mean()),
    }


def _heuristic_concept(concept: str, stats: Dict[str, float]) -> float:
    key_map = {"water": "water_frac", "vegetation": "veg_frac",
               "built-up": "built_frac", "bare": "bare_frac"}
    key = key_map.get(concept)
    if key:
        return float(min(1.0, stats[key] * 6.0))  # fraction -> pseudo-probability
    if concept == "road":
        return float(min(1.0, stats["built_frac"] * 3.0))
    if concept == "agriculture":
        return float(min(1.0, stats["veg_frac"] * 2.5))
    return 0.0


def heuristic_class_scores(stats: Dict[str, float]) -> List[Tuple[str, float]]:
    cand = []
    if stats["water_frac"] > 0.01:
        cand.append(("River" if stats["water_frac"] < 0.5 else "SeaLake",
                     min(0.95, 0.4 + stats["water_frac"])))
    veg = stats["veg_frac"]
    built = stats["built_frac"]
    bare = stats["bare_frac"]
    if veg > 0.15:
        cand.append(("Forest" if veg > 0.55 else "HerbaceousVegetation",
                     min(0.93, 0.35 + veg)))
    if built > 0.04:
        cand.append(("Residential" if stats["brightness"] < 0.62 else "Industrial",
                     min(0.92, 0.30 + built * 2.0)))
    if bare > 0.10:
        cand.append(("PermanentCrop", min(0.7, 0.25 + bare)))
    if stats["built_frac"] > 0.02 and veg < 0.3:
        cand.append(("Highway", min(0.65, 0.2 + stats["built_frac"])))
    if not cand:
        cand.append(("AnnualCrop", 0.35))
    cand.sort(key=lambda t: -t[1])
    return [(n, round(float(s), 3)) for n, s in cand]


# Singleton accessor ------------------------------------------------------- #
_INSTANCE: Optional[SceneClassifier] = None
_scene_lock = threading.Lock()


def get_scene_classifier(device: Optional[str] = None) -> SceneClassifier:
    global _INSTANCE
    if device is not None:
        # Explicit device request → always create a fresh instance
        return SceneClassifier(device=device)
    if _INSTANCE is None:
        with _scene_lock:
            if _INSTANCE is None:
                _INSTANCE = SceneClassifier()
    return _INSTANCE
