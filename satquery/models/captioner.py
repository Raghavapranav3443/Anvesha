"""Captioning / scene-description specialist.

Primary path: compact transformer decoder fine-tuned on real BigEarthNet.txt
captions joined to co-registered Sentinel-2 patches (scripts/train_captioner.py),
conditioned on the RS-adapted SceneEncoder. Fallback path: deterministic
evidence-grounded template composition from scene classification + layout
statistics (always available, fully auditable).
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

import numpy as np

from ..config import CONFIG
from ..io_utils import RSImage, rgb_composite
from .backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor, _rgb3_compat
from .scene import get_scene_classifier


class LearnedCaptioner:
    """Wraps weights/captioner.pt (see scripts/train_captioner.py)."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or CONFIG.resolve_device()
        self.model = None
        self.vocab = None
        self.val_bleu = None
        path = CONFIG.weights_dir / "captioner.pt"
        if path.exists():
            try:
                import torch
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                from scripts.train_captioner import CaptionVocab, Captioner
                self.vocab = CaptionVocab([])
                self.vocab.itos = ckpt["vocab"]
                self.vocab.stoi = {w: i for i, w in enumerate(self.vocab.itos)}
                self.model = Captioner(len(self.vocab.itos)).to(self.device)
                self.model.load_state_dict(ckpt["model"])
                self.model.eval()
                self.val_bleu = ckpt.get("val_bleu")
            except Exception:
                self.model = None

    def available(self) -> bool:
        return self.model is not None

    def caption(self, img: RSImage) -> Optional[str]:
        if not self.available():
            return None
        import torch
        rgb = _rgb3_compat(img)
        x = to_tensor(resize_np(normalise_for_encoder(rgb, img.modality), 120)) \
            .to(self.device)
        with torch.no_grad():
            fmap = _encoder_feature_map(x, self.device)
            texts = self.model.generate(fmap, self.vocab)
        text = (texts[0] if texts else "") or ""
        if text:
            text = text[0].upper() + text[1:]     # sentence case
        return text or None


_ENC_CACHE: dict = {}


def _encoder_feature_map(x, device):
    import torch
    enc = _ENC_CACHE.get("enc")
    if enc is None:
        enc = SceneEncoder(3).to(device).eval()
        try:
            ckpt = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                              weights_only=False)
            enc.load_state_dict(ckpt["encoder"])
        except Exception:
            pass
        _ENC_CACHE["enc"] = enc
    with torch.no_grad():
        return enc.feature_map(x, stride=8)


_LEARNED: Optional[LearnedCaptioner] = None
_LEARNED_LOCK = threading.Lock()


def get_learned_captioner() -> LearnedCaptioner:
    global _LEARNED
    with _LEARNED_LOCK:
        if _LEARNED is None:
            _LEARNED = LearnedCaptioner()
    return _LEARNED


def _layout(img: RSImage) -> Dict[str, str]:
    """Dominant colour character of top/middle/bottom thirds (display space)."""
    comp = rgb_composite(img)
    h = comp.shape[0]
    thirds = {"upper": comp[: h // 3], "middle": comp[h // 3: 2 * h // 3],
              "lower": comp[2 * h // 3:]}
    out = {}
    for name, band in thirds.items():
        r, g, b = band[..., 0], band[..., 1], band[..., 2]
        exg, water = (2 * g - r - b), (g - r) / (g + r + 1e-6)
        bright = band.mean(axis=2)
        if (water > 0.05).mean() > 0.25:
            out[name] = "water-covered"
        elif (exg > 0.08).mean() > 0.35:
            out[name] = "vegetated"
        elif ((bright > 0.45) & ((band.max(2) - band.min(2)) < 0.18)).mean() > 0.25:
            out[name] = "built-up"
        else:
            out[name] = "open/other"
    return out


MODALITY_PHRASE = {
    "rgb": "optical RGB",
    "multispectral": "multispectral optical",
    "sar": "SAR",
    "grayscale": "single-band optical",
}


def describe(img: RSImage, query_hint: str = "") -> Dict:
    scene = get_scene_classifier()
    pred = scene.predict(img, top_k=5)
    labels = pred["labels"]
    layout = _layout(img)

    mod_phrase = MODALITY_PHRASE.get(img.modality, img.modality)
    parts: List[str] = []

    learned = get_learned_captioner().caption(img)
    source = "template composition (scene evidence)"
    if learned:
        parts.append(learned.strip().rstrip(".") + ".")
        src_bits = ["BigEarthNet.txt-trained caption decoder"]
        lc = get_learned_captioner()
        if lc.val_bleu is not None:
            src_bits.append(f"val BLEU {lc.val_bleu:.3f}")
        source = " · ".join(src_bits)

    if labels:
        top = ", ".join(n for n, _ in labels[:2])
        parts.append(f"Spectral analysis indicates dominance of {top}.")
    zones = [f"{k} third appears {v}" for k, v in layout.items()]
    parts.append("Spatially, the " + "; ".join(zones[:2]) +
                 ("." if len(zones) < 3 else f"; and the {zones[2]}."))

    features = []
    comp = rgb_composite(img)
    g = comp.mean(2)
    dark_frac = float((g < 0.18).mean())
    if layout.get("lower") == "water-covered" or layout.get("middle") == "water-covered":
        features.append("a distinct water body")
    if layout.get("upper") == "built-up" or layout.get("middle") == "built-up":
        features.append("clustered built structures")
    if dark_frac > 0.30 and img.modality == "sar":
        features.append("low-backscatter smooth surfaces (typical of calm water or flat ground)")
    if img.modality == "multispectral":
        features.append("spectral contrast suitable for vegetation/water discrimination")
    if features:
        parts.append("Major visible features include " + " and ".join(features) + ".")

    caption = " ".join(parts)
    conf = min(0.95, 0.45 + 0.12 * len(labels)) if not learned else \
        min(0.96, 0.62 + 0.1 * len(labels))
    return {
        "caption": caption,
        "confidence": round(float(conf), 3),
        "labels": labels,
        "layout": layout,
        "source": pred["source"] if not learned else source,
        "generative": bool(learned),
    }
