"""Backbone gate: frozen DINOv2-small probe vs our RS-adapted SceneEncoder.

Identical protocol for both backbones: linear probe trained on the same
200 img/class EuroSAT subset, evaluated on held-out 100 img/class. Adoption
rule (decided BEFORE running): swap only if DINOv2 wins by >= 2 points AND an
offline-bundlable weight export exists (~90 MB, Apache-2.0).

Run:  python scripts/gate_dinov2.py [--max-per-class 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.config import CONFIG, EUROSAT_CLASSES


def collect_split(root: Path, lo: int, hi: int):
    files, labels = [], []
    for ci, cls in enumerate(EUROSAT_CLASSES):
        fs = sorted((root / cls).glob("*.jpg"))
        take = fs[lo:hi]
        files += take
        labels += [ci] * len(take)
    return files, np.array(labels)


def load_batch(files, size):
    from PIL import Image
    import numpy as np
    out = np.zeros((len(files), size, size, 3), dtype=np.float32)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((size, size))
        out[i] = np.asarray(im, dtype=np.float32) / 255.0
    return out


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Featurizer:
    """Extracts frozen embeddings for one backbone."""

    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self.size = 224 if name == "dinov2" else 128
        if name == "dinov2":
            import torch
            self.model = torch.hub.load("facebookresearch/dinov2",
                                        "dinov2_vits14").to(device).eval()
            self.dim = 384
        else:
            import torch
            from anvesha.models.backbone import SceneEncoder
            enc = SceneEncoder(3).to(device).eval()
            ck = CONFIG.scene_encoder_weights
            if ck.exists():
                enc.load_state_dict(torch.load(ck, map_location="cpu",
                                               weights_only=False)["encoder"])
            self.model = enc
            self.dim = enc.FEATURE_DIM

    @torch.no_grad()
    def embed(self, files, bs=128):
        import torch
        feats = np.zeros((len(files), self.dim), dtype=np.float32)
        for i in range(0, len(files), bs):
            chunk = load_batch(files[i:i + bs], self.size)
            x = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).to(self.device)
            if self.name == "dinov2":
                x = (x - torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)
                     .to(self.device)) / torch.from_numpy(IMAGENET_STD) \
                    .view(1, 3, 1, 1).to(self.device)
                feats[i:i + bs] = self.model(x).cpu().numpy()
            else:
                # chunk is already [0,1] RGB; SceneEncoder normalises internally
                feats[i:i + bs] = self.model(x).cpu().numpy()
        return feats


def probe_accuracy(train_f, train_y, val_f, val_y) -> float:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_f, train_y)
    return float(clf.score(val_f, val_y))


def main(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    tr_files, tr_y = collect_split(root, 0, args.max_per_class)
    va_files, va_y = collect_split(root, args.max_per_class,
                                   args.max_per_class + args.val_per_class)
    print(f"train={len(tr_files)} val={len(va_files)} device={device}",
          flush=True)

    results = {}
    for name in ("scene_encoder", "dinov2"):
        try:
            fz = Featurizer(name, device)
        except Exception as e:
            print(f"[{name}] unavailable: {type(e).__name__}: {e}", flush=True)
            continue
        tr_f = fz.embed(tr_files)
        va_f = fz.embed(va_files)
        acc = probe_accuracy(tr_f, tr_y, va_f, va_y)
        results[name] = acc
        print(f"{name:<14} linear-probe val acc = {acc:.4f}", flush=True)

    if len(results) == 2:
        delta = results["dinov2"] - results["scene_encoder"]
        verdict = "ADOPT" if delta >= 0.02 else "KEEP scene encoder"
        print(f"\ndelta = {delta:+.4f}  -> gate: {verdict}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "eurosat" / "2750"))
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--val-per-class", type=int, default=100)
    main(ap.parse_args())
