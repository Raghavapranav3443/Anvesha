"""Train the optical-SAR dual-branch fusion network on co-registered
BigEarthNet v2 (reBEN) S1+S2 pairs. Also supports --synthetic mode to verify
the training pipeline end-to-end without the full archives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG, BEN19_CLASSES
from satquery.models.backbone import SceneEncoder


class ReBENPairs(Dataset):
    """Expects reBEN-style layout: root/S1/*.tif + root/S2/*.tif sharing ids,
    with labels.json mapping id -> list of class indices."""

    def __init__(self, s1_dir: Path, s2_dir: Path, labels_file: Path):
        self.s1_dir, self.s2_dir = s1_dir, s2_dir
        self.labels = json.loads(labels_file.read_text(encoding="utf-8")) \
            if labels_file.exists() else {}
        self.ids = sorted(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        import rasterio
        pid = self.ids[i]
        with rasterio.open(self.s1_dir / f"{pid}.tif") as src:
            s1 = np.moveaxis(src.read(), 0, -1).astype(np.float32)
            if s1.shape[2] > 2:
                s1 = s1[..., :2]
            elif s1.shape[2] == 1:
                s1 = np.repeat(s1, 2, axis=2)
        img2 = Image.open(self.s2_dir / f"{pid}.tif").convert("RGB")
        s2 = np.asarray(img2.resize((120, 120)), dtype=np.float32) / 255.0
        y = torch.zeros(len(BEN19_CLASSES))
        for ci in self.labels[pid]:
            y[ci] = 1.0
        return torch.from_numpy(s1.transpose(2, 0, 1)), \
            torch.from_numpy(s2.transpose(2, 0, 1)), y


class SyntheticPairs(Dataset):
    """Plumbing-test pairs: SAR = noisy grayscale of optical; random labels."""

    def __init__(self, n=256):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.RandomState(i)
        rgb = rng.rand(3, 64, 64).astype(np.float32) * 0.6
        gray = rgb.mean(axis=0, keepdims=True)
        sar = gray + rng.randn(1, 64, 64).astype(np.float32) * 0.05
        sar = np.repeat(sar, 2, axis=0)
        y = torch.zeros(len(BEN19_CLASSES))
        for c in rng.choice(len(BEN19_CLASSES), size=3, replace=False):
            y[c] = 1.0
        return torch.from_numpy(sar), torch.from_numpy(rgb), y


class BEN14KPairs(Dataset):
    """Real co-registered BigEarthNet v2 S1+S2 pairs (14K cross-modal subset,
    HF ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2).

    Expects data/bigearthnet_14k/BEN_14k/ containing BigEarthNet-S1/<split>/,
    BigEarthNet-S2/<split>/ and metadata.parquet.
    """

    def __init__(self, root: Path, split="train", max_pairs=None,
                 image_size: int = 120):
        import pandas as pd
        self.root = Path(root)
        self.image_size = image_size
        self.s1_dir = self.root / "BigEarthNet-S1" / split
        self.s2_dir = self.root / "BigEarthNet-S2" / split
        meta = pd.read_parquet(self.root / "metadata.parquet")
        meta = meta[meta["split"] == split]

        # Pair S1<->S2 by geographic position: filename tail encodes
        # tile + row + col of the reBEN patch grid.
        def keyed(d: Path):
            out = {}
            for p in d.glob("*.tif"):
                base, r, c = p.stem.rsplit("_", 2)
                tile = base.split("_")[-1].lstrip("T")
                out[(tile, r, c)] = p
            return out

        k1 = keyed(self.s1_dir)
        k2 = keyed(self.s2_dir)
        labelled = meta[meta["s1_name"].isin({p.stem for p in
                                              self.s1_dir.glob("*.tif")})] \
            .set_index("s1_name")

        self.samples = []
        for stem, p1 in ((p.stem, p) for p in sorted(self.s1_dir.glob("*.tif"))):
            base, r, c = stem.rsplit("_", 2)
            tile = base.split("_")[-1].lstrip("T")
            p2 = k2.get((tile, r, c))
            if p2 is None or stem not in labelled.index:
                continue
            row = labelled.loc[stem]
            self.samples.append((p1, p2, [str(x) for x in row["labels"]]))
        if max_pairs:
            self.samples = self.samples[:max_pairs]
        from satquery.config import BEN19_CLASSES
        self.classes = BEN19_CLASSES
        self.lut = {n: i for i, n in enumerate(BEN19_CLASSES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import rasterio
        p1, p2, labels = self.samples[i]
        with rasterio.open(p1) as src:
            s1 = np.moveaxis(src.read().astype(np.float32), 0, -1)   # dB scale
        with rasterio.open(p2) as src:
            s2 = np.moveaxis(src.read().astype(np.float32), 0, -1)   # 10 bands x10000
        rgb = np.clip(s2[..., [2, 1, 0]] / 10000.0, 0, 1)            # B04,B03,B02
        mu, sd = s1.mean(axis=(0, 1), keepdims=True), \
            s1.std(axis=(0, 1), keepdims=True) + 1e-6
        s1n = ((s1 - mu) / sd).astype(np.float32)

        from satquery.models.backbone import resize_np
        rgb_s = resize_np(rgb, self.image_size)
        sar_s = resize_np(s1n, self.image_size)[..., :2]

        y = torch.zeros(len(self.classes))
        for lab in labels:
            if lab in self.lut:
                y[self.lut[lab]] = 1.0
        return torch.from_numpy(sar_s.transpose(2, 0, 1)), \
            torch.from_numpy(rgb_s.transpose(2, 0, 1)), y


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "synthetic":
        ds = SyntheticPairs()
    elif args.dataset == "bigearthnet_14k":
        root = Path(args.data)
        ds_tr = BEN14KPairs(root, "train", args.max_pairs,
                            image_size=args.image_size)
        ds_va = BEN14KPairs(root, "validation", 512,
                            image_size=args.image_size)
    else:
        root = Path(args.data)
        ds_tr = ReBENPairs(root / "S1", root / "S2", root / "labels.json")
        ds_va = None
    print("train pairs:", len(ds_tr), "| device:", device)
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size) if ds_va else None

    sar_enc = SceneEncoder(2).to(device)
    opt_enc = SceneEncoder(3).to(device)
    head = nn.Linear(2 * SceneEncoder.FEATURE_DIM, len(BEN19_CLASSES)).to(device)
    opt = torch.optim.AdamW(
        list(sar_enc.parameters()) + list(opt_enc.parameters()) +
        list(head.parameters()), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    best_recall = -1.0
    for epoch in range(args.epochs):
        sar_enc.train(); opt_enc.train(); head.train(); seen = hits = 0
        for s1, s2, y in dl:
            s1, s2, y = s1.to(device), s2.to(device), y.to(device)
            opt.zero_grad()
            fused = torch.cat([sar_enc(s1), opt_enc(s2)], dim=1)
            logits = head(fused)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
        msg = f"epoch {epoch+1}/{args.epochs}"
        if dl_va:
            sar_enc.eval(); opt_enc.eval(); head.eval(); seen = hits = 0
            with torch.no_grad():
                for s1, s2, y in dl_va:
                    s1, s2, y = s1.to(device), s2.to(device), y.to(device)
                    logits = head(torch.cat([sar_enc(s1), opt_enc(s2)], dim=1))
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    hits += ((pred * y).sum(1) / y.sum(1).clamp(min=1)).sum().item()
                    seen += len(y)
            recall = hits / max(seen, 1)
            print(f"{msg} val_mean_recall={recall:.4f}")
            if recall > best_recall:
                best_recall = recall
                _save(sar_enc, opt_enc, head, args)
        else:
            _save(sar_enc, opt_enc, head, args)
    if not dl_va:
        pass
    else:
        print("saved best checkpoint (val_mean_recall=%.4f)" % best_recall)


def _save(sar_enc, opt_enc, head, args):
    torch.save({"opt_encoder": opt_enc.state_dict(),
                "sar_encoder": sar_enc.state_dict(),
                "head": head.state_dict(),
                "classes": BEN19_CLASSES,
                "synthetic": bool(args.dataset == "synthetic")},
               CONFIG.fusion_weights)
    print("saved", CONFIG.fusion_weights)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "bigearthnet_14k", "rebenn"])
    ap.add_argument("--data", default=str(CONFIG.data_dir / "bigearthnet_14k" / "BEN_14k"))
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    train(ap.parse_args())
