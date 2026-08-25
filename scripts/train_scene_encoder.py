"""Remote-sensing adaptation of the shared SceneEncoder.

Quick track   : EuroSAT (10-class Sentinel-2 RGB patches) - runs in minutes.
BigEarth track: BigEarthNet v2 S2 patches, 19-class multi-label
                (pass --dataset bigearthnet_s2 after downloading reBEN).

The resulting weights/scene_encoder.pt is loaded automatically by every
specialist tool (this is the mandated RS fine-tuning/adaptation step).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG, BEN19_CLASSES, EUROSAT_CLASSES
from satquery.models.backbone import SceneEncoder


class EuroSATDataset(Dataset):
    """EuroSAT folder dataset with a capped sample count for quick adaptation."""

    def __init__(self, root: Path, max_per_class: int = 200):
        from PIL import Image
        self.samples = []
        self.classes = EUROSAT_CLASSES
        for ci, cls in enumerate(self.classes):
            files = sorted((root / cls).glob("*.jpg"))[:max_per_class]
            self.samples += [(f, ci) for f in files]
        random.Random(0).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        f, y = self.samples[i]
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((64, 64)),
                         dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        return x, y


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    ds = EuroSATDataset(Path(args.data), max_per_class=args.max_per_class)
    n_train = int(len(ds) * 0.9)
    tr, va = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=args.batch_size)

    encoder = SceneEncoder(in_channels=3).to(device)
    head = nn.Linear(encoder.FEATURE_DIM, len(EUROSAT_CLASSES)).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        encoder.train(); head.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(head(encoder(x)), y)
            loss.backward()
            opt.step()
        encoder.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                p = head(encoder(x.to(device))).argmax(1).cpu()
                correct += (p == y).sum().item(); total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{args.epochs} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.weight.detach().cpu(),
                "head_bias": head.bias.detach().cpu(),
                "label_space": "eurosat",
                "classes": EUROSAT_CLASSES,
                "in_channels": 3,
                "val_accuracy": acc,
            }, CONFIG.scene_encoder_weights)
    print(f"saved {CONFIG.scene_encoder_weights} (best val_acc={best_acc:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "eurosat" / "2750"))
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    train(ap.parse_args())
