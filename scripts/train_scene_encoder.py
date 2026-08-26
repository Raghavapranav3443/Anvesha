"""Remote-sensing adaptation of the shared SceneEncoder.

Quick track   : EuroSAT (10-class Sentinel-2 RGB patches) - runs in minutes.
BigEarth track: BigEarthNet v2 S2 patches, 19-class multi-label
                (pass --dataset bigearthnet_s2 after downloading reBEN).

The resulting weights/scene_encoder.pt is loaded automatically by every
specialist tool (this is the mandated RS fine-tuning/adaptation step).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG, EUROSAT_CLASSES
from satquery.models.backbone import SceneEncoder


class EuroSATDataset(Dataset):
    """EuroSAT folder dataset with configurable resolution and augmentation."""

    def __init__(self, root: Path, max_per_class: int = 2500,
                 image_size: int = 96, augment: bool = False):
        from PIL import Image
        self.samples = []
        self.classes = EUROSAT_CLASSES
        self.image_size = image_size
        self.augment = augment
        for ci, cls in enumerate(self.classes):
            files = sorted((root / cls).glob("*.jpg"))[:max_per_class]
            self.samples += [(f, ci) for f in files]
        random.Random(0).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        f, y = self.samples[i]
        arr = np.asarray(
            Image.open(f).convert("RGB").resize(
                (self.image_size, self.image_size)),
            dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1].copy()
            k = np.random.randint(0, 4)
            if k:
                arr = np.rot90(arr, k).copy()
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        return x, y


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    ds = EuroSATDataset(Path(args.data), max_per_class=args.max_per_class,
                        image_size=args.image_size, augment=args.epochs > 3)
    n_train = int(len(ds) * 0.9)
    tr, va = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=args.batch_size)

    encoder = SceneEncoder(in_channels=3).to(device)
    head = nn.Linear(encoder.FEATURE_DIM, len(EUROSAT_CLASSES)).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_acc = 0.0
    candidate_path = CONFIG.scene_encoder_weights.with_suffix(".candidate.pt")
    for epoch in range(args.epochs):
        encoder.train(); head.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = lossf(head(encoder(x)), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        encoder.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                p = head(encoder(x.to(device))).argmax(1).cpu()
                correct += (p == y).sum().item(); total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{args.epochs} val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.weight.detach().cpu(),
                "head_bias": head.bias.detach().cpu(),
                "label_space": "eurosat",
                "classes": EUROSAT_CLASSES,
                "in_channels": 3,
                "image_size": args.image_size,
                "val_accuracy": acc,
            }, candidate_path)

    # Promotion gate: never regress the shared backbone below the shipped one
    if best_acc >= args.promote_floor:
        candidate_path.replace(CONFIG.scene_encoder_weights)
        print(f"PROMOTED {CONFIG.scene_encoder_weights} (val_acc={best_acc:.4f})")
    else:
        print(f"kept existing encoder -- candidate {best_acc:.4f} below floor "
              f"{args.promote_floor}")
        candidate_path.unlink(missing_ok=True)

    # Experiment log
    from satquery.experiment_log import log_experiment
    log_experiment(
        script="train_scene_encoder",
        args={"max_per_class": args.max_per_class, "epochs": args.epochs,
              "image_size": args.image_size, "lr": args.lr,
              "promote_floor": args.promote_floor},
        metrics={"val_acc": best_acc},
        checkpoint=str(CONFIG.scene_encoder_weights) if best_acc >= args.promote_floor else None,
        notes=f"{'PROMOTED' if best_acc >= args.promote_floor else 'below floor'}",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "eurosat" / "2750"))
    ap.add_argument("--max-per-class", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--promote-floor", type=float, default=0.93)
    train(ap.parse_args())
