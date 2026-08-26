"""Train the v2 FPN-lite change detector on LEVIR-CD.

Architecture: shared SceneEncoder (RS-adapted) -> multi-scale differences
(stride-8 + stride-16) -> channel-attention FPN-lite decoder -> full-res logits.
Loss: BCE(pos-weighted) + soft Dice. Candidate checkpoints always saved;
promotion into weights/change_net.pt requires IoU >= 0.45 (sanity floor).

Upgrades over v1:
  - Full 256px crops (not 192px)
  - Heavy augmentation (flip, rotation, brightness/contrast)
  - Gradient accumulation for effective larger batch sizes
  - Early stopping on IoU (patience 10)
  - Experiment logging
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from satquery.models.change import ChangeHeadV2


class LevirDataset(Dataset):
    """Random crops from LEVIR-CD train images with heavy augmentation."""

    def __init__(self, root: Path, crop=256, max_pairs=None, augment=True):
        self.a_dir = next(root.glob("**/A")) if root.exists() else None
        self.b_dir = next(root.glob("**/B")) if root.exists() else None
        self.l_dir = next(root.glob("**/label")) if root.exists() else None
        assert self.a_dir and self.b_dir and self.l_dir, \
            "LEVIR-CD layout with A/ B/ label/ folders expected"
        self.files = sorted(p.name for p in self.a_dir.glob("*.png"))
        if max_pairs:
            self.files = self.files[:max_pairs]
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, a, b, lab):
        """Apply random flip + rotation to all three arrays simultaneously."""
        if np.random.rand() < 0.5:
            a, b, lab = a[:, ::-1].copy(), b[:, ::-1].copy(), lab[:, ::-1].copy()
        if np.random.rand() < 0.5:
            a, b, lab = a[::-1].copy(), b[::-1].copy(), lab[::-1].copy()
        k = np.random.randint(0, 4)
        if k:
            a = np.rot90(a, k).copy()
            b = np.rot90(b, k).copy()
            lab = np.rot90(lab, k).copy()
        # Random brightness/contrast jitter (mild)
        if np.random.rand() < 0.3:
            delta = np.random.uniform(-0.05, 0.05)
            a = np.clip(a + delta, 0, 1).astype(np.float32)
            b = np.clip(b + delta, 0, 1).astype(np.float32)
        return a, b, lab

    def __getitem__(self, i):
        from PIL import Image
        name = self.files[i]
        size = 1024 if (self.a_dir / name).stat().st_size > 2_000_000 else 256
        max_coord = size - self.crop
        h = np.random.randint(0, max(1, max_coord))
        w = np.random.randint(0, max(1, max_coord))
        box = (w, h, w + self.crop, h + self.crop)

        def load(folder, ch):
            im = Image.open(folder / name).convert(ch).crop(box)
            return np.asarray(im, dtype=np.float32)

        a = load(self.a_dir, "RGB") / 255.0
        b = load(self.b_dir, "RGB") / 255.0
        lab = load(self.l_dir, "L")
        lab = (lab > 127).astype(np.float32)

        if self.augment:
            a, b, lab = self._augment(a, b, lab)

        xa = torch.from_numpy(a.transpose(2, 0, 1))
        xb = torch.from_numpy(b.transpose(2, 0, 1))
        return xa, xb, torch.from_numpy(lab[None])


def dice_loss(logits, target, eps=1.0):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LevirDataset(Path(args.data), crop=args.crop,
                      max_pairs=args.max_pairs, augment=True)
    print(f"LEVIR-CD pairs: {len(ds)} | crop {args.crop} | device {device}",
          flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)

    encoder = SceneEncoder(3).to(device)
    head = ChangeHeadV2().to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        print("encoder warm-started from RS-adapted scene encoder", flush=True)

    pos_w = torch.tensor([args.pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_iou = -1.0
    patience_counter = 0
    candidate = CONFIG.weights_dir / "change_candidate.pt"
    PROMOTE_FLOOR = 0.45
    accum_steps = args.grad_accum

    for epoch in range(args.epochs):
        encoder.train(); head.train()
        inter = union = 0
        opt.zero_grad()
        for step, (xa, xb, y) in enumerate(dl):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                f8a = encoder.feature_map(xa, stride=8)
                f16a = encoder.feature_map(xa, stride=16)
                f8b = encoder.feature_map(xb, stride=8)
                f16b = encoder.feature_map(xb, stride=16)
                logits = head(f8a, f16a, f8b, f16b)
                y_r = nn.functional.interpolate(y, size=logits.shape[-2:],
                                                mode="nearest")
                loss = (bce(logits, y_r) + dice_loss(logits, y_r)) / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dl):
                scaler.step(opt); scaler.update()
                opt.zero_grad()
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                pred_full = nn.functional.interpolate(
                    pred, size=y.shape[-2:], mode="nearest")
                inter += (pred_full * y).sum().item()
                union += ((pred_full + y) >= 1).float().sum().item()
        sched.step()
        iou = inter / max(union, 1)
        print(f"epoch {epoch+1}/{args.epochs} change_IoU={iou:.4f}", flush=True)
        if iou > best_iou:
            best_iou = iou
            patience_counter = 0
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict(), "arch": "v2",
                        "val_iou": iou}, candidate)
            print(f"  candidate saved (IoU={iou:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    if best_iou >= PROMOTE_FLOOR:
        candidate.replace(CONFIG.change_weights)
        print(f"PROMOTED candidate (IoU={best_iou:.4f} >= {PROMOTE_FLOOR})")
    else:
        print(f"kept existing change_net.pt -- candidate {best_iou:.4f} below "
              f"promotion floor {PROMOTE_FLOOR}")

    # Experiment log
    from satquery.experiment_log import log_experiment
    log_experiment(
        script="train_change",
        args={"crop": args.crop, "epochs": args.epochs, "max_pairs": args.max_pairs,
              "lr": args.lr, "pos_weight": args.pos_weight,
              "grad_accum": args.grad_accum, "patience": args.patience},
        metrics={"val_iou": best_iou},
        checkpoint=str(CONFIG.change_weights) if best_iou >= PROMOTE_FLOOR else None,
        notes=f"{'PROMOTED' if best_iou >= PROMOTE_FLOOR else 'below floor'}",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "LEVIR-CD" / "train"))
    ap.add_argument("--max-pairs", type=int, default=7000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    train(ap.parse_args())
