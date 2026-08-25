"""Train the v2 FPN-lite change detector on LEVIR-CD.

Architecture: shared SceneEncoder (RS-adapted) → multi-scale differences
(stride-8 + stride-16) → channel-attention FPN-lite decoder → full-res logits.
Loss: BCE(pos-weighted) + soft Dice. Candidate checkpoints always saved;
promotion into weights/change_net.pt requires IoU >= 0.45 (sanity floor).
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
    """Random 192px crops from LEVIR-CD train images."""

    def __init__(self, root: Path, crop=192, max_pairs=None):
        self.a_dir = next(root.glob("**/A")) if root.exists() else None
        self.b_dir = next(root.glob("**/B")) if root.exists() else None
        self.l_dir = next(root.glob("**/label")) if root.exists() else None
        assert self.a_dir and self.b_dir and self.l_dir, \
            "LEVIR-CD layout with A/ B/ label/ folders expected"
        self.files = sorted(p.name for p in self.a_dir.glob("*.png"))
        if max_pairs:
            self.files = self.files[:max_pairs]
        self.crop = crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        name = self.files[i]
        size = 1024 if (self.a_dir / name).stat().st_size > 2_000_000 else 256
        h = np.random.randint(0, 1024 - self.crop) if size == 1024 else \
            np.random.randint(0, 256 - self.crop)
        w = np.random.randint(0, 1024 - self.crop) if size == 1024 else \
            np.random.randint(0, 256 - self.crop)
        box = (w, h, w + self.crop, h + self.crop)

        def load(folder, ch):
            im = Image.open(folder / name).convert(ch).crop(box)
            return np.asarray(im, dtype=np.float32)

        a = load(self.a_dir, "RGB") / 255.0
        b = load(self.b_dir, "RGB") / 255.0
        lab = load(self.l_dir, "L")
        lab = (lab > 127).astype(np.float32)
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
    ds = LevirDataset(Path(args.data), crop=args.crop, max_pairs=args.max_pairs)
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
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_iou = -1.0
    candidate = CONFIG.weights_dir / "change_candidate.pt"
    PROMOTE_FLOOR = 0.45
    for epoch in range(args.epochs):
        encoder.train(); head.train(); inter = union = 0
        for xa, xb, y in dl:
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                f8a = encoder.feature_map(xa, stride=8)
                f16a = encoder.feature_map(xa, stride=16)
                f8b = encoder.feature_map(xb, stride=8)
                f16b = encoder.feature_map(xb, stride=16)
                logits = head(f8a, f16a, f8b, f16b)
                y_r = nn.functional.interpolate(y, size=logits.shape[-2:],
                                                mode="nearest")
                loss = bce(logits, y_r) + dice_loss(logits, y_r)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
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
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict(), "arch": "v2",
                        "val_iou": iou}, candidate)
            print(f"  candidate saved (IoU={iou:.4f})", flush=True)

    if best_iou >= PROMOTE_FLOOR:
        candidate.replace(CONFIG.change_weights)
        print(f"PROMOTED candidate (IoU={best_iou:.4f} >= {PROMOTE_FLOOR})")
    else:
        print(f"kept existing change_net.pt — candidate {best_iou:.4f} below "
              f"promotion floor {PROMOTE_FLOOR} (candidate preserved at "
              f"{candidate.name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "LEVIR-CD" / "train"))
    ap.add_argument("--max-pairs", type=int, default=1400)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    train(ap.parse_args())
