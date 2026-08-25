"""Train the Siamese change detector on LEVIR-CD (bi-temporal RGB pairs)."""
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


class _DiffHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, x):
        return self.net(x)


class LevirDataset(Dataset):
    """Random 128px crops from LEVIR-CD train images."""

    def __init__(self, root: Path, crop=128, max_pairs=None):
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
        h = np.random.randint(0, 1024 - self.crop)
        w = np.random.randint(0, 1024 - self.crop)
        box = (w, h, w + self.crop, h + self.crop)

        def load(folder, ch):
            im = Image.open(folder / name).convert(ch).crop(box)
            return np.asarray(im, dtype=np.float32)

        a = load(self.a_dir, "RGB") / 255.0
        b = load(self.b_dir, "RGB") / 255.0
        lab = load(self.l_dir, "L")
        lab = (np.asarray(Image.fromarray(lab.astype(np.uint8))
                          .resize((32, 32), Image.BILINEAR)) > 127).astype(np.float32)
        xa = torch.from_numpy(a.transpose(2, 0, 1))
        xb = torch.from_numpy(b.transpose(2, 0, 1))
        return xa, xb, torch.from_numpy(lab[None])


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LevirDataset(Path(args.data), max_pairs=args.max_pairs)
    print("LEVIR-CD pairs:", len(ds))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)

    encoder = SceneEncoder(3).to(device)
    head = _DiffHead().to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
    pos_w = torch.tensor([args.pos_weight]).to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_iou = -1.0
    for epoch in range(args.epochs):
        encoder.train(); head.train(); inter = union = 0
        for xa, xb, y in dl:
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            opt.zero_grad()
            e1 = encoder.feature_map(xa, stride=8)
            e2 = encoder.feature_map(xb, stride=8)
            logits = head(torch.cat([e1, e2], dim=1))
            y = nn.functional.interpolate(y, size=logits.shape[-2:], mode="nearest")
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            pred = (torch.sigmoid(logits) > 0.5).float()
            inter += (pred * y).sum().item()
            union += ((pred + y) >= 1).float().sum().item()
        sched.step()
        iou = inter / max(union, 1)
        print(f"epoch {epoch+1}/{args.epochs} change_IoU={iou:.4f}")
        if iou > best_iou:
            if iou < 0.10:
                print(f"  skipped saving (IoU {iou:.4f} below sanity floor)")
                continue
            best_iou = iou
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict()},
                       CONFIG.change_weights)
            print(f"  saved best (IoU={iou:.4f})")
    print("saved", CONFIG.change_weights, "best IoU", round(best_iou, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "LEVIR-CD" / "train"))
    ap.add_argument("--max-pairs", type=int, default=1400)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    train(ap.parse_args())
