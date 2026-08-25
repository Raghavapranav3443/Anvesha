"""Point-conditioned grounding: given a click point, predict the box of the
object/region it lands in. Trained on VRSBench objects (point = jittered box
centre). Powers the click-to-query UI feature.
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
from satquery.models.backbone import SceneEncoder, resize_np, to_tensor
from scripts.train_grounding import (GroundingHead, box_to_heat_target,
                                     heat_to_box)


class PointBoxDataset(Dataset):
    """VRSBench objects: point = box centre + jitter; target = the box."""

    def __init__(self, ann_dir: Path, images_root: Path, image_size=128,
                 limit=None, split_frac=0.9):
        import json
        self.items = []
        for jf in sorted(Path(ann_dir).glob("*.json")):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for obj in d.get("objects", []):
                coord = obj.get("obj_coord")
                if not coord or len(coord) != 4:
                    continue
                self.items.append((jf.stem, [float(v) for v in coord]))
                if limit and len(self.items) >= limit:
                    break
        self.images_root = Path(images_root)
        self.image_size = image_size
        n_train = int(len(self.items) * split_frac)
        self.train_items = self.items[:n_train]
        self.val_items = self.items[n_train:]

    def _image(self, stem):
        from PIL import Image
        for ext in (".png", ".jpg", ".jpeg"):
            f = self.images_root / f"{stem}{ext}"
            if f.exists():
                return np.asarray(Image.open(f).convert("RGB"),
                                  dtype=np.float32) / 255.0
        return None

    def sample(self, items, i):
        from PIL import Image
        stem, coord = items[i]
        arr = self._image(stem)
        if arr is None:
            arr = np.random.rand(128, 128, 3).astype(np.float32) * 0.3
        x = resize_np(arr, self.image_size)
        x0, y0, x1, y1 = coord
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 0.02), max(y1 - y0, 0.02)
        # jittered click point inside the box
        jx = cx + (np.random.rand() - 0.5) * w * 0.6
        jy = cy + (np.random.rand() - 0.5) * h * 0.6
        grid = self.image_size // 8
        tgt = box_to_heat_target([cx, cy, w, h], grid=grid)
        pt = np.array([jx, jy], dtype=np.float32)
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(pt), torch.tensor([cx, cy, w, h]),
                torch.from_numpy(tgt))

    def train_loader(self, batch_size):
        return DataLoader([self.sample(self.train_items, i)
                           for i in range(len(self.train_items))][:0] or
                          _Range(self, self.train_items),
                          batch_size=batch_size, shuffle=True, drop_last=True)


class _Range(Dataset):
    def __init__(self, owner, items):
        self.owner, self.items = owner, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.owner.sample(self.items, i)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    ds = PointBoxDataset(root / "Annotations_val", root / "Images_val",
                         image_size=args.image_size, limit=args.max_items)
    print(f"point-box train={len(ds.train_items)} val={len(ds.val_items)} "
          f"device={device}", flush=True)

    enc = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    head = GroundingHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    tr_items = ds.train_items
    va_items = ds.val_items

    def batch(items, idx):
        xs, pts, heats, ys = [], [], [], []
        for i in idx:
            x, pt, yb, ht = ds.sample(items, i)
            xs.append(x); pts.append(pt); heats.append(ht); ys.append(yb)
        return (torch.stack(xs).to(device), torch.stack(pts).to(device),
                torch.stack(ys).to(device), torch.stack(heats).to(device))

    best = -1.0
    tmp = CONFIG.weights_dir / "grounding_point.tmp.pt"
    g = args.image_size // 8
    for epoch in range(args.epochs):
        head.train(); tot = cnt = 0
        idx = np.random.permutation(len(tr_items))
        for b in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            x, pt, yb, heat_t = batch(tr_items, idx[b:b + args.batch_size])
            opt.zero_grad()
            with torch.no_grad():
                fmap = enc.feature_map(x, stride=8)
            # encode point as 2 normalised coords appended to a 510-d zero vec
            q = torch.zeros(x.shape[0], 512, device=device)
            q[:, 0] = pt[:, 0]; q[:, 1] = pt[:, 1]
            logits = head(fmap, q)
            loss = lossf(logits.squeeze(1), heat_t)
            loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        sched.step()

        head.eval(); hits = []
        with torch.no_grad():
            vi = np.random.permutation(len(va_items))[:400]
            for b in range(0, len(vi) - args.batch_size + 1, args.batch_size):
                x, pt, yb, _ = batch(va_items, vi[b:b + args.batch_size])
                fmap = enc.feature_map(x, stride=8)
                q = torch.zeros(x.shape[0], 512, device=device)
                q[:, 0] = pt[:, 0]; q[:, 1] = pt[:, 1]
                heat = torch.sigmoid(head(fmap, q)[:, 0]).cpu().numpy()
                for hm, yb_i in zip(heat, yb):
                    pb = torch.from_numpy(heat_to_box(hm))[None]
                    px0, py0 = pb[0, 0] - pb[0, 2] / 2, pb[0, 1] - pb[0, 3] / 2
                    px1, py1 = pb[0, 0] + pb[0, 2] / 2, pb[0, 1] + pb[0, 3] / 2
                    tx0, ty0 = yb_i[0] - yb_i[2] / 2, yb_i[1] - yb_i[3] / 2
                    tx1, ty1 = yb_i[0] + yb_i[2] / 2, yb_i[1] + yb_i[3] / 2
                    inter = max(0.0, float(min(px1, tx1) - max(px0, tx0))) * \
                        max(0.0, float(min(py1, ty1) - max(py0, ty0)))
                    union = float((px1 - px0) * (py1 - py0) +
                                  (tx1 - tx0) * (ty1 - ty0)) - inter
                    hits.append(inter / max(union, 1e-6) > 0.5)
        hit = float(np.mean(hits)) if hits else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(cnt,1):.4f} "
              f"val_point_IoU>0.5={hit:.3f}", flush=True)
        if hit > best:
            best = hit
            torch.save({"head": head.state_dict(), "val_hit": best,
                        "source": "vrsbench-points"}, tmp)
    tmp.replace(CONFIG.weights_dir / "grounding_point.pt")
    print("saved grounding_point.pt best", round(best, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "vrsbench"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-items", type=int, default=14000)
    main(ap.parse_args())
