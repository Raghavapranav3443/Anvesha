"""Train a learned referring-expression grounding head on BigEarthNet.txt
'reference' bounding-box annotations joined to our local S2 patches."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder, normalise_for_encoder, resize_np


def bow(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def parse_box(s: str):
    nums = [float(x) for x in re.findall(r"\d*\.?\d+", s)]
    if len(nums) < 4:
        return None
    x0, y0, x1, y1 = nums[:4]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, 0.01), max(y1 - y0, 0.01)
    return np.array([cx, cy, w, h], dtype=np.float32)


class RefDataset(Dataset):
    def __init__(self, join_path: Path, s2_root: Path, image_size=120):
        import pandas as pd
        df = pd.read_parquet(join_path)
        self.items = []
        for _, r in df.iterrows():
            m = re.search(r"<ref>(.*?)</ref>", str(r["input"]), re.S)
            box = parse_box(str(r["output"]))
            f = Path(str(s2_root).format(split=str(r["split"]))) / \
                f"{r['patch_id']}.tif"
            if m and box is not None and f.exists():
                expr = m.group(1).strip()
                # strip size/shape modifiers from the concept but keep them
                # as separate tokens the model can condition on
                self.items.append((f, expr, str(r["input"]).lower(), box,
                                   r["split"]))
        self.image_size = image_size
        splits = sorted({it[4] for it in self.items})
        self.split_set = set(splits)

    def subset(self, split: str) -> "RefDataset":
        new = object.__new__(RefDataset)
        new.items = [it for it in self.items if it[4] == split]
        new.image_size = self.image_size
        new.split_set = self.split_set
        return new

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import rasterio
        f, _expr, full_text, box, _sp = self.items[i]
        with rasterio.open(f) as src:
            arr = np.moveaxis(src.read().astype(np.float32), 0, -1)
        rgb = np.clip(arr[..., [2, 1, 0]] / 10000.0, 0, 1)
        x = resize_np(rgb, self.image_size)
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(bow(full_text)),
                torch.from_numpy(box))


def box_to_heat_target(box, grid=15):
    """Gaussian blob covering the GT box on the grid."""
    cx, cy, w, h = box
    ys, xs = np.mgrid[0:grid, 0:grid]
    x0, x1 = (cx - w / 2) * grid, (cx + w / 2) * grid
    y0, y1 = (cy - h / 2) * grid, (cy + h / 2) * grid
    inside = ((xs + 0.5 >= x0) & (xs + 0.5 <= x1) &
              (ys + 0.5 >= y0) & (ys + 0.5 <= y1)).astype(np.float32)
    # soften edges with a distance falloff
    dist = np.maximum(np.maximum(x0 - xs - 0.5, xs + 0.5 - x1),
                      np.maximum(y0 - ys - 0.5, ys + 0.5 - y1)) / grid
    return np.clip(inside + (1 - inside) * np.exp(-dist * 18) * 0.35, 0, 1) \
        .astype(np.float32)


def heat_to_box(heat, thr=0.4):
    m = heat >= thr
    if not m.any():
        m = heat >= max(float(heat.max()) * 0.75, thr * 0.8)
    ys, xs = np.where(m)
    if len(xs) == 0:                       # completely flat heatmap
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    h, w = heat.shape
    x0, x1 = xs.min() / w, (xs.max() + 1) / w
    y0, y1 = ys.min() / h, (ys.max() + 1) / h
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0],
                    dtype=np.float32)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_all = RefDataset(CONFIG.data_dir / "bentxt_join" / "refs.parquet",
                        Path(args.s2_layout), args.image_size)
    tr = ds_all.subset("train"); va = ds_all.subset("validation")
    print(f"refs train={len(tr)} val={len(va)} device={device}", flush=True)

    enc = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    head = GroundingHead().to(device)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def iou_at(pred, tgt, thr=0.5):
        p = pred.detach().cpu(); t = tgt.cpu()
        px0, py0 = p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2
        px1, py1 = p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2
        tx0, ty0 = t[:, 0] - t[:, 2] / 2, t[:, 1] - t[:, 3] / 2
        tx1, ty1 = t[:, 0] + t[:, 2] / 2, t[:, 1] + t[:, 3] / 2
        ix0, iy0 = np.maximum(px0, tx0), np.maximum(py0, ty0)
        ix1, iy1 = np.minimum(px1, tx1), np.minimum(py1, ty1)
        inter = (np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None))
        union = ((px1 - px0) * (py1 - py0) + (tx1 - tx0) * (ty1 - ty0) - inter)
        inter = np.asarray(inter, dtype=np.float32)
        union = np.asarray(union, dtype=np.float32)
        iou = inter / np.maximum(union, 1e-6)
        return float((iou > thr).mean())

    best = -1.0
    tmp = CONFIG.weights_dir / "grounding.tmp.pt"
    lossf = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        head.train(); tot = cnt = 0
        for x, q, yb in dl_tr:
            x, q, yb = x.to(device), q.to(device), yb.to(device)
            with torch.no_grad():
                fmap = enc.feature_map(x, stride=8)
            logits = head(fmap, q)
            heat_t = torch.stack([
                torch.from_numpy(box_to_heat_target(b.tolist()))
                for b in yb]).to(device)
            loss = lossf(logits.squeeze(1), heat_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        sched.step()
        head.eval(); accs = []
        with torch.no_grad():
            for bi, (x, q, yb) in enumerate(dl_va):
                fmap = enc.feature_map(x.to(device), stride=8)
                heat = torch.sigmoid(head(fmap, q.to(device))[:, 0]).cpu().numpy()
                for hm, yb_i in zip(heat, yb):
                    pb = torch.from_numpy(heat_to_box(hm))[None]
                    accs.append(iou_at(pb, yb_i[None]))
                if bi >= 15:
                    break
        hit = float(np.mean(accs)) if accs else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(cnt,1):.4f} "
              f"val_IoU>0.5={hit:.3f}", flush=True)
        if hit > best:
            best = hit
            torch.save({"head": head.state_dict(), "val_hit": best}, tmp)
    tmp.replace(CONFIG.weights_dir / "grounding.pt")
    print("saved", CONFIG.weights_dir / "grounding.pt", "best", round(best, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-layout",
                    default=str(CONFIG.data_dir / "bigearthnet_14k/BEN_14k/BigEarthNet-S2/{split}"))
    ap.add_argument("--image-size", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    main(ap.parse_args())
