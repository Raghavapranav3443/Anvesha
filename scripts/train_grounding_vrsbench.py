"""Train the grounding head on VRSBench referring expressions (object
vocabulary: vehicles, ships, aircraft, storage tanks...).

  python scripts/train_grounding_vrsbench.py --epochs 8

Reads data/vrsbench/annotations_val.json + Images_val/<...>.jpg produced by
scripts/download_vrsbench.py. Reuses the heatmap head architecture.
"""
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

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor
from scripts.train_grounding import GroundingHead, box_to_heat_target, heat_to_box


class VRSBenchRefs(Dataset):
    """Per-image JSONs in Annotations_val/ with objects[].referring_sentence
    and objects[].obj_coord = [x0,y0,x1,y1] normalised to [0,1]."""

    def __init__(self, ann_dir: Path, images_root: Path, image_size=128,
                 limit=None):
        self.images_root = Path(images_root)
        self.image_size = image_size
        self.items = []
        files = sorted(Path(ann_dir).glob("*.json"))
        for jf in files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            stem = jf.stem
            for obj in d.get("objects", []):
                expr = obj.get("referring_sentence")
                coord = obj.get("obj_coord")
                if not expr or not coord or len(coord) != 4:
                    continue
                self.items.append((stem, str(expr),
                                   [float(v) for v in coord]))
                if limit and len(self.items) >= limit:
                    break

    def __len__(self):
        return len(self.items)

    def _image_file(self, stem: str):
        for ext in (".jpg", ".jpeg", ".png"):
            f = self.images_root / f"{stem}{ext}"
            if f.exists():
                return f
        hits = list(self.images_root.glob(f"{stem}.*"))
        return hits[0] if hits else None

    def __getitem__(self, i):
        from PIL import Image
        stem, expr, coord = self.items[i]
        f = self._image_file(stem)
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        x = resize_np(arr, self.image_size)
        x0, y0, x1, y1 = coord
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 0.02), max(y1 - y0, 0.02)
        grid = self.image_size // 8
        tgt = box_to_heat_target([cx, cy, w, h], grid=grid)
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.tensor([0.0]),
                torch.tensor([cx, cy, w, h]),
                torch.from_numpy(tgt))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    ann_dir = root / "Annotations_val"
    imgs = root / "Images_val"
    ds = VRSBenchRefs(ann_dir, imgs, limit=args.max_items)
    print(f"vrsbench refs: {len(ds)} device={device}", flush=True)
    if len(ds) < 100:
        print("not enough samples; aborting")
        return

    n_val = max(200, len(ds) // 10)
    ds_tr = torch.utils.data.Subset(ds, range(0, len(ds) - n_val))
    ds_va = torch.utils.data.Subset(ds, range(len(ds) - n_val, len(ds)))
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)

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

    def iou_eval():
        head.eval(); hits = []
        with torch.no_grad():
            for x, _, yb, _t in dl_va:
                heat = torch.sigmoid(head(enc.feature_map(x.to(device), stride=8),
                                          torch.zeros(x.shape[0], 512,
                                                      device=device))[:, 0]) \
                    .cpu().numpy()
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
        return float(np.mean(hits)) if hits else 0.0

    best = -1.0
    tmp = CONFIG.weights_dir / "grounding_vrsbench.tmp.pt"
    for epoch in range(args.epochs):
        head.train(); tot = cnt = 0
        for x, _q, yb, heat_t in dl:
            x, yb, heat_t = x.to(device), yb.to(device), heat_t.to(device)
            with torch.no_grad():
                fmap = enc.feature_map(x, stride=8)
            logits = head(fmap, torch.zeros(x.shape[0], 512, device=device))
            loss = lossf(logits.squeeze(1), heat_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        sched.step()
        hit = iou_eval()
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(cnt,1):.4f} "
              f"val_IoU>0.5={hit:.3f}", flush=True)
        if hit > best:
            best = hit
            torch.save({"head": head.state_dict(), "val_hit": best,
                        "source": "vrsbench"}, tmp)
    tmp.replace(CONFIG.weights_dir / "grounding_vrsbench.pt")
    print("saved grounding_vrsbench.pt best", round(best, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "vrsbench"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-items", type=int, default=12000)
    main(ap.parse_args())
