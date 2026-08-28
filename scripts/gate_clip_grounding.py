"""Phase 1 spike — CLIP region-text grounding gate on VRSBench val.

Pre-registered adoption rule (decided BEFORE running, mirroring gate_dinov2.py):
  ADOPT iff VRSBench-val grounding IoU@0.5 >= 0.30 (2x the spectral baseline 0.15).
  Skip / report if the CLIP checkpoint cannot be loaded offline.

Usage: python scripts/gate_clip_grounding.py [--n 50] [--model openai/clip-vit-base-patch32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.config import CONFIG  # noqa: E402
from satquery.evaluate import _parse_coord, _vrsbench_val  # noqa: E402


def load_clip(model_name: str, device: str):
    """Load CLIP via transformers (already a dependency). Returns (model, proc)."""
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(model_name).to(device).eval()
    p = CLIPProcessor.from_pretrained(model_name)
    return m, p


@torch.no_grad()
def region_scores(m, p, img, query, device):
    """Score a set of sliding-window regions of img against query via CLIP."""
    from PIL import Image
    im = Image.fromarray(img).convert("RGB")
    W, H = im.size
    # region proposals: sparse multiscale sliding windows (fast spike)
    boxes = []
    for scale in (0.6, 1.0):
        rw, rh = int(W * scale), int(H * scale)
        step = max(16, int(min(rw, rh) * 0.33))
        for y in range(0, H - rh + 1, step):
            for x in range(0, W - rw + 1, step):
                boxes.append((x, y, x + rw, y + rh))
    crops = [im.crop((x, y, x1, y1)) for (x, y, x1, y1) in boxes]
    if not crops:
        return [], []
    inp = p(text=[query] * len(crops), images=crops, return_tensors="pt", padding=True).to(device)
    out = m(**inp)
    sims = (out.text_embeds * out.image_embeds).sum(dim=-1).cpu().numpy()
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    return boxes, sims.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] loading {args.model} on {device}", flush=True)
    try:
        m, p = load_clip(args.model, device)
    except Exception as e:
        print(f"[clip] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        return 1

    items = _vrsbench_val()
    if not items:
        print("[vrsbench] no items found", flush=True)
        return 1
    print(f"[vrsbench] {len(items)} items available, testing {args.n}", flush=True)

    ious, det = [], 0
    t0 = time.time()
    for it in items[:args.n]:
        from satquery.io_utils import load_image
        from PIL import Image
        img = load_image(it["image"])
        arr = np.asarray(img.rgb if hasattr(img, "rgb") else img.array)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        for o in it["objects"]:
            coord = _parse_coord(o.get("obj_coord"))
            if not coord:
                continue
            gt = np.array(coord, dtype=np.float32)
            expr = o.get("referring_sentence") or it["caption"]
            try:
                boxes, sims = region_scores(m, p, arr, expr, device)
            except Exception as e:
                print(f"[clip] skip {it['stem']}: {type(e).__name__}: {e}", flush=True)
                continue
            if not boxes:
                continue
            best_idx = int(np.argmax(sims))
            x0, y0, x1, y1 = boxes[best_idx]
            H, Wc = arr.shape[:2]
            bx = np.array([x0 / Wc, y0 / H, x1 / Wc, y1 / H], dtype=np.float32)
            inter = max(0.0, min(gt[2], bx[2]) - max(gt[0], bx[0])) * \
                    max(0.0, min(gt[3], bx[3]) - max(gt[1], bx[1]))
            union = (max(gt[2], bx[2]) - min(gt[0], bx[0])) * \
                    (max(gt[3], bx[3]) - min(gt[1], bx[1]))
            iou = inter / union if union > 0 else 0.0
            ious.append(iou)
            if iou >= 0.5:
                det += 1
        print(f"  [{len(ious):4d}] IoU={np.mean(ious):.3f} det={det}/{len(ious)} elapsed={time.time()-t0:.0f}s", flush=True)

    if not ious:
        print("[clip] no results", flush=True)
        return 1
    mean_iou = float(np.mean(ious))
    det_rate = det / len(ious)
    gate = 0.30
    verdict = "ADOPT" if mean_iou >= gate else "REJECT"
    print(f"\n[clip] VRSBench-val grounding IoU@0.5 = {mean_iou:.4f} ({len(ious)} refs) "
          f"det@0.5={det_rate:.3f}", flush=True)
    print(f"[clip] gate (>= {gate}) -> {verdict}", flush=True)

    # write result
    out = CONFIG.runs_dir / "clip_grounding_gate.json"
    out.write_text(json.dumps({"model": args.model, "device": device, "n": args.n,
                               "mean_iou": round(mean_iou, 4), "det_rate": round(det_rate, 4),
                               "gate": gate, "verdict": verdict}, indent=2), encoding="utf-8")
    return 0 if verdict == "ADOPT" else 2


if __name__ == "__main__":
    import torch  # noqa: F401  (module-level for @torch.no_grad decorator)
    sys.exit(main())