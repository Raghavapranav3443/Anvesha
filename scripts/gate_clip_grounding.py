"""Phase 1 spike — CLIP region-text grounding gate on VRSBench val.

Pre-registered adoption rule (decided BEFORE running, mirroring gate_dinov2.py):
  ADOPT iff VRSBench-val grounding IoU@0.5 >= 0.30 (2x the spectral baseline ~0.15).
  Skip / report if the CLIP checkpoint cannot be loaded.

Two loaders:

  --loader transformers  (default) generic OpenAI CLIP via transformers
      python scripts/gate_clip_grounding.py --n 50
      (openai/clip-vit-base-patch32 must be cached via cache_clip_weights.py)

  --loader open_clip     RS-domain checkpoints (RemoteCLIP class, open_clip format)
      pip install open_clip_torch           # once
      python scripts/gate_clip_grounding.py --n 50 --loader open_clip \\
          --arch ViT-B-32 --pretrained path\\to\\remoteclip_rsicd_vit_b_32.pt

Writes result to runs/clip_grounding_gate.json.
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


def load_clip(args, device):
    """Return (model, proc-or-tokenizer, loader-name)."""
    if args.loader == "open_clip":
        import open_clip
        model, preprocess, _ = open_clip.create_model_and_transforms(
            args.arch, pretrained=args.pretrained)
        model.to(device).eval()
        tok = open_clip.get_tokenizer(args.arch)
        return model, (tok, preprocess), "open_clip"
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(args.model).to(device).eval()
    p = CLIPProcessor.from_pretrained(args.model)
    return m, p, "transformers"


@torch.no_grad()
def region_scores(model, proc, loader, img, query, device, batch=64):
    """Score sliding-window regions of img against query via CLIP similarity.

    v2 harness correction (documented, pre-registered threshold unchanged):
    v1 only proposed 0.6/1.0-scale windows, which mathematically caps IoU@0.5
    below reach for small GT objects (a 0.6-scale window vs a 0.2-scale GT box
    tops out near IoU 0.11). v2 proposes a denser multi-scale grid and reports
    an oracle ceiling in main() so harness limits are separable from model
    capability. Threshold stays 0.30.
    """
    from PIL import Image
    from satquery.io_utils import _to_uint8_display
    # load_image returns float32 [0,1]; PIL needs uint8
    im = Image.fromarray(_to_uint8_display(np.asarray(img))).convert("RGB")
    W, H = im.size
    boxes = []
    for scale in (0.2, 0.35, 0.6, 1.0):
        rw, rh = int(W * scale), int(H * scale)
        if rw < 16 or rh < 16:
            continue
        step = max(8, int(min(rw, rh) * 0.4))
        for y in range(0, H - rh + 1, step):
            for x in range(0, W - rw + 1, step):
                boxes.append((x, y, x + rw, y + rh))
    crops = [im.crop((x, y, x1, y1)) for (x, y, x1, y1) in boxes]
    if not crops:
        return [], []
    # encode the query text once
    if loader == "open_clip":
        tok, preprocess = proc
        t_emb = model.encode_text(tok([query]).to(device))
    else:
        tin = proc(text=[query], return_tensors="pt", padding=True).to(device)
        t_emb = model.get_text_features(**tin)
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    sims_all = []
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        if loader == "open_clip":
            imgs = torch.stack([preprocess(c) for c in chunk]).to(device)
            i_emb = model.encode_image(imgs)
        else:
            pin = proc(images=chunk, return_tensors="pt").to(device)
            i_emb = model.get_image_features(pixel_values=pin["pixel_values"])
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        sims_all.append((i_emb * t_emb).sum(dim=-1).cpu().numpy())
    sims = np.concatenate(sims_all)
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    return boxes, sims.tolist()


def _iou_norm(gt, box, H, W):
    x0, y0, x1, y1 = box
    bx = np.array([x0 / W, y0 / H, x1 / W, y1 / H], dtype=np.float32)
    inter = max(0.0, min(gt[2], bx[2]) - max(gt[0], bx[0])) * \
            max(0.0, min(gt[3], bx[3]) - max(gt[1], bx[1]))
    union = (max(gt[2], bx[2]) - min(gt[0], bx[0])) * \
            (max(gt[3], bx[3]) - min(gt[1], bx[1]))
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--loader", choices=("transformers", "open_clip"),
                    default="transformers")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--arch", default="ViT-B-32")
    ap.add_argument("--pretrained", default=None,
                    help="open_clip checkpoint path (file) for RS-domain models")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] loader={args.loader} loading "
          f"{args.model if args.loader == 'transformers' else args.arch}"
          f" on {device}", flush=True)
    try:
        model, proc, loader = load_clip(args, device)
    except Exception as e:
        print(f"[clip] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        return 1

    items = _vrsbench_val()
    if not items:
        print("[vrsbench] no items found", flush=True)
        return 1
    print(f"[vrsbench] {len(items)} items available, testing {args.n}", flush=True)

    ious, oracles, det = [], [], 0
    t0 = time.time()
    for it in items[:args.n]:
        from satquery.io_utils import load_image
        img = load_image(it["image"])
        arr = img.array
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] not in (3, 4):
            arr = arr[..., :3]
        H, Wc = arr.shape[:2]
        for o in it["objects"]:
            coord = _parse_coord(o.get("obj_coord"))
            if not coord:
                continue
            gt = np.array(coord, dtype=np.float32)
            expr = o.get("referring_sentence") or it["caption"]
            try:
                boxes, sims = region_scores(model, proc, loader, arr, expr, device)
            except Exception as e:
                print(f"[clip] skip {it['stem']}: {type(e).__name__}: {e}", flush=True)
                continue
            if not boxes:
                continue
            # oracle ceiling: best IoU this proposal grid could ever produce
            oracles.append(max(_iou_norm(gt, b, H, Wc) for b in boxes))
            best_idx = int(np.argmax(sims))
            iou = _iou_norm(gt, boxes[best_idx], H, Wc)
            ious.append(iou)
            if iou >= 0.5:
                det += 1
        print(f"  [{len(ious):4d}] IoU={np.mean(ious):.3f} "
              f"oracle={np.mean(oracles):.3f} det={det}/{len(ious)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    if not ious:
        print("[clip] no results", flush=True)
        return 1
    mean_iou = float(np.mean(ious))
    det_rate = det / len(ious)
    oracle = float(np.mean(oracles))
    gate = 0.30
    verdict = "ADOPT" if mean_iou >= gate else "REJECT"
    print(f"\n[clip] VRSBench-val grounding IoU@0.5 = {mean_iou:.4f} "
          f"({len(ious)} refs) det@0.5={det_rate:.3f}", flush=True)
    print(f"[clip] proposal-grid oracle IoU (ceiling) = {oracle:.4f}", flush=True)
    print(f"[clip] gate (>= {gate}) -> {verdict}", flush=True)

    out = CONFIG.runs_dir / "clip_grounding_gate.json"
    out.write_text(json.dumps({
        "loader": args.loader, "model": args.model, "arch": args.arch,
        "pretrained": args.pretrained, "device": device, "n": args.n,
        "mean_iou": round(mean_iou, 4), "det_rate": round(det_rate, 4),
        "oracle_iou": round(oracle, 4),
        "gate": gate, "verdict": verdict}, indent=2), encoding="utf-8")
    return 0 if verdict == "ADOPT" else 2


if __name__ == "__main__":
    sys.exit(main())