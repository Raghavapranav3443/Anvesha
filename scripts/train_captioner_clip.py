"""Phase 2b: captioner A/B gate -- CLIP vision features vs SceneEncoder features.

Pre-registered gate (written before running, mirroring train_type_heads_clip.py):
  * Two stacks trained side-by-side on IDENTICAL data, split, vocab, epochs,
    batch size, lr, and dataloader order (same seeded generator):
      control: Captioner(in_ch=128) on frozen RS-adapted SceneEncoder stride-8
               features (the production conditioning).
      clip:    Captioner(in_ch=768) on frozen CLIP ViT-B/32 patch tokens.
  * Metric: greedy-decode simple_bleu on the same validation split, best epoch.
  * PROMOTE the CLIP stack to weights/captioner.pt only if
        clip_best > control_best
    (both recorded in runs/captioner_gate.json). The production checkpoint is
    never touched during training; on loss, nothing changes and the negative
    is logged.
  * Budget: one run. No re-rolls from the same drawing board.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder
from scripts.train_captioner import (CaptionDataset, CaptionVocab, Captioner,
                                     collate_pad, simple_bleu)


def _val_bleu(model, dl_va, vocab, feat_fn, device, max_batches=12):
    model.eval()
    bleus = []
    ignore, sos, eos = 0, 1, 2
    with torch.no_grad():
        for bi, (x, tgt, L, plan) in enumerate(dl_va):
            feats = feat_fn(x.to(device))
            preds = model.generate(feats, vocab)
            for pr, row in zip(preds, tgt.tolist()):
                ref_ids = [t for t in row if t not in (ignore, sos, eos)]
                ref = " ".join(vocab.itos[t] for t in ref_ids)
                bleus.append(simple_bleu(pr, ref))
            if bi + 1 >= max_batches:
                break
    return float(np.mean(bleus)) if bleus else 0.0


def _train_stack(tag, model, feat_fn, dl, dl_va, vocab, device, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ignore = vocab.pad
    pos_w = torch.tensor([2.0] * Captioner.N_PLAN, device=device)
    best = -1.0
    for epoch in range(epochs):
        model.train()
        tot = cnt = 0
        for x, tgt, L, plan in dl:
            x, tgt, plan = x.to(device), tgt.to(device), plan.to(device)
            feats = feat_fn(x)
            logits = model(feats, tgt[:, :-1], plan)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1),
                ignore_index=ignore)
            if model.cond:
                pooled = feats.mean(dim=(2, 3))
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    model.plan_head(pooled), plan, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            cnt += 1
        sched.step()
        b = _val_bleu(model, dl_va, vocab, feat_fn, device)
        best = max(best, b)
        print(f"[{tag}] epoch {epoch+1}/{epochs} loss={tot/max(cnt,1):.4f} "
              f"val_BLEU={b:.4f}", flush=True)
    return best


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--data",
                    default=str(CONFIG.data_dir / "bigearthnet_14k" / "BEN_14k"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    root = Path(args.data)

    # identical vocab/datasets as the production trainer
    import pandas as pd
    cap_file = root / "captions.parquet"
    if not cap_file.exists():
        cap_file = root / "BigEarthNet-S2" / "captions.parquet"
    df = pd.read_parquet(cap_file)
    train_texts = df[df.split == "train"]["output"].dropna().tolist()
    word_freq = collections.Counter()
    for t in train_texts:
        for tok in re.findall(r"[a-z0-9]+", str(t).lower()):
            word_freq[tok] += 1
    keep = [w for w, c in word_freq.most_common(5000) if c >= 3]
    vocab = CaptionVocab(keep)

    mi = args.max_items or None
    ds_tr = CaptionDataset(root, "train", vocab, max_items=mi, image_size=120)
    ds_va = CaptionDataset(root, "validation", vocab, max_items=300,
                           image_size=120)
    print(f"train={len(ds_tr)} val={len(ds_va)} vocab={len(vocab)}", flush=True)

    g = torch.Generator()
    g.manual_seed(1234)  # identical loader order for both stacks
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    generator=g, drop_last=True, collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size,
                       collate_fn=collate_pad)

    enc = SceneEncoder(3).to(device).eval()
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)

    def enc_feat(x):
        with torch.no_grad():
            return enc.feature_map(x, stride=8)

    clip = None
    try:
        from anvesha.models.clip_text import get_clip_text
        clip = get_clip_text()
    except Exception:
        clip = None

    def clip_feat(x):
        with torch.no_grad():
            return clip.vision_patch_tokens(x)

    m_ctl = Captioner(len(vocab), cond=True, in_ch=128).to(device)
    b_ctl = _train_stack("ctl", m_ctl, enc_feat, dl, dl_va, vocab, device,
                         args.epochs, args.lr)

    if clip is not None:
        m_clip = Captioner(len(vocab), cond=True, in_ch=768).to(device)
        b_clip = _train_stack("clip", m_clip, clip_feat, dl, dl_va, vocab,
                              device, args.epochs, args.lr)
    else:
        print("[clip] CLIP unavailable -- control-only run, no promotion")
        b_clip = -1.0

    gate = {"control_val_bleu": round(b_ctl, 4),
            "clip_val_bleu": round(b_clip, 4) if b_clip >= 0 else None,
            "adopted": bool(b_clip > b_ctl)}
    print(f"[gate] control={b_ctl:.4f} clip={b_clip:.4f} -> "
          f"{'PROMOTED' if gate['adopted'] else 'REJECTED (control stands)'}")

    if gate["adopted"]:
        ck = {"model": m_clip.state_dict(), "vocab": vocab.itos,
              "val_bleu": b_clip, "cond": True,
              "feat_kind": "clip", "in_ch": 768}
        tmp = CONFIG.weights_dir / "captioner.tmp.pt"
        torch.save(ck, tmp)
        tmp.replace(CONFIG.weights_dir / "captioner.pt")
        print("PROMOTED -> weights/captioner.pt")

    CONFIG.runs_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.artifact("captioner_gate.json").write_text(
        json.dumps(gate, indent=1), encoding="utf-8")
    print("logged runs/captioner_gate.json")


if __name__ == "__main__":
    main()
