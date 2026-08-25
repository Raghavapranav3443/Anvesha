"""Train a dedicated counting head for RSVQA count-type questions.

Count answers are small integers; a specialised classifier over the same
RS-adapted encoder substantially outperforms the general answer head on this
question type (~25% of RSVQA-LR).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from scripts.train_vqa import RSVQADataset, _Head, bow, evaluate


class CountDataset(Dataset):
    """Wraps RSVQADataset items, keeping count-type questions whose answers
    map to digit classes."""

    def __init__(self, base: RSVQADataset, max_count_answer: int = 9):
        self.base = base
        self.items = []
        for f, q, qtype, y in base.items:
            ans = base.answer_vocab[y]
            m = re.fullmatch(r"\d+", ans.strip())
            if qtype == "count" and m and int(ans) <= max_count_answer:
                self.items.append((f, q, qtype, int(ans)))
        self.classes = sorted({y for *_ , y in self.items})
        self.lut = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, q, qtype, digit = self.items[i]
        x, qb, t, _ = self.base[self.base.items.index((f, q, qtype,
                                                       self.base.vocab[str(digit)]))]
        return x, qb, t, torch.tensor(self.lut[digit])


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    base_tr = RSVQADataset(root, "train", image_size=args.image_size,
                           augment=True)
    base_va = RSVQADataset(root, "val", image_size=args.image_size,
                           type_vocab=base_tr.type_vocab)
    tr = CountDataset(base_tr)
    va = CountDataset(base_va, max_count_answer=99)
    print(f"count train={len(tr)} val={len(va)} classes={tr.classes}",
          flush=True)

    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size)

    encoder = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
    head = _Head(len(tr.classes), len(base_tr.type_vocab)).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    # exact-match eval on digit strings
    def val_acc():
        encoder.eval(); head.eval(); correct = seen = 0
        with torch.no_grad():
            for x, qb, t, y in dl_va:
                logits = head(encoder(x.to(device)), qb.to(device), t.to(device))
                pred_digit = tr.classes[int(logits.argmax(1)[0])]
                for pi, yi in zip(logits.argmax(1).cpu().tolist(), y.tolist()):
                    pred_digit = tr.classes[pi]
                    gt_digit = va.classes[yi]
                    correct += int(pred_digit == gt_digit)
                    seen += 1
        return correct / max(seen, 1)

    best = 0.0
    tmp = CONFIG.weights_dir / "count_head.tmp.pt"
    for epoch in range(args.epochs):
        encoder.train(); head.train(); seen = correct = 0
        for x, qb, t, y in dl:
            x, qb, t, y = x.to(device), qb.to(device), t.to(device), y.to(device)
            opt.zero_grad()
            logits = head(encoder(x), qb, t)
            loss = lossf(logits, y)
            loss.backward(); opt.step()
            correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
        sched.step()
        acc = val_acc()
        print(f"epoch {epoch+1}/{args.epochs} train_acc={correct/max(seen,1):.4f} "
              f"val_digit_acc={acc:.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        "classes": tr.classes,
                        "type_vocab": base_tr.type_vocab,
                        "input_size": args.image_size,
                        "bow_dim": 512,
                        "val_digit_acc": acc}, tmp)
    tmp.replace(CONFIG.weights_dir / "count_head.pt")
    print("saved", CONFIG.weights_dir / "count_head.pt", "best", round(best, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    main(ap.parse_args())
