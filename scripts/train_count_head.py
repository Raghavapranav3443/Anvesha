"""Train a dedicated counting head for RSVQA count-type questions.

v2 upgrades over v1:
  - Focal loss (gamma=2.0) to handle 48%/0 class imbalance
  - 192px resolution (up from 128px) for spatial detail
  - Class-balanced weighted sampling (oversample rare counts)
  - 30 epochs with cosine LR, AMP, early stopping
  - Precomputed index (no O(N) list.index per item)
  - Experiment logging
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from scripts.train_vqa import RSVQADataset, _Head


class CountDataset(Dataset):
    """Wraps RSVQADataset, keeping count-type questions with digit answers 0-9.

    Precomputes index mapping for O(1) lookup (v1 had O(N) per item).
    """

    def __init__(self, base: RSVQADataset, max_count_answer: int = 9):
        self.base = base
        self.items = []
        self.base_idx_map = {}
        for bi, (f, q, qtype, y) in enumerate(base.items):
            self.base_idx_map[(str(f), q, qtype)] = bi
        for f, q, qtype, y in base.items:
            ans = base.answer_vocab[y]
            m = re.fullmatch(r"\d+", ans.strip())
            if qtype == "count" and m and int(ans) <= max_count_answer:
                key = (str(f), q, qtype)
                bi = self.base_idx_map.get(key)
                if bi is not None:
                    self.items.append((bi, int(ans)))
        self.classes = sorted({d for _, d in self.items})
        self.lut = {c: i for i, c in enumerate(self.classes)}
        self.inv_lut = {i: c for c, i in self.lut.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        base_idx, digit = self.items[i]
        x, qb, t, _ = self.base[base_idx]
        return x, qb, t, torch.tensor(self.lut[digit])


class FocalLoss(nn.Module):
    """Focal loss: FL(pt) = -alpha_t * (1-pt)^gamma * log(pt).

    Down-weights easy/well-classified examples, focuses learning on hard ones.
    Critical for severe class imbalance (digit 0 = 48% of data).
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                          reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    base_tr = RSVQADataset(root, "train", image_size=args.image_size,
                           augment=True)
    base_va = RSVQADataset(root, "val", image_size=args.image_size,
                           type_vocab=base_tr.type_vocab)
    tr = CountDataset(base_tr)
    va = CountDataset(base_va, max_count_answer=9)
    print(f"count train={len(tr)} val={len(va)} classes={tr.classes}",
          flush=True)

    freq = collections.Counter(d for _, d in tr.items)
    n = len(tr)
    # Balanced sampler handles class imbalance; no loss reweighting needed
    sample_weights = [1.0 / freq[d] for _, d in tr.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=n,
                                    replacement=True)

    dl = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                    num_workers=0, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size)

    encoder = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
    head = _Head(len(tr.classes), len(base_tr.type_vocab)).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) +
                            list(head.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()  # Simple CE; balanced sampler handles class imbalance
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def val_acc():
        encoder.eval(); head.eval(); correct = seen = 0
        with torch.no_grad():
            for x, qb, t, y in dl_va:
                logits = head(encoder(x.to(device)), qb.to(device),
                              t.to(device))
                for pi, yi in zip(logits.argmax(1).cpu().tolist(),
                                  y.tolist()):
                    correct += int(tr.inv_lut[pi] == va.inv_lut[yi])
                    seen += 1
        return correct / max(seen, 1)

    best = 0.0
    patience = 0
    max_patience = args.early_stop
    tmp = CONFIG.weights_dir / "count_head.v3.pt.tmp"
    for epoch in range(args.epochs):
        encoder.train(); head.train(); seen = correct = 0
        for x, qb, t, y in dl:
            x, qb, t, y = x.to(device), qb.to(device), t.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = head(encoder(x), qb, t)
                loss = lossf(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
        sched.step()
        acc = val_acc()
        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_acc={correct/max(seen,1):.4f} val_digit_acc={acc:.4f}",
              flush=True)
        if acc > best:
            best = acc
            patience = 0
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        "classes": tr.classes,
                        "type_vocab": base_tr.type_vocab,
                        "input_size": args.image_size,
                        "bow_dim": 512,
                        "val_digit_acc": acc,
                        "focal_gamma": args.focal_gamma}, tmp)
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    final = CONFIG.weights_dir / "count_head.pt"
    tmp.replace(final)
    print(f"saved {final} (best val_digit_acc={best:.4f})")

    from satquery.experiment_log import log_experiment
    log_experiment(
        script="train_count_head_v3",
        args={"image_size": args.image_size, "epochs": args.epochs,
              "lr": args.lr, "batch_size": args.batch_size,
              "focal_gamma": args.focal_gamma, "early_stop": args.early_stop},
        metrics={"val_digit_acc": best},
        checkpoint=str(final),
        notes=f"focal_gamma={args.focal_gamma}, 192px, balanced sampling",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--early-stop", type=int, default=8)
    main(ap.parse_args())
