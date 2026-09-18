#!/usr/bin/env python3
"""Train a density-map counting head for RSVQA count-type questions (v5).

Design (total-count-supervised density regression):
  RSVQA provides only a total count per image - no point annotations. We use
  the standard count-only trick: a small conv head predicts a per-pixel
  density map whose SUM is trained (L1) to equal the true count. The density
  representation gives the head a spatial inductive bias that plain
  classification lacks, and an auxiliary ordinal-CE digit head (multi-task)
  provides a direct classification signal.

Gate: promote only if val digit-accuracy beats the ordinal v4 head (0.436).
The v4 checkpoint stays the shipped fallback on gate failure.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder
from anvesha.models.count_density import DensityHead
from scripts.train_count_head import CountDataset, OrdinalSoftCE
from scripts.train_vqa import RSVQADataset


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    base_tr = RSVQADataset(root, "train", image_size=args.image_size,
                           augment=True)
    base_va = RSVQADataset(root, "val", image_size=args.image_size,
                           type_vocab=base_tr.type_vocab)
    tr = CountDataset(base_tr)
    va = CountDataset(base_va, max_count_answer=9)
    n_classes = len(tr.classes)
    print(f"count train={len(tr)} val={len(va)} classes={tr.classes} "
          f"device={device}", flush=True)

    freq = collections.Counter(d for _, d in tr.items)
    sample_weights = [1.0 / freq[d] for _, d in tr.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr),
                                    replacement=True)
    dl = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                    num_workers=0, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size)

    encoder = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
    head = DensityHead(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) +
                            list(head.parameters()),
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ord_loss = OrdinalSoftCE(n_classes, sigma=0.7)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    classes_t = torch.tensor([float(c) for c in tr.classes],
                             device=device)[None, :]

    def val_acc():
        encoder.eval(); head.eval(); correct = seen = 0
        with torch.no_grad():
            for x, qb, t, y in dl_va:
                fmap = encoder.feature_map(x.to(device), stride=8)
                dmap, logits = head(fmap)
                dens = torch.nn.functional.softplus(dmap)
                pred_count = dens.sum(dim=(2, 3)).squeeze(1)   # B
                pred_d = torch.argmin(torch.abs(
                    pred_count[:, None] - classes_t), dim=1)
                pred_c = logits.argmax(1)
                for pd, pc, yi in zip(pred_d.tolist(), pred_c.tolist(),
                                      y.tolist()):
                    # density wins when it agrees with the digit head +-1
                    guess = pd if abs(tr.classes[pd] - tr.classes[pc]) <= 1 else pc
                    correct += int(tr.inv_lut[guess] == va.inv_lut[yi])
                    seen += 1
        return correct / max(seen, 1)

    best = 0.0
    patience = 0
    tmp = CONFIG.weights_dir / "count_head_density.pt.tmp"
    for epoch in range(args.epochs):
        encoder.train(); head.train(); seen = correct = 0
        for x, qb, t, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                fmap = encoder.feature_map(x, stride=8)
                dmap, logits = head(fmap)
                dens = torch.nn.functional.softplus(dmap)
                # count-only density supervision: rescale so sum == label
                s = dens.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                target = dens * (y.float().view(-1, 1, 1, 1) / s)
                loss_d = torch.nn.functional.l1_loss(dens, target.detach())
                loss_c = ord_loss(logits, y)
                loss = loss_d + args.digit_weight * loss_c
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
        sched.step()
        acc = val_acc()
        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_digit_acc={correct/max(seen,1):.4f} "
              f"val_digit_acc={acc:.4f}", flush=True)
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
                        "arch": "density_v5"}, tmp)
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ---- GATE: promote only if the density head beats the ordinal v4 head --
    GATE = 0.436
    final = CONFIG.weights_dir / "count_head_density.pt"
    if best > GATE:
        tmp.replace(final)
        verdict = f"PROMOTED (beats ordinal v4 gate {GATE})"
    else:
        tmp.unlink(missing_ok=True)
        verdict = f"NOT PROMOTED - gate {GATE} not met; ordinal v4 stays shipped"
    print(f"best val_digit_acc={best:.4f} -> {verdict}")

    from anvesha.experiment_log import log_experiment
    log_experiment(
        script="train_count_density_v5",
        args={"image_size": args.image_size, "epochs": args.epochs,
              "lr": args.lr, "batch_size": args.batch_size,
              "digit_weight": args.digit_weight, "early_stop": args.early_stop,
              "gate": GATE},
        metrics={"val_digit_acc": best},
        checkpoint=str(final) if best > GATE else "",
        notes=verdict,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--digit-weight", type=float, default=1.0,
                    help="weight of the auxiliary ordinal digit loss")
    ap.add_argument("--early-stop", type=int, default=8)
    main(ap.parse_args())