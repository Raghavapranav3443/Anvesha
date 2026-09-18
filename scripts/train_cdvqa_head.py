#!/usr/bin/env python3
"""Train a change-conditioned CDVQA head (learned, gated).

Motivation: the shipped CDVQA predictor is a rule-based reasoner over
spectral-presence deltas + the change detector's area fraction. It sits
1.5 pts below the majority baseline on the full test set. This script
trains a *learned* head conditioned on the change detector's difference
features (the same stride-8/stride-16 diff the FPN-lite decoder consumes),
the spectral deltas, and the question BOW embedding - one output head per
CDVQA question type (mirrors the RSVQA per-type specialist pattern).

Gate: promote only if val accuracy beats the calibrated rule-based
predictor (0.4942). The rule-based predictor stays the shipped default
until this gate passes - the demo is never exposed to a regression.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from anvesha.config import CONFIG

CDVQA_CATS = ["buildings", "water", "trees", "low_vegetation", "NVG_surface"]
QTYPES = ["change_or_not", "increase_or_not", "decrease_or_not",
          "change_ratio", "change_ratio_types", "change_to_what",
          "largest_change", "smallest_change"]


def bow(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


class ChangeCondCDVQA(nn.Module):
    """Per-type answer heads over [diff-pooled | spectral deltas | q-BOW]."""

    def __init__(self, diff_dim: int = 256, n_deltas: int = 5,
                 bow_dim: int = 512, d: int = 256):
        super().__init__()
        self.diff_proj = nn.Linear(diff_dim, d)
        self.delta_proj = nn.Linear(n_deltas, 32)
        self.q_proj = nn.Linear(bow_dim, d)
        self.heads = nn.ModuleDict({
            qt: nn.Sequential(nn.Linear(d + 32 + d, 256), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(256, n_ans))
            for qt, n_ans in [
                ("change_or_not", 2), ("increase_or_not", 2),
                ("decrease_or_not", 2), ("change_ratio", 10),
                ("change_ratio_types", 2), ("change_to_what", 5),
                ("largest_change", 5), ("smallest_change", 5)]
        })

    def forward(self, diff, deltas, q, qtype):
        z = torch.cat([torch.relu(self.diff_proj(diff)),
                       torch.relu(self.delta_proj(deltas)),
                       torch.relu(self.q_proj(q))], dim=1)
        return self.heads[qtype](z)


def build_records(split: str, max_pairs: int, cache: Path) -> list:
    """Extract (diff-pooled features, deltas, af) per pair once, cache to disk."""
    if cache.exists():
        print(f"  loading cached pair features from {cache.name}")
        return json.loads(cache.read_text())

    from anvesha.io_utils import load_image
    from anvesha.models.change import ChangeDetectorNet
    from scripts.eval_cdvqa import compute_fine_presence, _parse_cat

    base = CONFIG.data_dir / "CDVQA"
    imgs = json.loads((base / f"{split.capitalize()}_images.json").read_text())["images"]
    qs = json.loads((base / f"{split.capitalize()}_questions.json").read_text())["questions"]
    ans = json.loads((base / f"{split.capitalize()}_answers.json").read_text())["answers"]
    a_lookup = {a["id"]: a["answer"] for a in ans}
    i_lookup = {i["id"]: i["file_name"] for i in imgs}
    i2q = defaultdict(list)
    for q in qs:
        i2q[q["img_id"]].append(q)
    fn_to_ids = defaultdict(list)
    for iid in i2q:
        fn = i_lookup.get(iid, "")
        if fn:
            fn_to_ids[fn].append(iid)
    unique_fns = sorted(fn_to_ids.keys())[:max_pairs] if max_pairs > 0 \
        else sorted(fn_to_ids.keys())

    second = CONFIG.data_dir / "SECOND" / "SECOND_test"
    det = ChangeDetectorNet()
    records = []
    for k, fn in enumerate(unique_fns):
        p1, p2 = second / "im1" / fn, second / "im2" / fn
        if not p1.exists() or not p2.exists():
            continue
        try:
            ia, ib = load_image(p1), load_image(p2)
            diff_feat, af = det.difference_features(ia, ib)
        except Exception:
            continue

        def _rgb(img):
            a = img.array
            return a[..., :3].astype(float) if a.ndim == 3 and a.shape[2] >= 3 \
                else a.astype(float)

        pa = compute_fine_presence(_rgb(ia))
        pb = compute_fine_presence(_rgb(ib))
        deltas = [pb[c] - pa[c] for c in CDVQA_CATS]
        for iid in fn_to_ids[fn]:
            for q in i2q[iid]:
                gt = a_lookup.get(q["answers_ids"][0], "") if q["answers_ids"] else ""
                if not gt:
                    continue
                cat = _parse_cat(q["question"])
                records.append({
                    "qt": q["type"], "q": q["question"], "gt": gt,
                    "cat": cat or "", "deltas": deltas,
                    "diff": [round(float(v), 5) for v in diff_feat],
                    "af": round(float(af), 5),
                })
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(unique_fns)}] pairs -> {len(records)} records",
                  flush=True)
    cache.write_text(json.dumps(records))
    print(f"  cached {len(records)} records -> {cache.name}")
    return records


ANSWER_LUTS = {
    "change_or_not": ["no", "yes"],
    "increase_or_not": ["no", "yes"],
    "decrease_or_not": ["no", "yes"],
    "change_ratio": ["0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
                     "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100"],
    "change_ratio_types": ["0", "0_to_10"],
    "change_to_what": CDVQA_CATS,
    "largest_change": CDVQA_CATS,
    "smallest_change": CDVQA_CATS,
}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = build_records("val", args.max_pairs,
                       CONFIG.weights_dir / "cdvqa_feat_cache_val.json")
    if len(tr) < 500:
        print("not enough records; aborting")
        return
    # split off a held-out tail for validation (records are pair-grouped;
    # a random split would leak pair features across train/val)
    cut = int(len(tr) * 0.85)
    tr_ds, va_ds = tr[:cut], tr[cut:]
    print(f"train={len(tr_ds)} val={len(va_ds)} device={device}")

    model = ChangeCondCDVQA().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def to_tensor_row(r):
        d = torch.tensor(r["deltas"], dtype=torch.float32)
        f = torch.tensor(r["diff"], dtype=torch.float32)
        q = torch.from_numpy(bow(r["q"]))
        return f, d, q

    def eval_split(rows):
        model.eval()
        per_type = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for r in rows:
                f, d, q = to_tensor_row(r)
                qt = r["qt"]
                if qt not in ANSWER_LUTS:
                    continue
                logits = model(f.unsqueeze(0).to(device),
                               d.unsqueeze(0).to(device),
                               q.unsqueeze(0).to(device), qt)
                pred = ANSWER_LUTS[qt][int(logits.argmax(1))]
                per_type[qt][0] += int(pred.strip().lower() ==
                                       r["gt"].strip().lower())
                per_type[qt][1] += 1
        tot = sum(c for c, _ in per_type.values())
        ok = sum(c for c, _ in per_type.values() for c in [c])  # noqa
        ok = sum(v[0] for v in per_type.values())
        n = sum(v[1] for v in per_type.values())
        return ok / max(n, 1), {k: round(v[0] / max(v[1], 1), 3)
                                for k, v in per_type.items()}

    rows_by_type = collections.defaultdict(list)
    for r in tr_ds:
        if r["qt"] in ANSWER_LUTS:
            rows_by_type[r["qt"]].append(r)

    best = 0.0
    patience = 0
    tmp = CONFIG.weights_dir / "cdvqa_head.pt.tmp"
    for epoch in range(args.epochs):
        model.train()
        # balanced sampling across question types
        epoch_rows = []
        per_type_n = max(len(v) for v in rows_by_type.values())
        for qt, rows in rows_by_type.items():
            idxs = np.random.choice(len(rows), size=per_type_n, replace=True)
            epoch_rows.extend(rows[i] for i in idxs)
        np.random.shuffle(epoch_rows)
        seen = correct = 0
        for i in range(0, len(epoch_rows), args.batch_size):
            batch = epoch_rows[i:i + args.batch_size]
            opt.zero_grad()
            losses = []
            for r in batch:
                f, d, q = to_tensor_row(r)
                logits = model(f.unsqueeze(0).to(device),
                               d.unsqueeze(0).to(device),
                               q.unsqueeze(0).to(device), r["qt"])
                tgt = ANSWER_LUTS[r["qt"]].index(
                    r["gt"]) if r["gt"] in ANSWER_LUTS[r["qt"]] else None
                if tgt is None:
                    continue
                losses.append(torch.nn.functional.cross_entropy(
                    logits, torch.tensor([tgt], device=device)))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            seen += len(losses)
        sched.step()
        va, per_type = eval_split(va_ds)
        print(f"epoch {epoch+1}/{args.epochs} val_acc={va:.4f} {per_type}",
              flush=True)
        if va > best:
            best = va
            patience = 0
            torch.save({"model": model.state_dict(),
                        "val_acc": va,
                        "per_type": per_type,
                        "arch": "change_cond_v1"}, tmp)
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    GATE = 0.4942
    final = CONFIG.weights_dir / "cdvqa_head.pt"
    if best > GATE:
        tmp.replace(final)
        verdict = f"PROMOTED (beats calibrated rules gate {GATE})"
    else:
        tmp.unlink(missing_ok=True)
        verdict = (f"NOT PROMOTED - gate {GATE} not met; calibrated "
                   f"rule-based predictor stays shipped")
    print(f"best val_acc={best:.4f} -> {verdict}")

    from anvesha.experiment_log import log_experiment
    log_experiment(
        script="train_cdvqa_head",
        args={"epochs": args.epochs, "lr": args.lr,
              "batch_size": args.batch_size, "max_pairs": args.max_pairs,
              "gate": GATE},
        metrics={"val_acc": best},
        checkpoint=str(final) if best > GATE else "",
        notes=verdict,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--early-stop", type=int, default=6)
    main(ap.parse_args())