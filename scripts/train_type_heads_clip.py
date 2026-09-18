"""Phase 2 A/B trainer: CLIP text features vs hash-BOW for VQA type heads.

Pre-registered gate (written before running; DINOv2/CLIP-gate discipline):
Both stacks (BOW control, CLIP) train on identical data, identical 90/10
train/val split (seed 42), identical hyperparameters, identical frozen
SceneEncoder image features. Features are precomputed once — equivalent to
the production trainer's no_grad regime (its optimizer includes encoder
params but no_grad means they receive no gradient). Two deviations from the
production run, applied equally to BOTH stacks so the A/B stays fair:
features use unaugmented pixels (cleaner than per-epoch augmentation), and
model selection is by held-out val mean-per-type accuracy.
The CLIP stack REPLACES weights/type_heads.pt only if:
  gate 1: CLIP val mean-per-type accuracy > BOW val mean-per-type accuracy
  gate 2: CLIP val presence accuracy >= BOW presence accuracy - 0.005
Otherwise the production type_heads.pt is left untouched. The full A/B is
logged to runs/phase2_vqa_gate.json either way.
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

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder
from anvesha.models.vqa import _FusionHead, _hashed_bow
from anvesha.models.clip_text import get_clip_text
from scripts.train_vqa import RSVQADataset


class CORALHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.k_minus_1 = n_classes - 1
        self.fc = nn.Linear(feat_dim, 1, bias=False)
        self.a = nn.Parameter(torch.zeros(self.k_minus_1))
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(self, z):
        return self.fc(z) + self.a


def coral_loss(logits, ranks, importance):
    return -(ranks * nn.functional.logsigmoid(logits) +
             (1 - ranks) * nn.functional.logsigmoid(-logits) * importance
             ).sum(dim=1).mean()


def digit_ranks(digits, k_minus_1):
    d = digits.unsqueeze(1)
    return (torch.arange(k_minus_1, device=d.device).unsqueeze(0) < d).float()


def _px(f: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(f).convert("RGB").resize((128, 128)),
                      dtype=np.float32) / 255.0


def precompute_encoder_features(files, device, batch=128):
    """Frozen warm-started SceneEncoder features for every unique image."""
    enc = SceneEncoder(3).to(device).eval()
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
        print("encoder warm-started from RS-adapted scene encoder", flush=True)
    feats = {}
    uniq = sorted(set(files))
    with torch.no_grad():
        for s in range(0, len(uniq), batch):
            chunk = uniq[s:s + batch]
            xs = torch.stack([
                torch.from_numpy(_px(f).transpose(2, 0, 1))
                for f in chunk]).to(device)
            out = enc(xs).float().cpu().numpy()
            for f, v in zip(chunk, out):
                feats[f] = v
    print(f"precomputed {len(feats)} image features", flush=True)
    return feats, enc


def build_items(args):
    """Production-identical item builder (unaugmented) + image-level split."""
    base = RSVQADataset(Path(args.data), "train",
                        image_size=args.image_size, augment=False)
    items = []
    for f, q, qt, y in base.items:
        digit = None
        if qt == "count":
            m = re.fullmatch(r"\d+", base.answer_vocab[y].strip())
            if m and int(base.answer_vocab[y]) < 10:
                digit = int(base.answer_vocab[y])
        items.append((f, q, qt, y, digit))

    type_answers = collections.defaultdict(set)
    for f, q, qt, y, digit in items:
        if qt == "count":
            if digit is not None:
                type_answers[qt].add(digit)
        else:
            type_answers[qt].add(y)
    vocab_sizes = {t: len(s) for t, s in sorted(type_answers.items())
                   if t != "count"}
    local_lut = {t: {y: i for i, y in enumerate(sorted(s))}
                 for t, s in type_answers.items() if t != "count"}

    by_type = collections.defaultdict(list)
    for i, (f, q, qt, y, digit) in enumerate(items):
        if qt == "count" and digit is not None:
            by_type["count"].append(i)
        elif qt in local_lut and y in local_lut[qt]:
            by_type[qt].append(i)

    # Image-level 90/10 split (seed 42) — no question of the same image leaks
    # between train and val. Identical for both stacks (fair A/B).
    uniq = sorted({str(it[0]) for it in items})
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(0.1 * len(uniq)))
    val_files = {uniq[i] for i in perm[:n_val]}
    tr = [i for i, it in enumerate(items) if str(it[0]) not in val_files]
    va = [i for i, it in enumerate(items) if str(it[0]) in val_files]
    return base, items, vocab_sizes, local_lut, by_type, tr, va


def question_features(kind: str, questions, bow_dim: int = 512):
    """kind='bow' -> hash-BOW dict; kind='clip' -> CLIP text dict (or None)."""
    if kind == "bow":
        return {q: _hashed_bow(q, dim=bow_dim) for q in set(questions)}
    enc = get_clip_text()
    if enc is None:
        return None
    uniq = list(dict.fromkeys(questions))
    mat = enc.embed(uniq)
    return {q: mat[i] for i, q in enumerate(uniq)}


def make_evaluator(heads, coral):
    """Val accuracy per type; image features computed on demand and cached."""
    def evaluate(encoder, feats, qfeats, items, by_type, local_lut, va, device):
        import torch
        for h in heads.values():
            h.eval()
        coral.eval()
        accs = collections.defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for s in range(0, len(va), 256):
                idx = va[s:s + 256]
                miss = [items[i][0] for i in idx if items[i][0] not in feats]
                if miss:
                    xs = torch.stack([torch.from_numpy(
                        _px(str(f)).transpose(2, 0, 1)) for f in miss]).to(device)
                    out = encoder(xs).float().cpu().numpy()
                    for f, v in zip(miss, out):
                        feats[f] = v
                x = torch.from_numpy(
                    np.stack([feats[items[i][0]] for i in idx])).to(device)
                qb = torch.from_numpy(
                    np.stack([qfeats[items[i][1]] for i in idx])).to(device)
                for t_name, head in heads.items():
                    sel = [j for j, i in enumerate(idx)
                           if items[i][2] == t_name
                           and items[i][3] in local_lut[t_name]]
                    if not sel:
                        continue
                    logits = head(x[sel], qb[sel],
                                  torch.zeros(len(sel), dtype=torch.long,
                                              device=device))
                    ys = torch.tensor([local_lut[t_name][items[idx[j]][3]]
                                       for j in sel], device=device)
                    accs[t_name][0] += (logits.argmax(1) == ys).sum().item()
                    accs[t_name][1] += len(sel)
                sel_c = [j for j, i in enumerate(idx)
                         if items[i][2] == "count" and items[i][4] is not None]
                if sel_c:
                    digits = torch.tensor([items[idx[j]][4] for j in sel_c],
                                          device=device)
                    pred = (torch.sigmoid(coral(x[sel_c])) > 0.5).sum(1)
                    accs["count"][0] += (pred == digits).sum().item()
                    accs["count"][1] += len(sel_c)
        return {t: c[0] / max(c[1], 1)
                for t, c in sorted(accs.items()) if c[1]}
    return evaluate


def train_stack(kind, qfeats, items, by_type, vocab_sizes, local_lut,
                feats, tr, va, encoder, device, args):
    """Train heads+CORAL on frozen features; select by val mean-per-type."""
    import torch
    torch.manual_seed(42)
    heads = {t: _FusionHead(n, 1).to(device) for t, n in vocab_sizes.items()}
    coral = CORALHead(256, 10).to(device)
    params = [p for h in heads.values() for p in h.parameters()] + \
        list(coral.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    c_counts = collections.Counter(items[i][4] for i in by_type["count"])
    imp = torch.tensor([1.0 / np.sqrt(c_counts.get(k, 0) + 1) for k in range(9)])
    imp = (imp / imp.mean()).to(device)
    evaluate = make_evaluator(heads, coral)

    rng = np.random.RandomState(123)
    best = {"mean": -1.0, "accs": None, "state": None}
    for epoch in range(args.epochs):
        for h in heads.values():
            h.train()
        coral.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), args.batch_size):
            idx = [tr[j] for j in order[s:s + args.batch_size]]
            x = torch.from_numpy(
                np.stack([feats[items[i][0]] for i in idx])).to(device)
            qb = torch.from_numpy(
                np.stack([qfeats[items[i][1]] for i in idx])).to(device)
            opt.zero_grad()
            total = 0.0
            for t_name, head in heads.items():
                sel = [j for j, i in enumerate(idx)
                       if items[i][2] == t_name
                       and items[i][3] in local_lut[t_name]]
                if not sel:
                    continue
                logits = head(x[sel], qb[sel],
                              torch.zeros(len(sel), dtype=torch.long,
                                          device=device))
                ys = torch.tensor([local_lut[t_name][items[idx[j]][3]]
                                   for j in sel], device=device)
                total = total + ce(logits, ys)
            sel_c = [j for j, i in enumerate(idx)
                     if items[i][2] == "count" and items[i][4] is not None]
            if sel_c:
                digits = torch.tensor([items[idx[j]][4] for j in sel_c],
                                      device=device)
                total = total + coral_loss(coral(x[sel_c]),
                                           digit_ranks(digits, 9), imp)
            if torch.is_tensor(total):
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        sched.step()
        accs = evaluate(encoder, feats, qfeats, items, by_type, local_lut,
                        va, device)
        mean = float(np.mean(list(accs.values()))) if accs else 0.0
        print(f"[{kind}] epoch {epoch+1}/{args.epochs} val_mean={mean:.4f} "
              + " ".join(f"{t}={a:.3f}" for t, a in sorted(accs.items())),
              flush=True)
        if mean > best["mean"]:
            best = {"mean": mean, "accs": accs,
                    "state": ({t: {k: v.detach().cpu().clone()
                                   for k, v in h.state_dict().items()}
                               for t, h in heads.items()},
                              {k: v.detach().cpu().clone()
                               for k, v in coral.state_dict().items()})}
    return best


# ==== PART3 ====
# ==== PART3 ====


def main(args):
    import torch
    device = CONFIG.resolve_device()
    base, items, vocab_sizes, local_lut, by_type, tr, va = build_items(args)
    print(f"items={len(items)} train={len(tr)} val={len(va)} "
          f"types={sorted(by_type)}", flush=True)

    files = [it[0] for it in items]
    feats, encoder = precompute_encoder_features(files, device)
    questions = list(dict.fromkeys(it[1] for it in items))

    results: dict = {}
    best_stacks: dict = {}
    for kind in ("bow", "clip"):
        qfeats = question_features(kind, questions)
        if qfeats is None:
            print(f"[{kind}] unavailable -> stack skipped", flush=True)
            results[kind] = None
            continue
        best = train_stack(kind, qfeats, items, by_type, vocab_sizes,
                           local_lut, feats, tr, va, encoder, device, args)
        results[kind] = {"val_mean": round(float(best["mean"]), 4),
                         "val_accs": {t: round(float(a), 4)
                                      for t, a in (best["accs"] or {}).items()}}
        best_stacks[kind] = best

    bow_r, clip_r = results.get("bow"), results.get("clip")
    decision, reason = "REJECT", ""
    if bow_r is None or clip_r is None:
        reason = "a stack was unavailable (CLIP load failure?)"
    else:
        gate1 = clip_r["val_mean"] > bow_r["val_mean"]
        pres_b = bow_r["val_accs"].get("presence", 0.0)
        pres_c = clip_r["val_accs"].get("presence", 0.0)
        gate2 = pres_c >= pres_b - 0.005
        decision = "ADOPT" if (gate1 and gate2) else "REJECT"
        reason = (f"gate1(clip>bow mean)={gate1} "
                  f"gate2(presence no-regress)={gate2} "
                  f"[bow={pres_b:.4f} clip={pres_c:.4f}]")

    if decision == "ADOPT":
        heads_state, coral_state = best_stacks["clip"]["state"]
        ck = {"heads": heads_state, "coral": coral_state,
              "vocab_sizes": vocab_sizes, "local_lut": local_lut,
              "type_vocab": base.type_vocab, "backbone": "scene",
              "encoder_state": encoder.state_dict(),
              "image_size": args.image_size, "bow_dim": 512,
              "qfeat_kind": "clip",
              "val_mean_acc": clip_r["val_mean"]}
        tmp = CONFIG.weights_dir / "type_heads.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(CONFIG.weights_dir / "type_heads.pt")
        print(f"PROMOTED CLIP stack (val_mean={clip_r['val_mean']} vs "
              f"bow {bow_r['val_mean']}) -> weights/type_heads.pt", flush=True)
    else:
        print(f"kept existing type_heads.pt -- {reason}", flush=True)

    out = {"gate": decision, "reason": reason,
           "bow": bow_r, "clip": clip_r,
           "split": {"train": len(tr), "val": len(va)},
           "args": {k: str(v) for k, v in vars(args).items()}}
    CONFIG.artifact("phase2_vqa_gate.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("logged runs/phase2_vqa_gate.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    main(ap.parse_args())

