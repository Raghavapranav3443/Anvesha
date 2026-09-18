"""Per-type VQA specialist heads + CORAL ordinal counting.

Heads train on the FROZEN shared RS-adapted encoder: features are computed
once per batch under no_grad and each sample is routed per-sample to its
type's head, so specialists can never drift the encoder. At inference the
detected question type routes to the matching head.

Outputs weights/type_heads.pt consumed by anvesha.models.vqa routing.
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
from torch.utils.data import DataLoader

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder
from anvesha.models.vqa import _FusionHead, _hashed_bow
from scripts.train_vqa import RSVQADataset


class CORALHead(nn.Module):
    """Consistent rank logistic regression: K-1 thresholds, shared weight,
    independent biases. Prediction = number of thresholds exceeded."""

    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.k_minus_1 = n_classes - 1
        self.fc = nn.Linear(feat_dim, 1, bias=False)
        self.a = nn.Parameter(torch.zeros(self.k_minus_1))
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, z):
        return self.fc(z) + self.a


def coral_loss(logits, ranks, importance):
    return -(ranks * nn.functional.logsigmoid(logits) +
             (1 - ranks) * nn.functional.logsigmoid(-logits) * importance
             ).sum(dim=1).mean()


def digit_ranks(digits: torch.Tensor, k_minus_1: int) -> torch.Tensor:
    d = digits.unsqueeze(1)
    return (torch.arange(k_minus_1, device=d.device).unsqueeze(0) < d).float()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = RSVQADataset(Path(args.data), "train",
                        image_size=args.image_size, augment=True)
    print(f"base items: {len(base)} | device {device}", flush=True)

    # annotate items: (f, q, qt, y, digit_or_None)
    items = []
    for f, q, qt, y in base.items:
        digit = None
        if qt == "count":
            m = re.fullmatch(r"\d+", base.answer_vocab[y].strip())
            if m and int(base.answer_vocab[y]) < 10:
                digit = int(base.answer_vocab[y])
        items.append((f, q, qt, y, digit))

    # per-type local answer vocabs (non-count) + count digit set
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

    # group item indices by type
    by_type = collections.defaultdict(list)
    for i, (f, q, qt, y, digit) in enumerate(items):
        if qt == "count" and digit is not None:
            by_type["count"].append(i)
        elif qt in local_lut and y in local_lut[qt]:
            by_type[qt].append(i)

    selected = None
    if args.types:
        wanted = [t.strip() for t in args.types.split(",") if t.strip()]
        selected = {t for t in wanted if t in vocab_sizes}
        vocab_sizes = {t: n for t, n in vocab_sizes.items() if t in selected}
        local_lut = {t: lut for t, lut in local_lut.items() if t in selected}

    print("per-type counts:", {t: len(v) for t, v in sorted(by_type.items())
                               if not selected or t in selected}, flush=True)

    dl = DataLoader(list(range(len(items))), batch_size=args.batch_size,
                    shuffle=True)

    backbone_kind = args.backbone
    if backbone_kind == "dino":
        from anvesha.models.dino_encoder import DinoEncoder
        encoder = DinoEncoder(weights=str(CONFIG.weights_dir / "dinov2_vits14.pt"),
                              device=device)
    else:
        encoder = SceneEncoder(3).to(device)
        if CONFIG.scene_encoder_weights.exists():
            ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                            weights_only=False)
            encoder.load_state_dict(ck["encoder"])
            print("encoder warm-started from RS-adapted scene encoder", flush=True)

    heads = {t: _FusionHead(n, 1).to(device)
             for t, n in vocab_sizes.items()}
    coral = CORALHead(256, 10).to(device)
    all_params = (list(encoder.parameters()) +
                  [p for h in heads.values() for p in h.parameters()] +
                  list(coral.parameters()))
    opt = torch.optim.AdamW(all_params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    c_counts = collections.Counter(items[i][4] for i in by_type["count"])
    imp = torch.tensor([1.0 / np.sqrt(c_counts.get(k, 0) + 1)
                        for k in range(9)])
    imp = (imp / imp.mean()).to(device)

    best_mean = -1.0
    tmp = CONFIG.weights_dir / "type_heads.tmp.pt"
    for epoch in range(args.epochs):
        encoder.train()
        for h in heads.values():
            h.train()
        coral.train()
        accs = collections.defaultdict(lambda: [0, 0])
        loss_sum = loss_n = 0
        for idx in dl:
            batch = [items[i] for i in idx.tolist()]
            x = torch.stack([
                torch.from_numpy(_px(f).transpose(2, 0, 1))
                for f, q, qt, y, d in batch]).to(device)
            qb = torch.stack([
                torch.from_numpy(_hashed_bow(q)) for f, q, qt, y, d in batch]).to(device)
            opt.zero_grad()
            with torch.no_grad():
                feat = encoder(x)
            total = 0.0
            for t_name, head in heads.items():
                sel = [j for j, it in enumerate(batch) if it[2] == t_name
                       and it[3] in local_lut[t_name]]
                if not sel:
                    continue
                logits = head(feat[sel], qb[sel],
                              torch.zeros(len(sel), dtype=torch.long,
                                          device=device))
                ys = torch.tensor([local_lut[t_name][batch[j][3]] for j in sel],
                                  device=device)
                total = total + ce(logits, ys)
                accs[t_name][0] += (logits.argmax(1) == ys).sum().item()
                accs[t_name][1] += len(sel)
            sel_c = [j for j, it in enumerate(batch)
                     if it[2] == "count" and it[4] is not None]
            if sel_c:
                digits = torch.tensor([batch[j][4] for j in sel_c],
                                      device=device)
                logits = coral(feat[sel_c])
                ranks = digit_ranks(digits, 9)
                total = total + coral_loss(logits, ranks, imp)
                pred = (torch.sigmoid(logits) > 0.5).sum(1)
                accs["count"][0] += (pred == digits).sum().item()
                accs["count"][1] += len(sel_c)
            if torch.is_tensor(total):
                total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                loss_sum += total.item(); loss_n += 1
        sched.step()
        mean_acc = np.mean([c[0] / max(c[1], 1) for c in accs.values()]) \
            if accs else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={loss_sum/max(loss_n,1):.4f} "
              + " ".join(f"{t}={c[0]/max(c[1],1):.3f}"
                         for t, c in sorted(accs.items())), flush=True)
        if mean_acc > best_mean:
            best_mean = mean_acc
            torch.save({"heads": {t: h.state_dict() for t, h in heads.items()},
                        "coral": coral.state_dict(),
                        "vocab_sizes": vocab_sizes,
                        "local_lut": local_lut,
                        "type_vocab": base.type_vocab,
                        "backbone": backbone_kind,
                        "encoder_state": encoder.state_dict(),
                        "image_size": args.image_size,
                        "bow_dim": 512,
                        "val_mean_acc": round(float(best_mean), 4)},
                       tmp)
    tmp.replace(CONFIG.weights_dir / "type_heads.pt")
    print("saved", CONFIG.weights_dir / "type_heads.pt",
          "best mean acc", round(best_mean, 4))


def _px(f: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(f).convert("RGB").resize((128, 128)),
                      dtype=np.float32) / 255.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone", choices=("scene", "dino"), default="scene")
    ap.add_argument("--types", default="",
                    help="comma list of non-count types to train (default all)")
    main(ap.parse_args())
