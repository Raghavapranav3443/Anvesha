"""Fine-tune the RSVQA specialist v3.

Upgrades over v2:
  - 192px image resolution (up from 128px) for better spatial detail
  - 30 epochs with cosine LR (up from 20)
  - AMP (mixed precision) for faster training
  - Experiment logging
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from anvesha.config import CONFIG
from anvesha.models.backbone import SceneEncoder


def bow(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class _Head(nn.Module):
    def __init__(self, n_answers: int, n_types: int):
        super().__init__()
        self.img_proj = nn.Linear(SceneEncoder.FEATURE_DIM, 256)
        self.q_proj = nn.Linear(512, 256)
        self.t_embed = nn.Embedding(n_types + 1, 32)
        self.mlp = nn.Sequential(nn.Linear(256 + 256 + 32, 384), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(384, n_answers))

    def forward(self, feat, q, t):
        z = torch.cat([torch.relu(self.img_proj(feat)),
                       torch.relu(self.q_proj(q)),
                       self.t_embed(t)], dim=1)
        return self.mlp(z)


class RSVQADataset(Dataset):
    """RSVQA LowRes with question-type conditioning."""

    def __init__(self, root: Path, split="train", max_items=None,
                 image_size: int = 192, max_answers=100, augment=False,
                 type_vocab=None):
        from PIL import Image
        root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.img_dir = root / "Images_LR"
        qs = json.loads((root / f"LR_split_{split}_questions.json")
                        .read_text(encoding="utf-8"))["questions"]
        raw_ans = json.loads((root / f"LR_split_{split}_answers.json")
                             .read_text(encoding="utf-8"))["answers"]
        ans_by_id = {a["id"]: str(a["answer"]) for a in raw_ans
                     if "answer" in a}
        triplets = []
        for q in qs:
            if not q.get("active", True) or not q.get("answers_ids"):
                continue
            gt = ans_by_id.get(q["answers_ids"][0])
            if gt is not None:
                triplets.append((q["img_id"], q["question"],
                                 str(q.get("type", "other")), gt))

        counts = collections.Counter(t[3] for t in triplets)
        keep = {a for a, _ in counts.most_common(max_answers)}
        self.vocab = {a: i for i, a in enumerate(sorted(keep))}
        self.items = []
        for img_id, question, qtype, gt in triplets[:max_items or len(triplets)]:
            if gt not in keep:
                continue
            f = self.img_dir / f"{img_id}.tif"
            if f.exists():
                self.items.append((f, question, qtype, self.vocab[gt]))
        if type_vocab is None:
            types = sorted({t[2] for t in self.items})
            self.type_vocab = {t: i for i, t in enumerate(types)}
        else:
            self.type_vocab = type_vocab

    @property
    def answer_vocab(self):
        inv = {i: a for a, i in self.vocab.items()}
        return [inv[i] for i in range(len(inv))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        f, q, qtype, y = self.items[i]
        arr = np.asarray(Image.open(f).convert("RGB").resize(
            (self.image_size,) * 2), dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1].copy()
            k = np.random.randint(0, 4)
            if k:
                arr = np.rot90(arr, k).copy()
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        qb = torch.from_numpy(bow(q))
        t = self.type_vocab.get(qtype, len(self.type_vocab))
        return x, qb, torch.tensor(t), y


def evaluate(model, encoder, dl, device):
    model.eval(); encoder.eval()
    correct = seen = 0
    with torch.no_grad():
        for x, qb, t, y in dl:
            logits = model(encoder(x.to(device)), qb.to(device),
                           t.to(device))
            correct += (logits.argmax(1).cpu() == y).sum().item()
            seen += len(y)
    return correct / max(seen, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    ds_tr = RSVQADataset(root, "train", args.max_items,
                         image_size=args.image_size, augment=True)
    ds_va = RSVQADataset(root, "val", image_size=args.image_size,
                         type_vocab=ds_tr.type_vocab)
    print(f"train={len(ds_tr)} val={len(ds_va)} answers={len(ds_tr.answer_vocab)}"
          f" types={len(ds_tr.type_vocab)} device={device}", flush=True)

    # class-balanced weights (inverse sqrt frequency)
    freq = collections.Counter(it[3] for it in ds_tr.items)
    w = torch.tensor([1.0 / float(np.sqrt(freq[i]))
                      for i in range(len(ds_tr.answer_vocab))],
                     dtype=torch.float32)
    w = (w / w.mean()).to(device)

    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)

    encoder = SceneEncoder(3).to(device)
    head = _Head(len(ds_tr.answer_vocab), len(ds_tr.type_vocab)).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        print("encoder warm-started from RS-adapted scene encoder")

    opt = torch.optim.AdamW(list(encoder.parameters()) +
                            list(head.parameters()), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = 0.0
    ckpt_path = CONFIG.vqa_weights.with_suffix(".v2.pt.tmp")
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
        va = evaluate(head, encoder, dl_va, device)
        print(f"epoch {epoch+1}/{args.epochs} train_acc={correct/max(seen,1):.4f} "
              f"val_acc={va:.4f}", flush=True)
        if va > best:
            best = va
            torch.save({"encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        "answer_vocab": ds_tr.answer_vocab,
                        "input_size": args.image_size,
                        "bow_dim": 512,
                        "type_vocab": ds_tr.type_vocab,
                        "val_accuracy": va}, ckpt_path)

    final = CONFIG.vqa_weights.with_suffix(".pt")
    ckpt_path.replace(CONFIG.vqa_weights)
    print(f"saved {CONFIG.vqa_weights} (best val_acc={best:.4f})")

    # Experiment log
    from anvesha.experiment_log import log_experiment
    log_experiment(
        script="train_vqa",
        args={"image_size": args.image_size, "epochs": args.epochs,
              "lr": args.lr, "batch_size": args.batch_size},
        metrics={"val_acc": best},
        checkpoint=str(CONFIG.vqa_weights),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "rsvqa_lr"))
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    train(ap.parse_args())
