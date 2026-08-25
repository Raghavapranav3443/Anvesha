"""Temperature scaling for specialist heads: minimises NLL on held-out data.

  python scripts/calibrate.py            # calibrates VQA + count head
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from satquery.models.vqa import _FusionHead, _hashed_bow, infer_question_type
from scripts.train_vqa import RSVQADataset


def collect_logits(encoder, head, ds, bow_dim, type_vocab, device,
                   input_size, limit=4000):
    dl = DataLoader(ds, batch_size=128)
    logits, ys = [], []
    n = 0
    with torch.no_grad():
        for x, qb, t, y in dl:
            x = x.to(device)
            feat = encoder(x)
            q = torch.zeros(x.shape[0], bow_dim)
            for i, item in enumerate(ds.items[n:n + x.shape[0]]):
                q[i] = torch.from_numpy(_hashed_bow(item[1], dim=bow_dim))
            t_idx = torch.tensor([infer_question_type(it[1], type_vocab)
                                  for it in ds.items[n:n + x.shape[0]]])
            lg = head(feat, q.to(device), t_idx.to(device))
            logits.append(lg.cpu())
            ys.append(y)
            n += x.shape[0]
            if n >= limit:
                break
    return torch.cat(logits), torch.cat(ys)


def best_temperature(logits, ys):
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.3, 4.0, 75):
        nll = nn.functional.cross_entropy(logits / t, ys).item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 3), round(best_nll, 4)


def calibrate_vqa(device):
    path = CONFIG.vqa_weights
    if not path.exists():
        print("vqa weights missing; skip")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bow_dim = int(ck.get("bow_dim", 512))
    type_vocab = ck.get("type_vocab", {}) or {}
    enc = SceneEncoder(3).to(device)
    enc.load_state_dict(ck["encoder"])
    head = _FusionHead(len(ck["answer_vocab"]), len(type_vocab)).to(device)
    head.load_state_dict(ck["head"])
    ds = RSVQADataset(CONFIG.data_dir / "rsvqa_lr", "val",
                      image_size=int(ck.get("input_size", 128)),
                      type_vocab=type_vocab)
    logits, ys = collect_logits(enc, head, ds, bow_dim, type_vocab, device,
                                int(ck.get("input_size", 128)))
    t, nll = best_temperature(logits, ys)
    print(f"VQA temperature={t} (val NLL {nll})")
    ck["temperature"] = t
    torch.save(ck, path)

    cpath = CONFIG.weights_dir / "count_head.pt"
    if cpath.exists():
        ckc = torch.load(cpath, map_location="cpu", weights_only=False)
        ckc["temperature"] = t   # same encoder/training family; reuse
        torch.save(ckc, cpath)
        print(f"count head temperature set to {t}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calibrate_vqa(device)
