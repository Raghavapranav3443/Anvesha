"""Export TorchScript + int8 dynamic quantization of the VQA encoder/head,
with a measured accuracy gate (adopt only if delta <= 0.5%).

  python scripts/export_torchscript.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from satquery.models.vqa import _FusionHead, infer_question_type
from scripts.train_vqa import RSVQADataset

TS_DIR = CONFIG.weights_dir / "ts"
GATE = 0.005


@torch.no_grad()
def accuracy(encoder, head, ds, bow_dim, device, limit=2500) -> float:
    dl = DataLoader(ds, batch_size=128)
    correct = seen = 0
    for x, qb, t, y in dl:
        x = x.to(device)
        feat = encoder(x)
        q = torch.zeros(x.shape[0], bow_dim)
        for i, it in enumerate(ds.items[seen:seen + x.shape[0]]):
            q[i] = torch.from_numpy(_hash_bow(it[1], bow_dim))
        tt = torch.tensor([infer_question_type(it[1], ds.type_vocab)
                           for it in ds.items[seen:seen + x.shape[0]]])
        logits = head(feat, q.to(device), tt.to(device))
        correct += (logits.argmax(1).cpu() == y).sum().item()
        seen += x.shape[0]
        if seen >= limit:
            break
    return correct / max(seen, 1)


def _hash_bow(text, dim):
    import hashlib, re
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class VQAWrapper(nn.Module):
    """Encoder + head fused for tracing (question tensor passed in)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, q, t):
        return self.head(self.encoder(x), q, t)


def main():
    device = "cpu"  # export targets the CPU contract
    ck = torch.load(CONFIG.vqa_weights, map_location="cpu", weights_only=False)
    bow_dim = int(ck.get("bow_dim", 512))
    type_vocab = ck.get("type_vocab", {}) or {}
    ds = RSVQADataset(CONFIG.data_dir / "rsvqa_lr", "val",
                      image_size=int(ck.get("input_size", 128)),
                      type_vocab=type_vocab)

    encoder = SceneEncoder(3).eval()
    encoder.load_state_dict(ck["encoder"])
    head = _FusionHead(len(ck["answer_vocab"]), len(type_vocab)).eval()
    head.load_state_dict(ck["head"])

    acc_fp32 = accuracy(encoder, head, ds, bow_dim, device)
    print(f"fp32 val acc: {acc_fp32:.4f}")

    TS_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = VQAWrapper(encoder, head).eval()
    example = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        # warm-up trace through encoder and head separately (dynamic shapes
        # in the fusion MLP make whole-graph tracing brittle)
        ts_enc = torch.jit.trace(encoder, example)
    q0 = torch.zeros(1, bow_dim)
    t0 = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        feat = ts_enc(example)
        ts_head = torch.jit.trace(head, (feat, q0, t0))
    ts_enc.save(TS_DIR / "vqa_encoder.ts")
    ts_head.save(TS_DIR / "vqa_head.ts")

    q_enc = torch.quantization.quantize_dynamic(
        encoder, {nn.Conv2d}, dtype=torch.qint8)
    q_head = torch.quantization.quantize_dynamic(
        head, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        ts_qenc = torch.jit.trace(q_enc, example)
    feat_q = ts_qenc(example)
    ts_qhead = torch.jit.trace(q_head, (feat_q, q0, t0))
    ts_qenc.save(TS_DIR / "vqa_encoder_int8.ts")
    ts_qhead.save(TS_DIR / "vqa_head_int8.ts")

    acc_q = accuracy(ts_qenc, ts_qhead, ds, bow_dim, device)
    delta = acc_fp32 - acc_q
    print(f"int8 val acc: {acc_q:.4f} (delta {delta:+.4f})")

    # latency probe
    import time
    x = torch.zeros(1, 3, 128, 128)
    for name, enc_m, head_m in (("fp32", encoder, head),
                                ("int8", q_enc, q_head)):
        with torch.no_grad():
            enc_m(x)
            t_start = time.time()
            for _ in range(30):
                head_m(enc_m(x), q0, t0)
            dt = (time.time() - t_start) / 30 * 1000
        print(f"{name} latency: {dt:.1f} ms/image")

    adopt = abs(delta) <= GATE
    meta = {"fp32_acc": acc_fp32, "int8_acc": acc_q, "delta": round(delta, 4),
            "adopted": bool(adopt), "gate": GATE}
    (TS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("adopted:", adopt, "| meta ->", TS_DIR / "meta.json")


if __name__ == "__main__":
    main()
