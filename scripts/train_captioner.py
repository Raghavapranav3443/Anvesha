"""Train a compact transformer caption decoder conditioned on the
RS-adapted SceneEncoder, using real BigEarthNet.txt captions joined to our
local co-registered S2 patches (data/bentxt_join/captions.parquet).
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
from satquery.models.backbone import SceneEncoder, normalise_for_encoder, resize_np


def simple_bleu(pred: str, ref: str, n_max: int = 4) -> float:
    pt = re.findall(r"[a-z0-9]+", pred.lower())
    rt = re.findall(r"[a-z0-9]+", ref.lower())
    if not pt or not rt:
        return 0.0
    log_prec = 0.0
    for n in range(1, n_max + 1):
        pg = [tuple(pt[i:i + n]) for i in range(len(pt) - n + 1)]
        rg = collections.Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
        if not pg:
            break
        clip = sum(min(c, rg.get(g, 0)) for g, c in collections.Counter(pg).items())
        log_prec += np.log((clip + 1e-9) / len(pg))
    bp = min(1.0, np.exp(1 - len(rt) / max(len(pt), 1)))
    return float(bp * np.exp(log_prec / n_max))


class CaptionVocab:
    def __init__(self, texts, min_freq=3, max_size=6000):
        counts = collections.Counter()
        for t in texts:
            counts.update(re.findall(r"[a-z0-9]+", t.lower()))
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"]
        for w, c in counts.most_common(max_size):
            if c >= min_freq:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad = 0

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=44):
        ids = [self.stoi["<bos>"]]
        for w in re.findall(r"[a-z0-9]+", text.lower())[:max_len - 2]:
            ids.append(self.stoi.get(w, 2))
        ids.append(self.stoi["<eos>"])
        return ids


class CaptionDataset(Dataset):
    def __init__(self, join_path: Path, s2_dir: Path, vocab=None,
                 split_filter=None, image_size=120, max_len=44):
        import pandas as pd
        df = pd.read_parquet(join_path)
        # one random prompt per patch keeps epochs diverse without dupes
        df = df.sample(frac=1.0, random_state=0).drop_duplicates("patch_id")
        keep = []
        for _, r in df.iterrows():
            f = s2_dir / f"{r['patch_id']}.tif"
            if f.exists():
                keep.append((f, str(r["output"])))
        self.items = keep
        if vocab is None:
            self.vocab = CaptionVocab([t for _, t in self.items])
        else:
            self.vocab = vocab
        self.image_size = image_size
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import rasterio
        f, text = self.items[i]
        with rasterio.open(f) as src:
            arr = np.moveaxis(src.read().astype(np.float32), 0, -1)
        rgb = np.clip(arr[..., [2, 1, 0]] / 10000.0, 0, 1)
        x = resize_np(rgb, self.image_size)
        ids = self.vocab.encode(text, self.max_len)
        pad = self.vocab.pad
        L = len(ids)
        tgt = torch.full((self.max_len,), pad, dtype=torch.long)
        tgt[:L] = torch.tensor(ids)
        return torch.from_numpy(x.transpose(2, 0, 1)), tgt, L


class Captioner(nn.Module):
    """Encoder features cross-attended by a small causal transformer."""

    def __init__(self, vocab_size, d=256, n_layers=4, n_heads=8,
                 ff=768, max_len=44, feat_ch=128):
        super().__init__()
        self.d = d
        self.feat_proj = nn.Conv2d(feat_ch, d, 1)
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(1, max_len + 225, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, batch_first=True,
                                           norm_first=True)
        self.dec = nn.TransformerEncoder(layer, n_layers)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, fmap, tokens):
        B = fmap.shape[0]
        mem = self.feat_proj(fmap).flatten(2).transpose(1, 2)      # B x 225 x d
        mem = mem + self.pos[:, :mem.shape[1]]
        L = tokens.shape[1]
        x = self.tok(tokens) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=tokens.device), 1).bool()
        h = self.dec(x, mask=mask)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, fmap, vocab, max_len=44):
        B = fmap.shape[0]
        tokens = torch.full((B, 1), vocab.stoi["<bos>"], dtype=torch.long,
                            device=fmap.device)
        done = torch.zeros(B, dtype=torch.bool, device=fmap.device)
        for _ in range(max_len - 1):
            logits = self(fmap, tokens)[:, -1]
            nxt = logits.argmax(-1, keepdim=True)
            nxt[done] = vocab.stoi["<pad>"]
            tokens = torch.cat([tokens, nxt], dim=1)
            done |= nxt.squeeze(1) == vocab.stoi["<eos>"]
            if done.all():
                break
        out = []
        for row in tokens.tolist():
            words = [vocab.itos[t] for t in row[1:]
                     if t not in (vocab.stoi["<pad>" if False else "<pad>"],
                                  vocab.stoi["<bos>"], vocab.stoi["<eos>"])]
            out.append(" ".join(words))
        return out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    s2_dir = root / "BigEarthNet-S2"
    ds_tr = CaptionDataset(CONFIG.data_dir / "bentxt_join" / "captions.parquet",
                           root / "BigEarthNet-S2" / "train")
    ds_va = CaptionDataset(CONFIG.data_dir / "bentxt_join" / "captions.parquet",
                           root / "BigEarthNet-S2" / "validation",
                           vocab=ds_tr.vocab)
    print(f"captions train={len(ds_tr)} val={len(ds_va)} "
          f"vocab={len(ds_tr.vocab)} device={device}", flush=True)

    enc = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    model = Captioner(len(ds_tr.vocab)).to(device)
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ignore = ds_tr.vocab.pad

    def feat_of(x):
        with torch.no_grad():
            return enc.feature_map(x, stride=8)

    best = -1.0
    tmp = CONFIG.weights_dir / "captioner.tmp.pt"
    for epoch in range(args.epochs):
        model.train(); tot = cnt = 0
        for x, tgt, L in dl:
            x, tgt = x.to(device), tgt.to(device)
            logits = model(feat_of(x), tgt[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1),
                ignore_index=ignore)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
        sched.step()
        # validation BLEU
        model.eval()
        bleus = []
        with torch.no_grad():
            for bi, (x, tgt, L) in enumerate(dl_va):
                x = x.to(device)
                preds = model.generate(feat_of(x), ds_va.vocab)
                for pr, row in zip(preds, tgt.tolist()):
                    ref_ids = [t for t in row
                               if t not in (ignore, 1, 2)]
                    ref = " ".join(ds_va.vocab.itos[t] for t in ref_ids)
                    bleus.append(simple_bleu(pr, ref))
                if bi >= 12:
                    break
        b = float(np.mean(bleus)) if bleus else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={tot/max(cnt,1):.4f} "
              f"val_BLEU={b:.4f}", flush=True)
        if b > best:
            best = b
            torch.save({"model": model.state_dict(),
                        "vocab": ds_tr.vocab.itos,
                        "val_bleu": best}, tmp)
    tmp.replace(CONFIG.weights_dir / "captioner.pt")
    print("saved", CONFIG.weights_dir / "captioner.pt", "best BLEU",
          round(best, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "bigearthnet_14k" / "BEN_14k"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    main(ap.parse_args())
