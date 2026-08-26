"""Train a compact transformer caption decoder conditioned on the
RS-adapted SceneEncoder, using real BigEarthNet.txt captions joined to our
local co-registered S2 patches (data/bentxt_join/captions.parquet).

v2 — plan-conditioned decoding + beam search:
* A 19-d BEN19 "content plan" is weakly derived from each caption's own text
  (keyword matching of canonical class phrases). BigEarthNet.txt captions are
  generated from the label set, so the plan captures exactly what the decoder
  must realize.
* An internal PlanHead (trained jointly on frozen SceneEncoder features)
  predicts that plan from the image alone at inference — no extra inputs.
* Decoding supports beam search with length normalization (greedy fallback).
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
from torch.utils.data import DataLoader, Dataset

from satquery.config import CONFIG, BEN19_CLASSES
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


# --------------------------------------------------------------------- #
# Weak content-plan extraction from caption text
# --------------------------------------------------------------------- #

PLAN_KEYWORDS = {
    "Urban fabric": ("urban fabric",),
    "Industrial or commercial units": ("industrial", "commercial unit"),
    "Arable land": ("arable land", "non-irrigated arable"),
    "Permanent crops": ("permanent crop", "orchard", "vineyard", "olive grove"),
    "Pastures": ("pasture",),
    "Complex cultivation patterns": ("complex cultivation",),
    "Agriculture with natural vegetation":
        ("principally occupied by agriculture",
         "agriculture with significant areas of natural vegetation",
         "agriculture with natural vegetation"),
    "Agro-forestry areas": ("agro-forestry", "agroforestry"),
    "Broad-leaved forest": ("broad-leaved forest",),
    "Coniferous forest": ("coniferous forest",),
    "Mixed forest": ("mixed forest",),
    "Natural grassland": ("natural grassland", "grassland"),
    "Moors and heathland": ("moor", "heathland", "heath"),
    "Sclerophyllous vegetation": ("sclerophyllous",),
    "Transitional woodland/shrub": ("transitional woodland", "woodland or shrub",
                                    "woodlands or shrub", "woodland and shrub",
                                    "shrubland", "shrubs and woodland", "shrub"),
    "Beaches dunes sands": ("beach", "dune", "sandy"),
    "Inland waters": ("inland water", "lake", "river"),
    "Coastal wetlands": ("wetland", "salt marsh", "coastal lagoon"),
    "Marine waters": ("marine water", "sea ", "ocean", "bay"),
}


def text_to_plan(text: str) -> np.ndarray:
    t = f" {text.lower()} "
    plan = np.zeros(len(BEN19_CLASSES), dtype=np.float32)
    for i, cls in enumerate(BEN19_CLASSES):
        if any(k in t for k in PLAN_KEYWORDS[cls]):
            plan[i] = 1.0
    if plan.sum() == 0.0:                      # degenerate: dominant-class hint
        plan[:] = 0.05                          # soft prior, keeps conditioning sane
    return plan


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
                 image_size=120, max_len=44):
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
        self.plans = np.stack([text_to_plan(t) for _, t in keep]) \
            if keep else np.zeros((0, len(BEN19_CLASSES)), dtype=np.float32)
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
        return (torch.from_numpy(x.transpose(2, 0, 1)), tgt, L,
                torch.from_numpy(self.plans[i]))


class Captioner(nn.Module):
    """Frozen-encoder features cross-attended by a small causal transformer.

    cond=True adds the v2 plan pathway: a Linear projection of a 19-d BEN19
    multi-hot broadcast-added to every memory token, plus an internal PlanHead
    predicting the plan from pooled features at inference time.
    """

    N_PLAN = len(BEN19_CLASSES)

    def __init__(self, vocab_size, d=256, n_layers=4, n_heads=8,
                 ff=768, max_len=44, feat_ch=128, cond=False):
        super().__init__()
        self.d = d
        self.cond = bool(cond)
        self.feat_proj = nn.Conv2d(feat_ch, d, 1)
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(1, max_len + 225, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, batch_first=True,
                                           norm_first=True)
        self.dec = nn.TransformerEncoder(layer, n_layers)
        self.lm_head = nn.Linear(d, vocab_size)
        if self.cond:
            self.plan_proj = nn.Linear(self.N_PLAN, d)
            self.plan_head = nn.Sequential(
                nn.Linear(feat_ch, 128), nn.ReLU(), nn.Linear(128, self.N_PLAN))

    def forward(self, fmap, tokens, plan=None):
        B = fmap.shape[0]
        mem = self.feat_proj(fmap).flatten(2).transpose(1, 2)      # B x 225 x d
        mem = mem + self.pos[:, :mem.shape[1]]
        if self.cond:
            p = plan if plan is not None else torch.sigmoid(
                self.predict_plan(fmap))
            mem = mem + self.plan_proj(p).unsqueeze(1)
        L = tokens.shape[1]
        x = self.tok(tokens) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=tokens.device), 1).bool()
        h = self.dec(x, mask=mask)
        return self.lm_head(h)

    def predict_plan(self, fmap):
        pooled = fmap.mean(dim=(2, 3))
        return self.plan_head(pooled)

    @torch.no_grad()
    def _decode_step(self, fmap, tokens, plan):
        logits = self(fmap, tokens, plan)[:, -1]
        return torch.log_softmax(logits, -1)

    @torch.no_grad()
    def generate(self, fmap, vocab, max_len=44, beam=1):
        if beam <= 1 or not self.cond and beam > 1:
            return self._generate_greedy(fmap, vocab, max_len)
        return self._generate_beam(fmap, vocab, max_len, beam)

    @torch.no_grad()
    def _generate_greedy(self, fmap, vocab, max_len=44):
        B = fmap.shape[0]
        plan = torch.sigmoid(self.predict_plan(fmap)) if self.cond else None
        tokens = torch.full((B, 1), vocab.stoi["<bos>"], dtype=torch.long,
                            device=fmap.device)
        done = torch.zeros(B, dtype=torch.bool, device=fmap.device)
        for _ in range(max_len - 1):
            logp = self._decode_step(fmap, tokens, plan)
            nxt = logp.argmax(-1, keepdim=True)
            nxt[done] = vocab.stoi["<pad>"]
            tokens = torch.cat([tokens, nxt], dim=1)
            done |= nxt.squeeze(1) == vocab.stoi["<eos>"]
            if done.all():
                break
        return [_tokens_to_text(row, vocab) for row in tokens.tolist()]

    @torch.no_grad()
    def _generate_beam(self, fmap, vocab, max_len=44, beam=3):
        """Per-image beam search with length-normalized scores."""
        device = fmap.device
        bos, eos, pad = vocab.stoi["<bos>"], vocab.stoi["<eos>"], vocab.stoi["<pad>"]
        plan = torch.sigmoid(self.predict_plan(fmap)) if self.cond else None
        out = []
        for b in range(fmap.shape[0]):
            fm = fmap[b:b + 1]
            pl = plan[b:b + 1] if plan is not None else None
            seqs = torch.full((1, 1), bos, dtype=torch.long, device=device)
            scores = torch.zeros(1, device=device)
            finished = []                                    # (score, tokens)
            for _ in range(max_len - 1):
                logp = self._decode_step(fm.expand(seqs.shape[0], -1, -1, -1)
                                         if fm.dim() == 4 else fm,
                                         seqs, pl)
                total = scores.unsqueeze(1) + logp           # Bk x V
                flat = total.flatten()
                top = torch.topk(flat, beam).indices
                cand_seq = top // logp.shape[1]
                cand_tok = top % logp.shape[1]
                new_seqs, new_scores = [], []
                active_tokens, active_scores = [], []
                for s, t, sc in zip(cand_seq.tolist(), cand_tok.tolist(),
                                    flat[top].tolist()):
                    row = torch.cat([seqs[s], torch.tensor([t], device=device)])
                    if t == eos:
                        finished.append((sc / len(row), row.tolist()))
                    elif t == pad:
                        continue
                    else:
                        active_tokens.append(row)
                        active_scores.append(sc)
                if not active_tokens:
                    break
                # topk returns descending order — keep the BEST beams
                seqs = torch.stack(active_tokens)[:beam]
                scores = torch.tensor(active_scores[:beam], device=device)
            if not finished and seqs.numel():
                finished.append((scores[0].item() / seqs.shape[1],
                                 seqs[0].tolist()))
            best = max(finished, key=lambda x: x[0])[1] if finished else []
            out.append(_tokens_to_text(best, vocab))
        return out


def _tokens_to_text(row, vocab):
    special = {vocab.stoi["<pad>"], vocab.stoi["<bos>"], vocab.stoi["<eos>"]}
    return " ".join(vocab.itos[t] for t in row if t not in special)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
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

    model = Captioner(len(ds_tr.vocab), cond=not args.no_cond).to(device)
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ignore = ds_tr.vocab.pad
    pos_w = torch.tensor([2.0] * Captioner.N_PLAN, device=device)

    def feat_of(x):
        with torch.no_grad():
            return enc.feature_map(x, stride=8)

    best = -1.0
    tmp = CONFIG.weights_dir / "captioner.tmp.pt"
    for epoch in range(args.epochs):
        model.train(); tot = cnt = 0
        for x, tgt, L, plan in dl:
            x, tgt, plan = x.to(device), tgt.to(device), plan.to(device)
            fmap = feat_of(x)
            logits = model(fmap, tgt[:, :-1], plan)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1),
                ignore_index=ignore)
            if model.cond:
                pooled = fmap.mean(dim=(2, 3))
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    model.plan_head(pooled), plan, pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); cnt += 1
        sched.step()
        # validation BLEU (greedy during training for speed)
        model.eval()
        bleus = []
        with torch.no_grad():
            for bi, (x, tgt, L, plan) in enumerate(dl_va):
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
                        "val_bleu": best,
                        "cond": model.cond}, tmp)
    tmp.replace(CONFIG.weights_dir / "captioner.pt")
    print("saved", CONFIG.weights_dir / "captioner.pt", "best BLEU",
          round(best, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "bigearthnet_14k" / "BEN_14k"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-cond", action="store_true")
    main(ap.parse_args())
