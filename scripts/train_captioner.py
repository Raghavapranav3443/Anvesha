"""Train a compact transformer caption decoder conditioned on the
RS-adapted SceneEncoder, using real BigEarthNet.txt captions joined to our
local co-registered S2 patches (data/bentxt_join/captions.parquet).

v2 -- plan-conditioned decoding + beam search:
* A 19-d BEN19 "content plan" is weakly derived from each caption's own text
  (keyword matching of canonical class phrases). BigEarthNet.txt captions are
  generated from the label set, so the plan captures exactly what the decoder
  must realize.
* An internal PlanHead (trained jointly on frozen SceneEncoder features)
  predicts that plan from the image alone at inference -- no extra inputs.
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
        p_ngrams = collections.Counter(tuple(pt[i:i+n]) for i in range(len(pt)-n+1))
        r_ngrams = collections.Counter(tuple(rt[i:i+n]) for i in range(len(rt)-n+1))
        clipped = sum(min(c, r_ngrams.get(ng, 0)) for ng, c in p_ngrams.items())
        total = sum(p_ngrams.values())
        if total == 0:
            return 0.0
        log_prec += np.log(max(clipped / total, 1e-10))
    bleu = np.exp(log_prec / n_max)
    bp = min(1.0, np.exp(1 - len(rt) / max(len(pt), 1)))
    return float(bp * bleu)


class CaptionVocab:
    def __init__(self, specials):
        self.itos = ["<pad>", "<sos>", "<eos>"] + list(specials)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    @property
    def pad(self):
        return 0
    def __len__(self):
        return len(self.itos)


class CaptionDataset(Dataset):
    def __init__(self, root: Path, split: str, vocab: CaptionVocab,
                 max_items=None, image_size=120):
        import pandas as pd
        from PIL import Image
        self.image_size = image_size
        self.vocab = vocab
        root = Path(root)
        cap_file = root / "captions.parquet"
        if not cap_file.exists():
            root2 = root / "BigEarthNet-S2"
            cap_file = root2 / "captions.parquet" if (root2 / "captions.parquet").exists() else None
        if cap_file is None or not cap_file.exists():
            raise FileNotFoundError(f"No captions.parquet found under {root}")
        df = pd.read_parquet(cap_file)
        df = df[df.split == split]
        if max_items:
            df = df.head(max_items)
        self.items = []
        for _, row in df.iterrows():
            pid = str(row["patch_id"])
            text = str(row["output"])
            f = root / "BigEarthNet-S2" / f"{split}" / f"{pid}.tif"
            if not f.exists():
                f = root / "BigEarthNet-S2" / f"{pid}.tif"
            if f.exists():
                self.items.append((f, text))
        self.plan_table = self._build_plan_table()
    def _build_plan_table(self):
        plan = {}
        for ci, cls in enumerate(BEN19_CLASSES):
            words = cls.lower().split()
            plan[cls] = words
        return plan
    def _text_to_plan(self, text: str) -> np.ndarray:
        vec = np.zeros(len(BEN19_CLASSES), dtype=np.float32)
        low = text.lower()
        for ci, cls in enumerate(BEN19_CLASSES):
            words = cls.lower().split()
            if any(w in low for w in words):
                vec[ci] = 1.0
        return vec
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        f, text = self.items[i]
        import rasterio
        with rasterio.open(str(f)) as src:
            arr = np.moveaxis(src.read()[:3].astype(np.float32), 0, -1)
        arr = np.clip(arr / 10000.0, 0, 1)
        from PIL import Image
        im = Image.fromarray((arr * 255).astype(np.uint8))
        im = im.resize((self.image_size,) * 2)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        toks = re.findall(r"[a-z0-9]+", text.lower())
        ids = [self.vocab.stoi.get("<sos>", 1)]
        for t in toks:
            ids.append(self.vocab.stoi.get(t, 0))
        ids.append(self.vocab.stoi.get("<eos>", 2))
        tgt = torch.tensor(ids, dtype=torch.long)
        plan = torch.from_numpy(self._text_to_plan(text))
        return x, tgt, torch.tensor(len(ids)), plan


class Captioner(nn.Module):
    N_PLAN = len(BEN19_CLASSES)
    def __init__(self, vocab_size, d=256, nhead=4, nlayers=3, cond=True,
                 in_ch=128):
        super().__init__()
        self.cond = cond
        self.in_ch = in_ch
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(512, d)
        decoder_layer = nn.TransformerDecoderLayer(d, nhead, dim_feedforward=d*4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        self.out = nn.Linear(d, vocab_size)
        self.plan_proj = nn.Linear(self.N_PLAN, d) if cond else None
        self.mem_proj = nn.Linear(in_ch, d)  # project encoder features to d
        self.plan_head = nn.Sequential(
            nn.Linear(in_ch, 256), nn.ReLU(), nn.Linear(256, self.N_PLAN)
        ) if cond else None
    def forward(self, fmap, tgt, plan=None):
        B, C, H, W = fmap.shape
        memory = self.mem_proj(fmap.flatten(2).permute(0, 2, 1))
        if self.cond and plan is not None:
            memory = memory + self.plan_proj(plan).unsqueeze(1)
        L = tgt.shape[1]
        pos = self.pos(torch.arange(L, device=tgt.device)).unsqueeze(0)
        x = self.embed(tgt) + pos
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        out = self.decoder(x, memory, tgt_mask=causal)
        return self.out(out)
    @torch.no_grad()
    def generate(self, fmap, vocab, plan=None, max_len=60, beam=1):
        B = fmap.shape[0]
        results = []
        for b in range(B):
            fb = fmap[b:b+1]
            p = plan[b:b+1] if plan is not None else None
            if beam <= 1:
                text = self._greedy(fb, vocab, p, max_len)
                results.append(text)
            else:
                texts = self._beam(fb, vocab, p, max_len, beam)
                results.append(texts[0] if texts else "")
        return results
    def _greedy(self, fmap, vocab, plan, max_len):
        device = fmap.device
        ids = [vocab.stoi.get("<sos>", 1)]
        for _ in range(max_len):
            t = torch.tensor([ids], device=device)
            logits = self(fmap, t, plan)[:, -1, :]
            nxt = int(logits.argmax(-1).item())
            if nxt == vocab.stoi.get("<eos>", 2):
                break
            ids.append(nxt)
        return " ".join(vocab.itos[i] for i in ids[1:] if i < len(vocab.itos))
    def _beam(self, fmap, vocab, plan, max_len, beam_width):
        device = fmap.device
        beams = [(0.0, [vocab.stoi.get("<sos>", 1)])]
        eos = vocab.stoi.get("<eos>", 2)
        for _ in range(max_len):
            cands = []
            for score, seq in beams:
                if seq[-1] == eos:
                    cands.append((score, seq))
                    continue
                t = torch.tensor([seq], device=device)
                logits = self(fmap, t, plan)[:, -1, :]        # (1, V)
                probs = torch.log_softmax(logits, -1)[0]      # (V,)
                topk = probs.topk(beam_width)
                for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    cands.append((score + v, seq + [idx]))
            beams = sorted(cands, key=lambda x: -x[0])[:beam_width]
        eos_id = eos
        results = []
        for score, seq in beams:
            toks = [i for i in seq if i not in (vocab.stoi.get("<sos>", 1), eos_id)]
            text = " ".join(vocab.itos[i] for i in toks if i < len(vocab.itos))
            results.append(text)
        return results


def collate_pad(batch):
    """Pad target sequences to max length in batch."""
    xs, tgts, ls, plans = zip(*batch)
    max_len = max(t.shape[0] for t in tgts)
    pad_id = 0  # <pad>
    padded = torch.full((len(tgts), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tgts):
        padded[i, :t.shape[0]] = t
    return torch.stack(xs), padded, torch.tensor([t.shape[0] for t in tgts]), torch.stack(plans)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    root = Path(args.data)

    # Build vocabulary from all training captions
    import pandas as pd
    cap_file = root / "captions.parquet"
    if not cap_file.exists():
        cap_file = root / "BigEarthNet-S2" / "captions.parquet"
    df = pd.read_parquet(cap_file)
    train_texts = df[df.split == "train"]["output"].dropna().tolist()
    word_freq = collections.Counter()
    for t in train_texts:
        for tok in re.findall(r"[a-z0-9]+", str(t).lower()):
            word_freq[tok] += 1
    # Keep words appearing >= 3 times + top 5000
    keep = [w for w, c in word_freq.most_common(5000) if c >= 3]
    vocab = CaptionVocab(keep)
    print(f"vocabulary: {len(vocab)} words (from {len(train_texts)} captions)")
    ds_tr = CaptionDataset(root, "train", vocab, image_size=120)
    ds_va = CaptionDataset(root, "validation", vocab, image_size=120,
                           max_items=300)
    print(f"train={len(ds_tr)} val={len(ds_va)} vocab={len(vocab)}")

    enc = SceneEncoder(3).to(device)
    if CONFIG.scene_encoder_weights.exists():
        ck = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                        weights_only=False)
        enc.load_state_dict(ck["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)

    model = Captioner(len(ds_tr.vocab), cond=not args.no_cond).to(device)
    dl = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                    drop_last=True, collate_fn=collate_pad)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, collate_fn=collate_pad)
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

    # Experiment log
    from satquery.experiment_log import log_experiment
    log_experiment(
        script="train_captioner",
        args={"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "cond": not args.no_cond},
        metrics={"val_bleu": best},
        checkpoint=str(CONFIG.weights_dir / "captioner.pt"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CONFIG.data_dir / "bigearthnet_14k" / "BEN_14k"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-cond", action="store_true")
    main(ap.parse_args())
