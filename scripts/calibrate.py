"""Temperature scaling for specialist heads: minimises *calibration error*.

  python scripts/calibrate.py            # calibrates VQA + count head

Objective note -- this used to minimise NLL, which was wrong for this project.
NLL rewards sharpening the whole distribution and is dominated by the probability
assigned to the true class; the number Anvesha actually displays is a confidence
a non-expert acts on, so the honest objective is expected calibration error.
Measured on RSVQA-LR val (n=4096), the two objectives disagree in *direction*:

    T=1.7 (NLL-optimal)  -> ECE 0.076   (worse than shipping nothing)
    T=1.0 (identity)     -> ECE 0.025
    T=0.8 (ECE-optimal)  -> ECE 0.017

So the selector below optimises ECE and refuses to return a temperature that
fails to beat the identity, because a "calibration" that degrades the number
users see is worse than no calibration at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satquery.calib_metrics import select_temperature
from satquery.config import CONFIG
from satquery.models.backbone import SceneEncoder
from satquery.models.vqa import _FusionHead, _hashed_bow, infer_question_type
from scripts.train_vqa import RSVQADataset


def collect_logits(encoder, head, ds, bow_dim, type_vocab, device,
                   input_size, limit=4000):
    # Measurement must be deterministic. Left in the caller's default training
    # mode, dropout/BatchNorm make the logits stochastic -- verified: six
    # repeated forwards on identical input give different logits in train mode
    # and identical logits in eval mode. Fitting a temperature to that noise
    # produces a number that looks like calibration but is not, so the mode is
    # forced here rather than left to each caller.
    encoder.eval()
    head.eval()
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


def nll_at(logits, ys, t: float) -> float:
    """Negative log-likelihood at a temperature, for reporting only."""
    return float(nn.functional.cross_entropy(logits / max(float(t), 1e-6), ys).item())


def confidence_curves(logits, temperatures):
    """Max-softmax confidence per temperature, as NumPy, from one logit set."""
    curves = {}
    for t in temperatures:
        p = torch.softmax(logits / max(float(t), 1e-6), dim=-1)
        curves[float(t)] = p.max(dim=-1).values.detach().cpu().numpy()
    return curves


def best_temperature(logits, ys):
    """ECE-optimal temperature, never worse than the identity.

    Selection lives in ``satquery.calib_metrics`` so it can be unit-tested
    without loading a model.
    """
    correct = (logits.argmax(dim=-1) == ys).detach().cpu().numpy()
    candidates = sorted(set(round(float(t), 3)
                            for t in np.linspace(0.3, 4.0, 75)) | {1.0})
    pick = select_temperature(candidates,
                              confidence_curves(logits, candidates),
                              correct, identity=1.0)
    pick["nll"] = round(nll_at(logits, ys, pick["temperature"]), 4)
    return pick


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
    enc.eval()
    head = _FusionHead(len(ck["answer_vocab"]), len(type_vocab)).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    ds = RSVQADataset(CONFIG.data_dir / "rsvqa_lr", "val",
                      image_size=int(ck.get("input_size", 128)),
                      type_vocab=type_vocab)
    logits, ys = collect_logits(enc, head, ds, bow_dim, type_vocab, device,
                                int(ck.get("input_size", 128)))
    pick = best_temperature(logits, ys)
    t = pick["temperature"]
    if pick["improved"]:
        print(f"VQA temperature={t}  ECE {pick['identity_ece']} -> {pick['ece']}"
              f"  (val NLL {pick['nll']}, n={len(ys)})")
    else:
        print(f"VQA temperature left at identity ({t}): no fitted value beat "
              f"ECE {pick['identity_ece']} on val (n={len(ys)}). Reported as "
              f"uncalibrated rather than fitted.")

    # Record the provenance of the number so the runtime can verify the claim
    # instead of trusting a hand-written label (see satquery.confmeta.verify).
    ck["temperature"] = t
    ck["n_cal"] = int(len(ys))
    ck["calib"] = {"objective": "ece", "ece": pick["ece"],
                   "identity_ece": pick["identity_ece"],
                   "improved": bool(pick["improved"]),
                   "split": "rsvqa_lr/val", "n_cal": int(len(ys)),
                   "nll": pick["nll"]}
    torch.save(ck, path)

    cpath = CONFIG.weights_dir / "count_head.pt"
    if cpath.exists():
        ckc = torch.load(cpath, map_location="cpu", weights_only=False)
        ckc["temperature"] = t
        # Inherited, not measured. The count head has its own output distribution
        # (ordinal soft-CE), so reusing the VQA temperature is an assumption. It
        # is labelled as such rather than presented as an independent fit.
        ckc["calib"] = {"objective": "inherited", "inherited_from": "vqa",
                        "note": "count head shares the VQA encoder family; not "
                                "independently calibrated"}
        torch.save(ckc, cpath)
        print(f"count head temperature inherited from VQA ({t}) -- labelled "
              f"inherited, not measured")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calibrate_vqa(device)
