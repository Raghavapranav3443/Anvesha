"""Calibration audit -- does the reported confidence match observed accuracy?

Measures the shipped head; fits nothing. It answers the question the project's
positioning depends on: when Anvesha says 80%, is it right about 80% of the time?

It reports, at the identity temperature and at the checkpoint temperature:

  * accuracy, mean confidence, and the accuracy-confidence gap
  * expected calibration error (ECE) and Brier score
  * whether any prediction's argmax changed (temperature scaling must not)

and writes ``weights/calibration_report.json`` containing a *reliability table* --
confidence band -> measured accuracy on held-out data. The decision layer reads
that table instead of trusting the model's self-assessment, so advice given to a
non-expert is anchored to how often this system has actually been right at that
confidence, not to how sure it sounds.

The metric implementations live in ``satquery.calib_metrics`` (pure NumPy,
unit-tested); this script only supplies the model and the data.

  python scripts/eval_calibration.py [--limit 4000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from satquery.calib_metrics import (brier_score, expected_calibration_error,
                                    judge, reliability_table)
from satquery.config import CONFIG
from scripts.calibrate import collect_logits
from satquery.models.backbone import SceneEncoder
from satquery.models.vqa import _FusionHead
from scripts.train_vqa import RSVQADataset


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=-1)


def _load_vqa():
    ck = torch.load(CONFIG.vqa_weights, map_location="cpu", weights_only=False)
    enc = SceneEncoder(3)
    enc.load_state_dict(ck["encoder"])
    head = _FusionHead(len(ck["answer_vocab"]), len(ck.get("type_vocab", {}) or {}))
    head.load_state_dict(ck["head"])
    # Deterministic measurement; see the note in scripts/calibrate.collect_logits.
    enc.eval()
    head.eval()
    return ck, enc, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--json", default=str(CONFIG.weights_dir / "calibration_report.json"))
    args = ap.parse_args()

    ck, enc, head = _load_vqa()
    bow_dim = int(ck.get("bow_dim", 512))
    type_vocab = ck.get("type_vocab", {}) or {}
    fitted_t = float(ck.get("temperature", 1.0))
    input_size = int(ck.get("input_size", 128))

    ds = RSVQADataset(CONFIG.data_dir / "rsvqa_lr", "val",
                      image_size=input_size, type_vocab=type_vocab)
    logits, ys = collect_logits(enc, head, ds, bow_dim, type_vocab, "cpu",
                               input_size, limit=args.limit)
    if len(ys) == 0:
        print("no val samples; cannot audit")
        return 1

    print(f"=== VQA head calibration audit (held-out val, n={len(ys)}) ===")
    print(f"checkpoint temperature: {fitted_t}"
          f"  (n_cal={ck.get('n_cal', '<absent>')})")
    print()

    results = {}
    argmax_at_identity = None
    for label, t in (("identity", 1.0), ("checkpoint", fitted_t)):
        p = _probs(logits, t)
        conf, pred = p.max(dim=-1)
        correct = (pred == ys).numpy()
        conf_np = conf.detach().cpu().numpy()
        if label == "identity":
            argmax_at_identity = pred.detach().cpu().numpy()
        results[label] = {
            "temperature": t,
            "accuracy": round(float(correct.mean()), 4),
            "mean_confidence": round(float(conf_np.mean()), 4),
            "brier": round(brier_score(conf_np, correct), 4),
            "ece": round(expected_calibration_error(conf_np, correct), 4),
            "reliability_table": reliability_table(conf_np, correct),
            **judge(conf_np, correct),
        }
        r = results[label]
        print(f"[{label}] T={t:.3f}")
        print(f"  accuracy        {r['accuracy']:.4f}")
        print(f"  mean confidence {r['mean_confidence']:.4f}")
        print(f"  gap             {r['gap']:+.4f}  ({r['verdict']})")
        print(f"  ECE (15 bins)   {r['ece']:.4f}")
        print(f"  Brier           {r['brier']:.4f}")
        print()

    # Temperature scaling is a monotone rescaling of the logits, so argmax --
    # and therefore accuracy -- is invariant. Verify rather than assert.
    p2 = _probs(logits, fitted_t)
    same = bool((p2.max(dim=-1).indices.detach().cpu().numpy()
                 == argmax_at_identity).all())
    print(f"argmax unchanged by rescaling: {same} "
          f"(accuracy is invariant as required)")

    table = results["checkpoint"]["reliability_table"]
    print()
    print("=== reliability table (use this, not the model's opinion) ===")
    print(f"  {'band':<12} {'n':>6} {'claimed':>8} {'observed':>9} {'optimism':>9}  usable")
    for r in table:
        if not r["n"]:
            continue
        print(f"  {r['band']:<12} {r['n']:>6} {r['claimed_mean']:>8.3f} "
              f"{r['observed_accuracy']:>9.3f} {r.get('optimism', 0.0):>+9.3f}  "
              f"{'yes' if r['sufficient'] else 'small n'}")

    payload = {
        "component": "vqa",
        "n_val": int(len(ys)),
        "checkpoint_temperature": fitted_t,
        "checkpoint_calib": ck.get("calib"),
        "argmax_invariant": same,
        "identity": results["identity"],
        "checkpoint": results["checkpoint"],
        "note": ("Reliability measured on RSVQA-LR val. Bands with small n are "
                 "weak evidence; ``optimism`` positive means we claimed more "
                 "than we delivered."),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
