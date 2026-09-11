"""B7 — calibration extension: record method labels for all heads.

The runtime half (``satquery.confmeta``) stamps every confidence with
``{value, method, n_cal}``; this script produces the ``weights/calibration.json``
sidecar it reads. Philosophy: be honest about what each head's confidence is.

  * VQA / count heads    -> temperature (reuses ``scripts/calibrate.py`` fit,
                            written into the checkpoint + mirrored here)
  * cdvqa head           -> temperature recorded from cdvqa_head.pt
  * change / fusion / grounding -> formula blend (closed-form evidence combo);
                            the equation is documented, never an invented number

Nothing here touches specialist class code; absent or partial calibration
falls back to ``method=formula`` at runtime (``confmeta.get``), so the system
degrades gracefully. Safe, offline, additive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.config import CONFIG


def _read_vqa_temperature() -> dict | None:
    """Reuse the existing VQA/count temperature fit (scripts/calibrate.py)."""
    p = CONFIG.weights_dir / "vqa_head.pt"
    if not p.exists():
        return None
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        t = float(ck.get("temperature", 1.0))
        return {"vqa": {"method": "temp", "param": t, "n_cal": None},
                "counting": {"method": "temp", "param": t, "n_cal": None,
                             "note": "shares VQA encoder family temperature"}}
    except Exception as e:
        print(f"vqa temperature read failed: {e}")
        return None


def _record_change() -> dict | None:
    if not CONFIG.change_weights.exists():
        return None
    return {"change": {"method": "formula",
                       "equation": "thresholded change-probability map; "
                                   "F1 on LEVIR-CD val (thr=0.85)",
                       "gate_reference": "runs/phase3_levir_fulltest.json"}}


def _record_cdvqa() -> dict | None:
    p = CONFIG.weights_dir / "cdvqa_head.pt"
    if not p.exists():
        return None
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        t = float(ck.get("temperature", 1.0))
        n = int(ck.get("n_cal", 0) or 0)
        return {"cdvqa": {"method": "temp", "param": t, "n_cal": n,
                          "gate": "beat calibrated rule-based 0.4942 "
                                  "(promoted val 0.7134, full-test 0.683)"}}
    except Exception:
        return {"cdvqa": {"method": "formula",
                          "equation": "change-conditioned learned head; "
                                      "full-test 0.683"}}


def _record_fusion() -> dict | None:
    if not CONFIG.fusion_weights.exists():
        return None
    return {"fusion": {"method": "formula",
                       "equation": "0.6*optical + 0.4*SAR evidence blend"}}


def _record_grounding() -> dict | None:
    return {"grounding": {"method": "formula",
                          "equation": "0.35*lex + 0.45*purity + 0.2*area",
                          "benchmark": "IoU@0.5 ~0.126; learned variants "
                                       "gated off (D15.1)"}}


def main():
    out = CONFIG.weights_dir / "calibration.json"
    cal: dict = {}
    for fn in (_read_vqa_temperature, _record_change, _record_cdvqa,
               _record_fusion, _record_grounding):
        try:
            r = fn()
            if r:
                cal.update(r)
        except Exception as e:
            print(f"calibrate_all: {fn.__name__} failed: {e}")
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"calibration -> {out} ({len(cal)} heads)")
    for k, v in cal.items():
        print(f"  {k}: method={v['method']}")


if __name__ == "__main__":
    main()
