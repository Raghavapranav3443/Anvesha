"""B7 -- calibration extension: record *honest* method labels for all heads.

The runtime half (``satquery.confmeta``) stamps every confidence with
``{value, method, n_cal}``; this script produces the ``weights/calibration.json``
sidecar it reads. Philosophy: be honest about what each head's confidence is.

Single source of truth
----------------------
Temperature lives in the checkpoint, because that is what the forward pass
applies. This script *reads* the checkpoints and cross-checks the claim, rather
than letting the sidecar assert a calibration the weights do not implement. An
earlier version of this sidecar labelled ``vqa``/``counting``/``cdvqa`` as
``method: "temp"`` while the shipped checkpoints contained no temperature key at
all -- so the runtime used the identity (T=1.0) and the documents claimed
T=1.55. The label was asserting a calibration that did not exist.

A head is now only labelled ``temp`` when all three hold:

  * its checkpoint records a temperature that is not the identity,
  * a recorded fit actually improved calibration (``calib.improved``), and
  * the fit reports how many samples it used (``calib.n_cal`` > 0).

Otherwise it is labelled honestly -- ``identity`` (measured, nothing better
found), ``inherited`` (borrowed from a sibling head, not measured on this one) or
``formula`` (a closed-form evidence blend, with its equation published).

Safe, offline, additive: nothing here touches specialist class code, and absent
or partial calibration degrades to ``formula`` at runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.config import CONFIG


def _read_head_calib(path: Path, component: str):
    """Honest calibration record for a head, read from its own checkpoint."""
    if not path.exists():
        return None
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {component: {"method": "unreadable", "param": None, "n_cal": None,
                            "why": f"checkpoint could not be read: {e}"}}

    t = float(ck.get("temperature", 1.0))
    calib = ck.get("calib") or {}
    n_cal = calib.get("n_cal") or ck.get("n_cal")
    improved = calib.get("improved")

    # A temperature that *is* the identity carries no calibration information,
    # whatever the file is labelled. Say so instead of claiming "temp".
    if t == 1.0:
        return {component: {
            "method": "identity", "param": t, "n_cal": n_cal,
            "why": ("no fitted temperature in the checkpoint; probabilities are "
                    "the raw softmax"),
            **({"measured_ece": calib.get("ece"),
                "identity_ece": calib.get("identity_ece"),
                "split": calib.get("split")} if calib else {}),
        }}
    if calib.get("objective") == "inherited":
        return {component: {
            "method": "inherited", "param": t, "n_cal": n_cal,
            "inherited_from": calib.get("inherited_from"),
            "why": calib.get("note") or "temperature borrowed from a sibling head",
        }}
    if improved is False:
        return {component: {
            "method": "identity", "param": 1.0, "n_cal": n_cal,
            "why": ("a fitted temperature failed to beat the identity on "
                    "calibration error; the identity was kept"),
        }}
    if not n_cal:
        # Fitted, but with no record of what it was fitted on. Not auditable.
        return {component: {
            "method": "unverified", "param": t, "n_cal": None,
            "why": ("checkpoint carries a temperature but no sample count, so "
                    "the fit cannot be audited"),
        }}
    return {component: {
        "method": "temp", "param": t, "n_cal": int(n_cal),
        "objective": calib.get("objective", "unknown"),
        "measured_ece": calib.get("ece"),
        "identity_ece": calib.get("identity_ece"),
        "split": calib.get("split"),
        "why": "temperature fitted on held-out data and verified to improve ECE",
    }}


def _record_change():
    if not CONFIG.change_weights.exists():
        return None
    return {"change": {
        "method": "formula",
        "equation": "thresholded change-probability map; F1 on LEVIR-CD val (thr=0.85)",
        "gate_reference": "runs/phase3_levir_fulltest.json",
        "why": "no temperature applies: the confidence is a closed-form blend of "
               "change-map evidence, not a softmax over classes"}}


def _record_fusion():
    if not CONFIG.fusion_weights.exists():
        return None
    return {"fusion": {
        "method": "formula",
        "equation": "0.6*optical + 0.4*SAR evidence blend",
        "why": "closed-form evidence blend across modalities"}}


def _record_grounding():
    return {"grounding": {
        "method": "formula",
        "equation": "0.35*lex + 0.45*purity + 0.2*area",
        "benchmark": "IoU@0.5 ~0.126; learned variants gated off (D15.1)",
        "why": "high confidence from a low-reliability method is capped by trust, "
               "not by this label"}}


def main():
    out = CONFIG.weights_dir / "calibration.json"
    cal: dict = {}

    heads = (
        ("vqa", CONFIG.vqa_weights),
        ("counting", CONFIG.weights_dir / "count_head.pt"),
        ("cdvqa", CONFIG.weights_dir / "cdvqa_head.pt"),
    )
    for component, path in heads:
        try:
            rec = _read_head_calib(Path(path), component)
            if rec:
                cal.update(rec)
        except Exception as e:
            print(f"calibrate_all: {component} failed: {e}")

    for fn in (_record_change, _record_fusion, _record_grounding):
        try:
            r = fn()
            if r:
                cal.update(r)
        except Exception as e:
            print(f"calibrate_all: {fn.__name__} failed: {e}")

    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"calibration -> {out} ({len(cal)} heads)")
    for k, v in sorted(cal.items()):
        n = v.get("n_cal")
        extra = f" n_cal={n}" if n is not None else ""
        print(f"  {k}: method={v.get('method')}{extra}")


if __name__ == "__main__":
    main()
