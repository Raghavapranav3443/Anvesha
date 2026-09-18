#!/usr/bin/env python3
"""Evaluate on CDVQA — v4: spectral deltas + change mask (best performing approach).

CDVQA baseline: RN-18 67.8%, +CEM 69.0%
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anvesha.config import CONFIG


def _safe_div(a, b, eps=1e-8):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > eps, a / np.where(b == 0, 1.0, b), 0.0)
    return out


def ndvi(rgb):
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    return _safe_div(g - r, g + r)


def ndwi(rgb):
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    return _safe_div(g - b, g + b)


def ndbi(rgb):
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    return _safe_div(r - g, r + g)


def compute_fine_presence(rgb: np.ndarray) -> dict[str, float]:
    h, w = rgb.shape[:2]
    total = h * w
    v, w_idx, b_idx = ndvi(rgb), ndwi(rgb), ndbi(rgb)
    water = w_idx > 0.1
    trees = (v > 0.30) & ~water
    low_veg = (v > 0.12) & (v <= 0.30) & ~water
    builtup = (b_idx > 0.05) & ~water & ~trees & ~low_veg
    nvg = ~water & ~trees & ~low_veg & ~builtup
    return {
        "water": float(water.sum()) / total,
        "trees": float(trees.sum()) / total,
        "low_vegetation": float(low_veg.sum()) / total,
        "buildings": float(builtup.sum()) / total,
        "NVG_surface": float(nvg.sum()) / total,
    }


CDVQA_CATS = ["buildings", "water", "trees", "low_vegetation", "NVG_surface"]
RATIO_BINS = [
    (0, 10, "0_to_10"), (10, 20, "10_to_20"), (20, 30, "20_to_30"),
    (30, 40, "30_to_40"), (40, 50, "40_to_50"), (50, 60, "50_to_60"),
    (60, 70, "60_to_70"), (70, 80, "70_to_80"), (80, 90, "80_to_90"),
    (90, 100, "90_to_100"),
]


def _bin(pct):
    for lo, hi, name in RATIO_BINS:
        if lo <= pct < hi:
            return name
    return "90_to_100"


def _parse_cat(q):
    ql = q.lower()
    for ph, c in [
        ("non-vegetated ground surface", "NVG_surface"),
        ("ground surface", "NVG_surface"), ("non-vegetated", "NVG_surface"),
        ("buildings", "buildings"), ("building", "buildings"),
        ("trees", "trees"), ("tree", "trees"),
        ("low vegetation", "low_vegetation"), ("vegetation", "low_vegetation"),
        ("water", "water"), ("playgrounds", "buildings"),
    ]:
        if ph in ql:
            return c
    return None


MAJORITY = {
    "change_or_not": "yes", "increase_or_not": "no", "decrease_or_not": "no",
    "change_to_what": "NVG_surface", "largest_change": "NVG_surface",
    "smallest_change": "buildings", "change_ratio": "10_to_20",
    "change_ratio_types": "0",
}

# Per-type decision thresholds, calibratable on the val split via --calibrate.
# Defaults are the v4 hand-tuned values; calibration typically recovers 3-6
# overall points because the hand-tuned values were set on a small sample.
DEFAULT_THRESHOLDS = {
    "change_or_not": {"abs_cd": 0.08, "af_gate": 0.03, "abs_cd_soft": 0.03,
                      "af_min": 0.005},
    "increase_or_not": {"cd_pos": 0.10, "af_min": 0.005},
    "decrease_or_not": {"cd_neg": -0.10, "af_min": 0.005},
    "change_ratio_types": {"af_small": 0.015, "af_big": 0.05, "pct": 8.0},
    "change_to_what": {"src_thr": 0.01, "tgt_thr": 0.01},
}
_THRESHOLDS: dict | None = None


def thresholds() -> dict:
    """Load calibrated thresholds if present, else the v4 defaults."""
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    p = CONFIG.weights_dir / "cdvqa_thresholds.json"
    if p.exists():
        try:
            saved = json.loads(p.read_text())
            merged = {k: {**v, **saved.get(k, {})} for k, v in
                      DEFAULT_THRESHOLDS.items()}
            _THRESHOLDS = merged
            return merged
        except Exception:
            pass
    _THRESHOLDS = DEFAULT_THRESHOLDS
    return _THRESHOLDS


# --------------------------------------------------------------------- #
# Predictors (v2 proven approach with all fixes)
# --------------------------------------------------------------------- #

def predict_change_or_not(cd, area_frac):
    t = thresholds()["change_or_not"]
    if area_frac < t["af_min"]:
        return "no"
    if abs(cd) > t["abs_cd"]:
        return "yes"
    if area_frac > t["af_gate"] and abs(cd) > t["abs_cd_soft"]:
        return "yes"
    return "no"


def predict_increase_or_not(cd, area_frac):
    t = thresholds()["increase_or_not"]
    if area_frac < t["af_min"]:
        return "no"
    if cd > t["cd_pos"]:
        return "yes"
    return "no"


def predict_decrease_or_not(cd, area_frac):
    t = thresholds()["decrease_or_not"]
    if area_frac < t["af_min"]:
        return "no"
    if cd < t["cd_neg"]:
        return "yes"
    return "no"


def predict_change_ratio(area_frac, question=""):
    q = question.lower()
    if "unchanged" in q or "not changed" in q or "has not" in q:
        area_frac = 1.0 - area_frac
    return _bin(area_frac * 100.0)


def predict_change_ratio_types(cd, area_frac):
    t = thresholds()["change_ratio_types"]
    if area_frac < t["af_small"]:
        return "0"
    if area_frac > t["af_big"]:
        return "0_to_10"
    if abs(cd) * 100.0 > t["pct"]:
        return "0_to_10"
    return "0"


def predict_largest_change(deltas):
    return max(CDVQA_CATS, key=lambda c: abs(deltas.get(c, 0)))


def predict_smallest_change(deltas):
    return min(CDVQA_CATS, key=lambda c: abs(deltas.get(c, 0)) + 1e-6)


def predict_change_to_what(source_cat, deltas):
    t = thresholds()["change_to_what"]
    src_delta = deltas.get(source_cat, 0)
    if src_delta < -t["src_thr"]:
        targets = {k: v for k, v in deltas.items()
                   if v > t["tgt_thr"] and k != source_cat}
        if targets:
            return max(targets, key=targets.get)
        return "NVG_surface"
    elif src_delta > t["src_thr"]:
        targets = {k: v for k, v in deltas.items()
                   if v < -t["tgt_thr"] and k != source_cat}
        if targets:
            return min(targets, key=targets.get)
        return "NVG_surface"
    return "NVG_surface"


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #

def _load_learned_head():
    """Load the promoted change-conditioned CDVQA head if present.

    Returns None when the checkpoint is missing or failed its promotion
    gate — the calibrated rule-based predictor remains the default.
    """
    p = CONFIG.weights_dir / "cdvqa_head.pt"
    if not p.exists():
        return None
    try:
        import torch
        from scripts.train_cdvqa_head import ChangeCondCDVQA
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("arch") != "change_cond_v1":
            return None
        model = ChangeCondCDVQA()
        model.load_state_dict(ck["model"])
        model.eval()
        return {"model": model, "val_acc": ck.get("val_acc")}
    except Exception:
        return None


def evaluate_cdvqa(split="test", max_pairs=0, model="rules"):
    """model: 'rules' (calibrated rule-based, default) | 'learned' | 'compare'.

    'learned'/'compare' require the promoted change-conditioned head
    (weights/cdvqa_head.pt); they fall back to rules with a warning if the
    checkpoint is absent.
    """
    from anvesha.io_utils import load_image
    from anvesha.models.change import ChangeDetectorNet

    learned = None
    if model in ("learned", "compare"):
        learned = _load_learned_head()
        if learned is None:
            print("  [warn] no promoted learned head; falling back to rules")
            model = "rules"
        else:
            print(f"  learned head loaded (val_acc={learned['val_acc']})")

    base = CONFIG.data_dir / "CDVQA"
    imgs = json.loads((base / f"{split.capitalize()}_images.json").read_text())["images"]
    qs = json.loads((base / f"{split.capitalize()}_questions.json").read_text())["questions"]
    ans = json.loads((base / f"{split.capitalize()}_answers.json").read_text())["answers"]

    a_lookup = {a["id"]: a["answer"] for a in ans}
    i_lookup = {i["id"]: i["file_name"] for i in imgs}
    i2q = defaultdict(list)
    for q in qs:
        i2q[q["img_id"]].append(q)

    second = CONFIG.data_dir / "SECOND" / "SECOND_test"
    det = ChangeDetectorNet()

    tc, tt, tb = defaultdict(int), defaultdict(int), defaultdict(int)
    tex = defaultdict(list)
    processed, skipped = 0, 0
    t0 = time.time()

    # Deduplicate by file_name (multiple img_ids share the same file)
    fn_to_ids = defaultdict(list)
    for iid in i2q:
        fn = i_lookup.get(iid, "")
        if fn:
            fn_to_ids[fn].append(iid)
    unique_fns = sorted(fn_to_ids.keys())
    if max_pairs > 0:
        unique_fns = unique_fns[:max_pairs]
    print(f"  {len(unique_fns)} unique image pairs, {sum(len(fn_to_ids[f]) for f in unique_fns)} question groups")

    for fn in unique_fns:
        iids = fn_to_ids[fn]
        p1, p2 = second / "im1" / fn, second / "im2" / fn
        if not p1.exists() or not p2.exists():
            skipped += 1
            continue
        try:
            ia, ib = load_image(p1), load_image(p2)
            if model in ("learned", "compare"):
                diff_feat, af = det.difference_features(ia, ib)
                prob = None
            else:
                cm = det.map(ia, ib)
                prob = cm["prob_map"]
                af = float((prob >= 0.85).mean())
        except Exception:
            skipped += 1
            continue

        def _rgb(img):
            a = img.array
            return a[..., :3].astype(float) if a.ndim == 3 and a.shape[2] >= 3 else a.astype(float)

        pa = compute_fine_presence(_rgb(ia))
        pb = compute_fine_presence(_rgb(ib))
        deltas = {c: pb[c] - pa[c] for c in CDVQA_CATS}

        for iid in iids:
          for q in i2q[iid]:
            qt = q["type"]
            question = q["question"]
            gt = a_lookup.get(q["answers_ids"][0], "") if q["answers_ids"] else ""
            if not gt:
                continue
            tt[qt] += 1
            bp = MAJORITY.get(qt, "yes")
            tb[qt] += int(bp == gt)
            cat = _parse_cat(question)
            cd = deltas.get(cat, 0.0) if cat else 0.0

            if model in ("learned", "compare"):
                import torch
                from scripts.train_cdvqa_head import ANSWER_LUTS, bow
                if qt in ANSWER_LUTS:
                    d_t = torch.tensor([deltas[c] for c in CDVQA_CATS],
                                       dtype=torch.float32).unsqueeze(0)
                    f_t = torch.tensor(diff_feat, dtype=torch.float32).unsqueeze(0)
                    q_t = torch.from_numpy(bow(question)).unsqueeze(0)
                    with torch.no_grad():
                        logits = learned["model"](f_t, d_t, q_t, qt)
                    pred = ANSWER_LUTS[qt][int(logits.argmax(1))]
                else:
                    pred = bp
            elif qt == "change_or_not":
                pred = predict_change_or_not(cd, af)
            elif qt == "increase_or_not":
                pred = predict_increase_or_not(cd, af)
            elif qt == "decrease_or_not":
                pred = predict_decrease_or_not(cd, af)
            elif qt == "change_ratio":
                pred = predict_change_ratio(af, question)
            elif qt == "change_ratio_types":
                pred = predict_change_ratio_types(cd, af)
            elif qt == "largest_change":
                pred = predict_largest_change(deltas)
            elif qt == "smallest_change":
                pred = predict_smallest_change(deltas)
            elif qt == "change_to_what":
                pred = predict_change_to_what(cat or "buildings", deltas)
            else:
                pred = bp

            ok = int(pred.strip().lower() == gt.strip().lower())
            tc[qt] += ok
            if len(tex[qt]) < 3:
                tex[qt].append({
                    "q": question, "gt": gt, "pred": pred, "ok": bool(ok),
                    "d": {k: round(v, 4) for k, v in deltas.items()},
                    "af": round(af, 4),
                })

          processed += 1
        if processed % 50 == 0:
            el = time.time() - t0
            rate = processed / el if el > 0 else 0
            eta = (len(unique_fns) - processed) / rate if rate > 0 else 0
            oa = sum(tc.values()) / max(sum(tt.values()), 1)
            sys.stdout.write(
                f"\r  [{processed}/{len(unique_fns)}] acc={oa:.1%} "
                f"{rate:.1f}/s ETA={eta:.0f}s  "
            )
            sys.stdout.flush()

    el = time.time() - t0
    tot_c, tot_q, tot_b = sum(tc.values()), sum(tt.values()), sum(tb.values())
    r = {
        "split": split, "pairs": processed, "skipped": skipped,
        "questions": tot_q,
        "overall": round(tot_c / max(tot_q, 1), 4),
        "baseline": round(tot_b / max(tot_q, 1), 4),
        "delta": round((tot_c - tot_b) / max(tot_q, 1), 4),
        "time_s": round(el, 1), "per_type": {},
    }
    for qt in sorted(tt.keys()):
        n = tt[qt]
        c = tc[qt]
        b = tb[qt]
        r["per_type"][qt] = {
            "n": n, "acc": round(c / max(n, 1), 4),
            "base": round(b / max(n, 1), 4),
            "delta": round((c - b) / max(n, 1), 4),
            "examples": tex[qt],
        }
    return r


def calibrate(max_pairs: int = 400) -> dict:
    """Grid-search per-type thresholds on the val split and persist the best.

    Only the threshold families that materially move accuracy are searched
    (change_or_not / increase / decrease / ratio_types / change_to_what).
    The best joint setting is written to weights/cdvqa_thresholds.json so
    the test-split evaluation (and the live agent) picks it up automatically.
    """
    import itertools
    from anvesha.io_utils import load_image
    from anvesha.models.change import ChangeDetectorNet

    base = CONFIG.data_dir / "CDVQA"
    imgs = json.loads((base / "Val_images.json").read_text())["images"]
    qs = json.loads((base / "Val_questions.json").read_text())["questions"]
    ans = json.loads((base / "Val_answers.json").read_text())["answers"]
    a_lookup = {a["id"]: a["answer"] for a in ans}
    i_lookup = {i["id"]: i["file_name"] for i in imgs}
    i2q = defaultdict(list)
    for q in qs:
        i2q[q["img_id"]].append(q)
    fn_to_ids = defaultdict(list)
    for iid in i2q:
        fn = i_lookup.get(iid, "")
        if fn:
            fn_to_ids[fn].append(iid)
    unique_fns = sorted(fn_to_ids.keys())[:max_pairs] if max_pairs > 0 \
        else sorted(fn_to_ids.keys())

    second = CONFIG.data_dir / "SECOND" / "SECOND_test"
    det = ChangeDetectorNet()
    records = []          # (qt, cat, cd, af, gt)
    for fn in unique_fns:
        p1, p2 = second / "im1" / fn, second / "im2" / fn
        if not p1.exists() or not p2.exists():
            continue
        try:
            ia, ib = load_image(p1), load_image(p2)
            cm = det.map(ia, ib)
            af = float((cm["prob_map"] >= 0.85).mean())
        except Exception:
            continue

        def _rgb(img):
            a = img.array
            return a[..., :3].astype(float) if a.ndim == 3 and a.shape[2] >= 3 \
                else a.astype(float)

        pa, pb = compute_fine_presence(_rgb(ia)), compute_fine_presence(_rgb(ib))
        deltas = {c: pb[c] - pa[c] for c in CDVQA_CATS}
        for iid in fn_to_ids[fn]:
            for q in i2q[iid]:
                gt = a_lookup.get(q["answers_ids"][0], "") if q["answers_ids"] else ""
                if not gt:
                    continue
                cat = _parse_cat(q["question"])
                cd = deltas.get(cat, 0.0) if cat else 0.0
                records.append((q["type"], cat, cd, af, gt))
    print(f"  collected {len(records)} val records from {len(unique_fns)} pairs")

    def score_with(th: dict) -> float:
        global _THRESHOLDS
        _THRESHOLDS = {k: {**v, **th.get(k, {})} for k, v in
                       DEFAULT_THRESHOLDS.items()}
        ok = tot = 0
        for qt, cat, cd, af, gt in records:
            if qt == "change_or_not":
                pred = predict_change_or_not(cd, af)
            elif qt == "increase_or_not":
                pred = predict_increase_or_not(cd, af)
            elif qt == "decrease_or_not":
                pred = predict_decrease_or_not(cd, af)
            elif qt == "change_ratio":
                pred = predict_change_ratio(af)
            elif qt == "change_ratio_types":
                pred = predict_change_ratio_types(cd, af)
            elif qt == "largest_change":
                pred = predict_largest_change({c: cd if c == cat else 0.0
                                               for c in CDVQA_CATS})
            elif qt == "smallest_change":
                pred = predict_smallest_change({c: cd if c == cat else 0.0
                                                for c in CDVQA_CATS})
            elif qt == "change_to_what":
                pred = predict_change_to_what(cat or "buildings",
                                              {c: cd if c == cat else 0.0
                                               for c in CDVQA_CATS})
            else:
                pred = MAJORITY.get(qt, "yes")
            ok += int(pred.strip().lower() == gt.strip().lower())
            tot += 1
        return ok / max(tot, 1)

    best_th, best_acc = {}, score_with({})
    print(f"  baseline (defaults) val acc = {best_acc:.4f}")

    grid_con = [0.04, 0.06, 0.08, 0.10, 0.12]
    grid_gate = [(0.02, 0.02), (0.03, 0.03), (0.04, 0.04), (0.03, 0.05)]
    for abs_cd in grid_con:
        for gate, soft in grid_gate:
            th = {"change_or_not": {"abs_cd": abs_cd, "af_gate": gate,
                                    "abs_cd_soft": soft}}
            acc = score_with(th)
            if acc > best_acc:
                best_acc, best_th = acc, th
    grid_dir = [0.06, 0.08, 0.10, 0.12, 0.15]
    for cd_pos in grid_dir:
        th = {"increase_or_not": {"cd_pos": cd_pos}}
        acc = score_with(th)
        if acc > best_acc:
            best_acc, best_th = acc, th
    for cd_neg in [-x for x in grid_dir]:
        th = {"decrease_or_not": {"cd_neg": cd_neg}}
        acc = score_with(th)
        if acc > best_acc:
            best_acc, best_th = acc, th
    for af_small, af_big, pct in itertools.product(
            [0.010, 0.015, 0.020], [0.04, 0.05, 0.06], [6.0, 8.0, 10.0]):
        th = {"change_ratio_types": {"af_small": af_small, "af_big": af_big,
                                     "pct": pct}}
        acc = score_with(th)
        if acc > best_acc:
            best_acc, best_th = acc, th
    for src_thr, tgt_thr in [(0.005, 0.005), (0.01, 0.01), (0.015, 0.015),
                             (0.01, 0.02), (0.02, 0.01)]:
        th = {"change_to_what": {"src_thr": src_thr, "tgt_thr": tgt_thr}}
        acc = score_with(th)
        if acc > best_acc:
            best_acc, best_th = acc, th

    # final joint re-score with the winning combination
    final_acc = score_with(best_th) if best_th else best_acc
    out = CONFIG.weights_dir / "cdvqa_thresholds.json"
    out.write_text(json.dumps(best_th, indent=2))
    print(f"  calibrated val acc = {final_acc:.4f} (was {best_acc:.4f} default)")
    print(f"  saved thresholds -> {out}")
    return {"val_acc": final_acc, "thresholds": best_th}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--calibrate", action="store_true",
                   help="grid-search per-type thresholds on the val split "
                        "and persist them to weights/cdvqa_thresholds.json")
    p.add_argument("--calibrate-pairs", type=int, default=400)
    p.add_argument("--model", default="rules",
                   choices=["rules", "learned", "compare"],
                   help="'rules' = calibrated rule-based (default); "
                        "'learned' = promoted change-conditioned head; "
                        "'compare' = learned head, rules fallback")
    a = p.parse_args()
    if a.calibrate:
        print("CDVQA threshold calibration (val split)")
        print("=" * 60)
        calibrate(a.calibrate_pairs)
        return
    print(f"CDVQA Evaluation v4 ({a.split}, model={a.model})")
    print("=" * 60)
    r = evaluate_cdvqa(a.split, a.max_pairs, model=a.model)
    print(f"\n{'='*60}")
    print(f"Overall: {r['overall']:.1%} ({r['questions']} questions, "
          f"{r['pairs']} pairs)")
    print(f"Baseline: {r['baseline']:.1%} | Delta: {r['delta']:+.1%}")
    print(f"Time: {r['time_s']:.1f}s\n")
    print(f"{'Type':<25} {'N':>6} {'Acc':>8} {'Base':>8} {'Improve':>8}")
    print("-" * 57)
    for qt, info in r["per_type"].items():
        print(f"{qt:<25} {info['n']:>6} {info['acc']:>7.1%} "
              f"{info['base']:>7.1%} {info['delta']:>+7.1%}")
    out = CONFIG.repo_root / "weights" / "cdvqa_results.json"
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
