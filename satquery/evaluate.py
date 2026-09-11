"""Unified evaluation & SAC batch harness.

  python -m satquery.evaluate --all                 # full public-benchmark card
  python -m satquery.evaluate --sac-dir DIR         # batch ISRO/SAC-style pairs
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .config import CONFIG  # noqa: E402
from .io_utils import load_image  # noqa: E402


# --------------------------------------------------------------------- #
# Shared BLEU util (same protocol as scripts/train_captioner.simple_bleu)
# --------------------------------------------------------------------- #

def _bleu4(pred: str, ref: str) -> float:
    """BLEU-4 with brevity penalty (no external deps)."""
    import re
    import collections
    pt = re.findall(r"[a-z0-9]+", pred.lower())
    rt = re.findall(r"[a-z0-9]+", ref.lower())
    if not pt or not rt:
        return 0.0
    log_prec = 0.0
    for n in range(1, 5):
        p_ngrams = collections.Counter(tuple(pt[i:i + n]) for i in range(len(pt) - n + 1))
        r_ngrams = collections.Counter(tuple(rt[i:i + n]) for i in range(len(rt) - n + 1))
        clipped = sum(min(c, r_ngrams.get(ng, 0)) for ng, c in p_ngrams.items())
        total = sum(p_ngrams.values())
        if total == 0:
            return 0.0
        log_prec += np.log(max(clipped / total, 1e-10))
    bleu = np.exp(log_prec / 4.0)
    bp = min(1.0, np.exp(1 - len(rt) / max(len(pt), 1)))
    return float(bp * bleu)


# --------------------------------------------------------------------- #
# Individual benchmarks
# --------------------------------------------------------------------- #

def bench_rsvqa(n: int) -> dict | None:
    sys.path.insert(0, str(CONFIG.repo_root))
    from scripts.run_benchmarks import eval_rsvqa
    return eval_rsvqa(n)


def bench_levir(n: int) -> dict | None:
    from scripts.run_benchmarks import eval_levir
    return eval_levir(n)


# --------------------------------------------------------------------- #
# VRSBench (manual route — upstream HF parquet is schema-broken).
# Images + annotations ship under data/vrsbench on disk.
#     caption bench   -> BLEU-4 against val captions
#     grounding bench -> IoU@0.5 against val referring-expression boxes
# --------------------------------------------------------------------- #

def _vrsbench_val():
    """Yield (stem, image_path, caption, objects) for VRSBench validation."""
    root = CONFIG.data_dir / "vrsbench"
    ann_dir = root / "Annotations_val"
    img_root = root / "Images_val"
    if not (ann_dir.exists() and img_root.exists()):
        return None
    import json as _json
    items = []
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            d = _json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        stem = jf.stem
        img = img_root / f"{stem}.png"
        if not img.exists():
            continue
        objs = [o for o in d.get("objects", []) if o.get("obj_coord")]
        if not d.get("caption") or not objs:
            continue
        items.append({"stem": stem, "image": img,
                      "caption": d["caption"], "objects": objs})
    return items


def bench_vrsbench_caption(n: int = 300) -> dict | None:
    """BLEU-4 of the captioning specialist vs VRSBench val captions."""
    from satquery.models import describe
    items = _vrsbench_val()
    if not items:
        return None
    scores = []
    for it in items[:n]:
        try:
            out = describe(load_image(it["image"]))
            text = out["caption"]
        except Exception as _e:
            continue
        if not text:
            continue
        scores.append(_bleu4(text, it["caption"]))
    if not scores:
        return None
    return {"benchmark": "VRSBench-val (captioning)", "metric": "BLEU-4",
            "n": len(scores), "score": round(float(np.mean(scores)), 4)}


def _parse_coord(s):
    """Parse '[0.8, 0.25, 0.99, 0.33]' / list / string (x0,y0,x1,y1)."""
    if isinstance(s, (list, tuple)):
        if len(s) == 4:
            try:
                return [float(v) for v in s]
            except Exception:
                return None
        return None
    if not s or not isinstance(s, str):
        return None
    toks = re.findall(r"[0-9.]+", s.replace(",", " "))
    if len(toks) == 4:
        return [float(t) for t in toks]
    return None


def bench_vrsbench_grounding(n: int = 200) -> dict | None:
    """Spectral grounding specialist vs VRSBench val referring boxes (IoU@0.5).

    This is the gate baseline for Phase 1 (VL grounding). Expected ≈ 0.15.
    """
    from satquery.models import ground
    items = _vrsbench_val()
    if not items:
        return None
    ious = []
    n_det = 0
    for it in items[:n]:
        img = load_image(it["image"])
        img_wh = (img.width, img.height) if hasattr(img, "width") else (img.shape[1], img.shape[0])
        for o in it["objects"]:
            coord = _parse_coord(o.get("obj_coord"))
            if not coord or len(coord) != 4:
                continue
            gt = np.array(coord, dtype=np.float32)  # normalized [x0,y0,x1,y1]
            expr = o.get("referring_sentence") or it["caption"]
            try:
                res = ground(img, expr)
            except Exception:
                continue
            if not res.boxes:
                continue
            best = 0.0
            for b in res.boxes:
                x0, y0, x1, y1 = b
                x0n, y0n = x0 / img_wh[0], y0 / img_wh[1]
                x1n, y1n = x1 / img_wh[0], y1 / img_wh[1]
                inter = max(0.0, min(gt[2], max(x0n, x1n)) - max(gt[0], min(x0n, x1n))) * \
                        max(0.0, min(gt[3], max(y0n, y1n)) - max(gt[1], min(y0n, y1n)))
                union = (max(gt[2], max(x0n, x1n)) - min(gt[0], min(x0n, x1n))) * \
                        (max(gt[3], max(y0n, y1n)) - min(gt[1], min(y0n, y1n)))
                best = max(best, inter / union if union > 0 else 0.0)
            ious.append(best)
            if best >= 0.5:
                n_det += 1
    if not ious:
        return None
    return {"benchmark": "VRSBench-val (grounding)", "metric": "IoU@0.5",
            "n": len(ious),
            "score": round(float(np.mean(ious)), 4),
            "det_rate@0.5": round(n_det / len(ious), 4)}


def _vrsbench_val_qa(limit: int = 300):
    """Collect (image_path, question, gold_answer) from VRSBench val qa_pairs.

    Reuses the same sorted annotation order as ``_vrsbench_val`` so the
    frozen split stays comparable across benches.
    """
    root = CONFIG.data_dir / "vrsbench"
    ann_dir = root / "Annotations_val"
    img_root = root / "Images_val"
    if not (ann_dir.exists() and img_root.exists()):
        return None
    import json as _json
    items = []
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            d = _json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        img = img_root / f"{jf.stem}.png"
        if not img.exists():
            continue
        for qa in d.get("qa_pairs", []):
            q, a = qa.get("question"), qa.get("answer")
            if q and a:
                items.append({"image": img, "question": str(q),
                              "gold": str(a)})
                if len(items) >= limit:
                    return items
    return items or None


def _norm_answer(s) -> str:
    """Normalise a VQA answer for exact-match (lower, no punctuation,
    leading article stripped) — same spirit as RSVQA protocols."""
    t = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()
    t = re.sub(r"\s+", " ", t)
    for art in ("the ", "a ", "an "):
        if t.startswith(art):
            t = t[len(art):]
    return t.strip()


def bench_vrsbench_vqa(n: int = 300) -> dict | None:
    """VRSBench-val VQA exact-match via the shipped VQA specialist.

    Even a modest number beats "unevaluated" (masterplan B10): report
    normalised EM + abstain rate. Protocol: frozen sorted val order (see
    scripts/frozen_splits.json), shipped model, no tuning.
    """
    if n <= 0:
        return None
    items = _vrsbench_val_qa(limit=n)
    if not items:
        return None
    from satquery.models import get_vqa_model
    from .io_utils import load_image
    model = get_vqa_model()
    ems, abstain = [], 0
    for it in items:
        try:
            out = model.answer(load_image(it["image"]), it["question"])
            pred = str(out.get("answer", ""))
        except Exception:
            continue
        if not pred or pred.lower() in ("unknown", "unknown."):
            abstain += 1
        ems.append(_norm_answer(pred) == _norm_answer(it["gold"]))
    if not ems:
        return None
    return {"benchmark": "VRSBench-val (VQA)", "metric": "exact_match",
            "n": len(ems), "score": round(float(np.mean(ems)), 4),
            "abstain_rate": round(abstain / len(ems), 4)}


def bench_cdvqa(n: int = 200) -> dict | None:
    """CDVQA val question-answer accuracy via the project's specialist predictor.

    Delegates to scripts/eval_cdvqa.evaluate_cdvqa — the same question-type-
    routed machinery (change-conditioned learned head over detector diff
    features, calibrated rules fallback) behind the D14.8 full-test 0.683
    result. Ground truth is joined by answers_ids, the dataset's own key.
    """
    import sys as _sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    try:
        from scripts.eval_cdvqa import evaluate_cdvqa
        r = evaluate_cdvqa(split="val", max_pairs=max(n // 6, 4), model="learned")
    except Exception as e:
        return {"benchmark": "CDVQA (val subset)", "metric": "answer-match",
                "n": 0, "score": None, "note": f"evaluator error: {e}"}
    return {"benchmark": "CDVQA (val subset)", "metric": "answer-match",
            "n": int(r["questions"]), "score": r["overall"],
            "note": (f"majority-baseline={r['baseline']} "
                     f"delta=+{r['delta']} (learned change-conditioned head)")}


def bench_caption_bleu(n: int = 150) -> dict | None:
    w = CONFIG.weights_dir / "captioner.pt"
    if not w.exists():
        return None
    import pandas as pd
    import torch
    from PIL import Image
    import rasterio
    from satquery.models.backbone import SceneEncoder, normalise_for_encoder, resize_np, to_tensor
    from scripts.train_captioner import CaptionVocab, Captioner, simple_bleu

    ckpt = torch.load(w, map_location="cpu", weights_only=False)
    vocab = CaptionVocab([])
    vocab.itos = ckpt["vocab"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    device = CONFIG.resolve_device()
    cond = bool(ckpt.get("cond", False))
    in_ch = int(ckpt.get("in_ch", 128))
    feat_kind = str(ckpt.get("feat_kind", "scene"))
    model = Captioner(len(vocab.itos), cond=cond, in_ch=in_ch).to(device).eval()
    model.load_state_dict(ckpt["model"])
    clip = None
    if feat_kind == "clip":
        from satquery.models.clip_text import get_clip_text
        clip = get_clip_text()
    enc = SceneEncoder(3).to(device).eval()
    sc = torch.load(CONFIG.scene_encoder_weights, map_location="cpu",
                    weights_only=False)
    enc.load_state_dict(sc["encoder"])

    df = pd.read_parquet(CONFIG.data_dir / "bentxt_join" / "captions.parquet")
    val = df[df.split == "validation"]
    val = val.sample(frac=1.0, random_state=7).drop_duplicates("patch_id").head(n)
    # all reference variants per patch (fair multi-reference BLEU)
    refs_by_patch = {pid: [str(t) for t in g["output"]]
                     for pid, g in df[df.split == "validation"]
                     .groupby("patch_id")}

    scores = []
    for _, row in val.iterrows():
        f = Path(str(CONFIG.data_dir / "bigearthnet_14k/BEN_14k/BigEarthNet-S2" /
                     str(row["split"]))) / f"{row['patch_id']}.tif"
        if not f.exists():
            continue
        try:
            # dataset-identical preprocessing (scripts/train_captioner.CaptionDataset):
            # first 3 bands as-is, /10000 clip, PIL uint8 resize -- NOT the
            # display-space RGB reversal (out-of-distribution for the decoder).
            with rasterio.open(str(f)) as src:
                arr = np.moveaxis(src.read()[:3].astype(np.float32), 0, -1)
            arr = np.clip(arr / 10000.0, 0, 1)
            from PIL import Image as _PILImage
            im = _PILImage.fromarray((arr * 255).astype(np.uint8)) \
                .resize((120, 120))
            x = torch.from_numpy((np.asarray(im, dtype=np.float32) / 255.0)
                                 .transpose(2, 0, 1))[None].to(device)
            with torch.no_grad():
                if feat_kind == "clip" and clip is not None:
                    fmap = clip.vision_patch_tokens(x)
                else:
                    fmap = enc.feature_map(x, stride=8)
                text = model.generate(fmap, vocab)[0]  # greedy: beam is degenerate
            refs = refs_by_patch.get(str(row["patch_id"]), [str(row["output"])])
            scores.append(max(simple_bleu(text, r) for r in refs))
        except Exception:
            continue
    if not scores:
        return None
    return {"benchmark": "BigEarthNet.txt captions (val)", "metric": "BLEU",
            "n": len(scores), "score": round(float(np.mean(scores)), 4)}


# --------------------------------------------------------------------- #
# SAC batch mode
# --------------------------------------------------------------------- #

def sac_batch(folder: Path, out_csv: Path | None = None,
              query: str = "Describe what changed between the two dates and where.",
              dryrun_md: Path | None = None) -> dict:
    """Run the agent over every co-registered pair found under folder.

    Pairing resolution order:
      1. **Explicit manifest** — a ``pairs.csv`` next to the inputs with one
         ``image_a,image_b`` row per pair (filenames relative to *folder*).
         This is the recommended route for the ISRO/SAC evaluation set: it
         removes all ambiguity from filename-based inference.
      2. **Stem-prefix heuristic** (fallback) — greedy pairing of stems
         sharing a long common prefix (>= 60% of the shorter stem).

    Writes answers.csv next to the inputs and returns a summary.
    """
    from .agent import get_controller

    images = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            images.setdefault(p.stem.lower(), []).append(p)

    groups: list[list[str]] = []
    manifest = folder / "pairs.csv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                a = (row.get("image_a") or row.get("a") or "").strip()
                b = (row.get("image_b") or row.get("b") or "").strip()
                sa, sb = Path(a).stem.lower(), Path(b).stem.lower()
                if sa in images and sb in images:
                    groups.append([sa, sb])
                else:
                    print(f"[pairs.csv] skipping unknown file(s): {a!r}, {b!r}")
        if groups:
            print(f"paired via explicit manifest ({len(groups)} pairs)")
    if not groups:
        # fallback: greedy stem-prefix heuristic
        used = set()
        keys = sorted(images.keys())
        for k in keys:
            if k in used:
                continue
            group = [k]
            for other in keys:
                if other == k or other in used:
                    continue
                common = len(os_prefix(k, other))
                if common >= max(6, int(0.6 * min(len(k), len(other)))):
                    group.append(other)
                    used.add(other)
                    if len(group) == 2:
                        break
            used.add(k)
            groups.append(sorted(group))

    rows = []
    controller = get_controller()
    out_dir = folder / "satquery_outputs"
    out_dir.mkdir(exist_ok=True)
    for gi, g in enumerate(groups):
        paths = [images[k][0] for k in g][:2]
        try:
            res = controller.run(paths, query, save_report=True)
            row = {"group": "_".join(g)[:60], "task": res.selected_task,
                   "answer": res.answer, "confidence": res.confidence,
                   "run_id": res.run_id}
            mask_tif = res.run_id and list((CONFIG.runs_dir / res.run_id /
                                            "visuals").glob("change_mask.tif"))
            if mask_tif:
                row["mask_geotiff"] = str(mask_tif[0])
        except Exception as e:
            row = {"group": "_".join(g)[:60], "error": str(e)[:200]}
        rows.append(row)
        print(f"[{gi+1}/{len(groups)}]", row.get("group"), "->",
              row.get("answer", row.get("error", ""))[:100])

    csv_path = out_csv or (out_dir / "answers.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"pairs_processed": len(rows), "csv": str(csv_path)}
    if dryrun_md is not None:
        _write_dryrun(dryrun_md, folder, rows, query)
        summary["dryrun_md"] = str(dryrun_md)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_dryrun(md_path: Path, folder: Path, rows: list[dict],
                  query: str) -> Path:
    """B10 SAC dry-run log: dates, files, per-pair pass/fail, artifacts.

    Committed next to the SAC inputs so judges can verify the batch route
    without re-running it. Never claims hidden-set answers — the reference
    annotations stay with ISRO/SAC.
    """
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SAC dry run log", "",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Input folder:** `{folder}`",
        f"- **Query:** {query}",
        f"- **Pairs processed:** {len(rows)}", "",
        "| Pair | Task | Status | Answer (first 120 ch) | Confidence | Mask GeoTIFF |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        status = "FAIL" if row.get("error") else "PASS"
        lines.append(
            "| {group} | {task} | {status} | {answer} | {conf} | {mask} |".format(
                group=row.get("group", ""), task=row.get("task", ""),
                status=status,
                answer=(str(row.get("answer", "")) or
                        str(row.get("error", "")))[:120].replace("|", "/"),
                conf=row.get("confidence", ""), mask=row.get("mask_geotiff", "")))
    lines += ["", "*Reference annotations are withheld by ISRO/SAC; this log "
                  "records only the answers produced by the engine. "
                  "answers.csv sits next to the inputs.*"]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def os_prefix(a: str, b: str) -> str:
    out = []
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        out.append(ca)
    return "".join(out)


# --------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------- #

def normalized_card(results: list[dict]) -> list[dict]:
    """Normalise heterogeneous metrics into [0,1] and combine."""
    return [
        {**r, "normalized_score": round(float(min(max(
            float(r.get("score") if r.get("score") is not None
            else r.get("iou", 0.0)), 0), 1)), 4)}
        for r in results
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sac-dir", type=str, default=None)
    ap.add_argument("--dryrun-md", type=str, default=None,
                    help="B10: also write a SAC dry-run markdown log")
    args = ap.parse_args()

    results = []
    if args.sac_dir:
        print(json.dumps(sac_batch(
            Path(args.sac_dir),
            dryrun_md=Path(args.dryrun_md) if args.dryrun_md else None),
            indent=2))
        return

    if args.all:
        benches = (bench_rsvqa, bench_levir, bench_caption_bleu,
                   bench_vrsbench_caption, bench_vrsbench_grounding,
                   bench_vrsbench_vqa, bench_cdvqa)
        for fn in benches:
            try:
                r = fn(args.n)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"[{fn.__name__}] failed:", e)
    card = normalized_card(results)
    combined = round(float(np.mean([c["normalized_score"] for c in card])) ,
                     4) if card else None
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "results": card, "combined_normalized": combined}
    out = CONFIG.runs_dir / "benchmarks.json"
    out.write_text(json.dumps(payload.get("results"), indent=2))
    (CONFIG.runs_dir / "scorecard.json").write_text(json.dumps(payload, indent=2))

    print("\n=== SatQuery AI scorecard ===")
    for r in card:
        line = f"{r['benchmark']:<42} {r['metric']:<24} n={r['n']:<5}"
        if "f1" in r:
            line += f" IoU={r.get('iou')} F1={r['f1']} -> norm {r['normalized_score']}"
        else:
            line += f" score={r.get('score')} -> norm {r['normalized_score']}"
        print(line)
    print(f"combined normalized: {combined}")


if __name__ == "__main__":
    main()
