"""Unified evaluation & SAC batch harness.

  python -m satquery.evaluate --all                 # full public-benchmark card
  python -m satquery.evaluate --sac-dir DIR         # batch ISRO/SAC-style pairs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .config import CONFIG  # noqa: E402


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
    model = Captioner(len(vocab.itos), cond=cond).to(device).eval()
    model.load_state_dict(ckpt["model"])
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
            with rasterio.open(f) as src:
                arr = np.moveaxis(src.read().astype(np.float32), 0, -1)
            rgb = np.clip(arr[..., [2, 1, 0]] / 10000.0, 0, 1)
            x = to_tensor(resize_np(rgb, 120)).to(device)
            with torch.no_grad():
                fmap = enc.feature_map(x, stride=8)
                text = model.generate(fmap, vocab, beam=3)[0]
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
              query: str = "Describe what changed between the two dates and where.") -> dict:
    """Run the agent over every co-registered pair found under folder.

    Expected layout (flexible): folders or files named such that an optical
    and SAR/before-after pairing can be inferred by shared stem prefix.
    Writes answers.csv next to the inputs and returns a summary.
    """
    from .agent import get_controller

    images = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            images.setdefault(p.stem.lower(), []).append(p)

    groups = []
    used = set()
    keys = sorted(images.keys())
    for k in keys:
        if k in used:
            continue
        group = [k]
        # greedy: pair stems sharing a long common prefix
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


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
    args = ap.parse_args()

    results = []
    if args.sac_dir:
        print(json.dumps(sac_batch(Path(args.sac_dir)), indent=2))
        return

    if args.all:
        for fn in (bench_rsvqa, bench_levir, bench_caption_bleu):
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
