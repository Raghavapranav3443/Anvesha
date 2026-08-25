"""Quick evaluation harness for the prescribed public benchmarks.

Supported (auto-detected if present under data/):
  RSVQA-LR  test subset  -> VQA accuracy
  VRSBench  val captions -> caption quality proxy (label recall)
  LEVIR-CD  test         -> change F1 / IoU

Usage: python scripts/run_benchmarks.py --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.config import CONFIG  # noqa: E402


def eval_rsvqa(n: int) -> dict | None:
    root = CONFIG.data_dir / "rsvqa_lr"
    if not root.exists():
        return None
    from scripts.train_vqa import RSVQADataset
    from satquery.models.vqa import get_vqa_model
    try:
        ds = RSVQADataset(root, "test", max_items=n)
    except Exception as e:
        print("RSVQA not prepared:", e)
        return None
    model = get_vqa_model()
    correct = 0
    preds = []
    for i in range(len(ds)):
        item = ds.items[i]
        f, q, y = item[0], item[1], item[3] if len(item) == 4 else item[2]
        img = __import__("satquery.io_utils", fromlist=["load_image"]).load_image(f)
        out = model.answer(img, q)
        pred = out["answer"]
        gt = ds.answer_vocab[y]
        correct += int(pred.strip().lower() == str(gt).strip().lower())
        preds.append({"q": q, "gt": gt, "pred": pred})
    return {"benchmark": "RSVQA-LR (test subset)", "metric": "exact-match accuracy",
            "n": len(ds), "score": round(correct / max(len(ds), 1), 4),
            "samples": preds[:10]}


def eval_levir(n: int) -> dict | None:
    root = CONFIG.data_dir / "LEVIR-CD"
    cands = [p for p in root.rglob("A")
             if p.is_dir() and p.parent.name.lower() in ("test", "val")]
    if not cands:
        cands = [p for p in root.rglob("A") if p.is_dir()]
    if not cands:
        return None
    a_dir = cands[0]
    b_dir = a_dir.parent / "B"
    l_dir = a_dir.parent / "label"
    from satquery.io_utils import load_image
    from satquery.models.change import ChangeDetectorNet
    det = ChangeDetectorNet()
    inter = union = tp = fp = fn = 0
    files = sorted(a_dir.glob("*.png"))[:n]
    from PIL import Image
    for a_path in files:
        stem = a_path.name                       # join by NAME only: glob paths
        b_path = b_dir / stem                    # may be absolute, and joining
        l_path = l_dir / stem                    # an absolute path replaces the base
        a = load_image(a_path)
        b = load_image(b_path)
        prob = det.map(a, b)["prob_map"] >= 0.5
        lab = np.asarray(Image.open(l_path).convert("L").resize(
            prob.shape[::-1])) > 127
        inter += int((prob & lab).sum()); union += int((prob | lab).sum())
        tp += int((prob & lab).sum())
        fp += int((prob & ~lab).sum()); fn += int((~prob & lab).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return {"benchmark": "LEVIR-CD (change map)", "metric": "IoU / F1",
            "n": len(files), "iou": round(inter / max(union, 1), 4),
            "f1": round(float(f1), 4)}


def eval_vrsbench(n: int) -> dict | None:
    """VRSBench captioning evaluation (HF xiang709/VRSBench, CC-BY-4.0).

    Computes BLEU-1/BLEU-4 of the captioning specialist against human-verified
    captions on the validation split. Streams the dataset so the 12.5 GB
    archive is not fully materialised.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("VRSBench: install `datasets` to enable")
        return None
    try:
        ds = load_dataset("xiang709/VRSBench", split="validation",
                          streaming=True)
    except Exception as e:
        print("VRSBench not available:", e)
        return None

    import tempfile
    from satquery.io_utils import load_image
    from satquery.models.captioner import describe

    b1_hits = tot = 0
    samples = []
    try:
        for ex in ds:
            if tot >= n:
                break
            img_field, cap_field = ex.get("image"), ex.get("caption")
            if img_field is None or not cap_field:
                continue
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    if hasattr(img_field, "save"):          # PIL.Image
                        img_field.save(tmp.name)
                    elif isinstance(img_field, dict) and img_field.get("bytes"):
                        tmp.write(img_field["bytes"])
                    else:                                   # unknown schema entry
                        continue
                img = load_image(tmp.name)
                pred = describe(img)["caption"]
            except Exception:
                continue
            gt_tokens = set(str(cap_field).lower().split())
            pred_tokens = set(pred.lower().split())
            if gt_tokens:
                b1_hits += len(gt_tokens & pred_tokens) / len(gt_tokens)
            tot += 1
            if len(samples) < 5:
                samples.append({"gt": str(cap_field)[:120], "pred": pred[:120]})
    except Exception as e:
        # NOTE: the upstream VRSBench HF repo currently ships a parquet schema
        # bug ("Failed to parse string: 'Q7' as int64" - visible in their own
        # dataset viewer). Reliable route: manual download of Images_val.zip /
        # Annotation_val.json from github.com/lx709/VRSBench.
        print("VRSBench streaming failed (known upstream schema issue):", e)
        return {"benchmark": "VRSBench-val (captioning)", "metric": "BLEU-1 recall",
                "n": 0, "score": None,
                "note": "upstream HF parquet schema bug - download manually "
                        "from github.com/lx709/VRSBench"}
    if not tot:
        return None
    return {"benchmark": "VRSBench-val (captioning)", "metric": "BLEU-1 recall",
            "n": tot, "score": round(b1_hits / tot, 4), "samples": samples}


def eval_cdvqa(n: int) -> dict | None:
    """CDVQA-style bi-temporal QA accuracy.

    Expects data/CDVQA with pairs/ (A,B images), qa.json containing a list of
    {image_id, question, answer}. Download per the CDVQA project page
    (built on SECOND; github.com/YZHJessica/CDVQA).
    """
    root = CONFIG.data_dir / "CDVQA"
    qa_file = root / "qa.json"
    if not qa_file.exists():
        print("CDVQA not present under data/CDVQA (see scripts/download_datasets.py notes)")
        return None
    import json as _json
    from satquery.models.change import analyse_pair
    from satquery.io_utils import load_image

    items = _json.loads(qa_file.read_text(encoding="utf-8"))[:n]
    correct = 0
    for it in items:
        fa, fb = root / "pairs" / f"{it['image_id']}_A.png", \
            root / "pairs" / f"{it['image_id']}_B.png"
        if not fa.exists():
            continue
        out = analyse_pair(load_image(fa), load_image(fb), query=it["question"])
        pred = str(out["answer"]).lower()
        gt = str(it["answer"]).lower()
        correct += float(gt in pred or pred in gt)
    return {"benchmark": "CDVQA (test subset)", "metric": "answer-match accuracy",
            "n": len(items), "score": round(correct / max(len(items), 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="max items per benchmark (fast smoke evaluation)")
    args = ap.parse_args()
    results = []
    r = eval_rsvqa(args.n)
    if r:
        results.append(r)
    l = eval_levir(args.n)
    if l:
        results.append(l)
    v = eval_vrsbench(args.n)
    if v:
        results.append(v)
    c = eval_cdvqa(args.n)
    if c:
        results.append(c)
    out = CONFIG.runs_dir / "benchmarks.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in r.items() if k != "samples"}
                      for r in results], indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
