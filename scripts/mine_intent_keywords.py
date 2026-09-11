"""B2 — mine intent keywords from RSVQA/CDVQA train questions (deterministic).

Reads the project's own benchmark question files (already on disk in data/),
extracts the top-N content tokens per intent bucket, and writes:
  * weights/task_keywords.json   -> {task: [tokens]}  (consumed by satquery.rerank)
  * scripts/golden_intent.json   -> [{query, expected_task}] golden set
Run offline; no network, no models. Safe to re-run (idempotent output).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

STOP = {"the", "a", "an", "is", "are", "of", "in", "on", "this", "that",
        "to", "and", "or", "at", "it", "be", "for", "with", "what", "where",
        "which", "how", "has", "have", "there", "between", "these", "did",
        "do", "does", "can", "any"}

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{1,}")


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


CHANGE_Q = {"what changed", "how much", "has the", "did the", "have the"}
GROUND = {"highlight", "locate", "where is", "where are", "mark", "find",
          "show me", "localise", "localize", "outline"}
FUSE = {"optical", "sar", "radar", "cross-modal", "sentinel", "modality",
        "modalities", "fuse", "fused"}
CAP = {"describe", "land-cover", "scene", "caption", "what is shown",
       "major objects", "summarise", "summarize"}


def bucket(text: str) -> str | None:
    q = text.lower()
    # strongest multi-word signals first (phrase-level, not token-level)
    if "compare these" in q or "compare the" in q or \
            q.startswith("describe the change"):
        return "change_analysis"
    if "investigate" in q or "investigation" in q or "impact of" in q:
        return "investigation"
    for phrase in CHANGE_Q:
        if q.startswith(phrase):
            return "change_vqa"
    toks = set(tokenize(text))
    if toks & {"change", "changed", "changes", "before", "after", "dates",
               "increased", "decreased", "remained"}:
        return "change_vqa"
    if toks & {"highlight", "locate", "localise", "localize", "mark"}:
        return "grounding"
    if "where is" in q or "where are" in q or "show me" in q:
        return "grounding"
    if toks & {"sar", "radar", "cross-modal", "modality", "modalities"}:
        return "optical_sar"
    if toks & CAP or "what is shown" in q:
        return "captioning"
    if any(q.startswith(p) for p in ["is there", "are there", "does this",
                                     "how many", "is it", "are they"]):
        return "single_vqa"
    return None


def mine_rsvqa(n_per_task: int = 50) -> dict:
    f = DATA / "rsvqa_lr" / "LR_split_train_questions.json"
    if not f.exists():
        return {}
    qs = json.loads(f.read_text(encoding="utf-8")).get("questions", [])
    counts = Counter()
    for q in qs:
        text = q.get("question", "") if isinstance(q, dict) else str(q)
        for tok in tokenize(text):
            counts[tok] += 1
    top = [t for t, _ in counts.most_common(n_per_task)]
    return {"single_vqa": top, "captioning": top[: n_per_task // 2]}


def mine_cdvqa(n_per_task: int = 50) -> dict:
    results = {}
    for split in ("Val", "Train"):
        f = DATA / "CDVQA" / f"{split}_questions.json"
        if not f.exists():
            continue
        qs = json.loads(f.read_text(encoding="utf-8")).get("questions", [])
        counts = Counter()
        for q in qs:
            text = q.get("question", "") if isinstance(q, dict) else str(q)
            for tok in tokenize(text):
                counts[tok] += 1
        top = [t for t, _ in counts.most_common(n_per_task)]
        results["change_vqa"] = top
        results["change_analysis"] = top
    return results


def build_golden(n_golden: int = 500) -> list:
    """Golden intent set spanning all tasks, deterministic order."""
    seeds = [
        ("Is there a water area?", "single_vqa"),
        ("How many buildings are there?", "single_vqa"),
        ("Is it a rural or an urban area", "single_vqa"),
        ("Describe the land-cover of this image.", "captioning"),
        ("What is shown in this image?", "captioning"),
        ("Highlight the water body in this image.", "grounding"),
        ("Where is the vegetation located?", "grounding"),
        ("Show me where the buildings are.", "grounding"),
        ("What changed between these two dates?", "change_vqa"),
        ("Has the built-up area increased or decreased?", "change_vqa"),
        ("Did the areas of trees change?", "change_vqa"),
        ("Fuse the optical and SAR evidence.", "optical_sar"),
        ("Use the optical and SAR image together.", "optical_sar"),
        ("Compare these two dates and describe the change.", "change_analysis"),
        ("Investigate the change around the water body.", "investigation"),
    ]
    # Balanced-ish padding: interleave RSVQA (single/caption/ground) with
    # CDVQA (change) 2:1 so no task dominates the B2 metric.
    rsvqa, cdvqa = [], []
    rsvqa_f = DATA / "rsvqa_lr" / "LR_split_train_questions.json"
    if rsvqa_f.exists():
        qs = json.loads(rsvqa_f.read_text(encoding="utf-8")).get("questions", [])
        for q in qs:
            text = q.get("question", "") if isinstance(q, dict) else str(q)
            b = bucket(text)
            if b in ("single_vqa", "captioning", "grounding"):
                rsvqa.append({"query": text, "expected_task": b})
            if len(rsvqa) >= n_golden:
                break
    for split in ("Val", "Train"):
        f = DATA / "CDVQA" / f"{split}_questions.json"
        if not f.exists():
            continue
        qs = json.loads(f.read_text(encoding="utf-8")).get("questions", [])
        for q in qs:
            text = q.get("question", "") if isinstance(q, dict) else str(q)
            b = bucket(text)
            if b in ("change_vqa", "change_analysis"):
                cdvqa.append({"query": text, "expected_task": b})
            if len(cdvqa) >= n_golden // 2:
                break
        if len(cdvqa) >= n_golden // 2:
            break
    quota = n_golden - len(seeds)
    extra = []
    for i in range(max(len(rsvqa), len(cdvqa))):
        if len(extra) >= quota:
            break
        for _ in range(2):
            if i < len(rsvqa) and len(extra) < quota:
                extra.append(rsvqa[i])
        if i < len(cdvqa) and len(extra) < quota:
            extra.append(cdvqa[i])
    golden = [{"query": q, "expected_task": t} for q, t in seeds] + extra
    return golden[:n_golden]


def eval_golden() -> dict:
    """B2 metric: intent accuracy of classify_task (+rerank) on the golden set."""
    from satquery.agent import classify_task
    from satquery.rerank import re_rank
    golden = build_golden(500)
    CFG = {"change_vqa": "bitemporal_pair",
           "change_analysis": "bitemporal_pair",
           "impact_analysis": "bitemporal_pair",
           "investigation": "bitemporal_pair",
           "optical_sar": "optical_sar_pair"}
    ok = rerank_ok = n = 0
    by_task: dict = {}
    for item in golden:
        q, exp = item["query"], item["expected_task"]
        cfg = CFG.get(exp, "single")
        info = classify_task(q, cfg)
        base = info["task"]
        ranked = list(info.get("ranked_candidates", [])) or [(base, 1.0)]
        rr, _ = re_rank(q, ranked, top_k=1, configuration=cfg)
        after = rr[0][0] if rr else base
        n += 1
        ok += base == exp
        rerank_ok += after == exp
        d = by_task.setdefault(exp, {"correct": 0, "total": 0})
        d["correct"] += after == exp
        d["total"] += 1
    result = {
        "n": n,
        "accuracy_classify_task": round(ok / max(n, 1), 4),
        "accuracy_with_rerank": round(rerank_ok / max(n, 1), 4),
        "per_task": {t: {**v, "acc": round(v["correct"] / max(v["total"], 1), 3)}
                     for t, v in sorted(by_task.items())},
        "note": ("mined golden set (miner heuristic labels); the real judge "
                 "criterion is the observable trace, not internal routing"),
    }
    out = REPO / "scripts" / "golden_accuracy.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"golden_accuracy -> {out}")
    print(f"  n={result['n']}  classify={result['accuracy_classify_task']}  "
          f"+rerank={result['accuracy_with_rerank']}")
    for t, v in result["per_task"].items():
        print(f"  {t:<18} {v['correct']}/{v['total']} = {v['acc']}")
    return result


def main():
    if "--eval" in sys.argv:
        eval_golden()
        return
    kw = {}
    kw.update(mine_rsvqa())
    kw.update(mine_cdvqa())
    kw_path = REPO / "weights" / "task_keywords.json"
    kw_path.write_text(json.dumps(kw, indent=2), encoding="utf-8")
    print(f"task_keywords -> {kw_path} ({sum(len(v) for v in kw.values())} tokens, "
          f"{len(kw)} tasks)")
    golden = build_golden(500)
    gold_path = REPO / "scripts" / "golden_intent.json"
    gold_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"golden_intent -> {gold_path} ({len(golden)} queries)")


if __name__ == "__main__":
    main()

