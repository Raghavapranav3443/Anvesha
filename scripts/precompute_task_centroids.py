"""Precompute task centroids from RSVQA training questions.

Instead of computing centroids from keyword lists (which only capture keyword
similarity), this script reads the actual training questions and computes
per-type BOW centroids.  This gives the routing system a much better
representation of what real user queries look like for each task type.

Usage:
    python scripts/precompute_task_centroids.py

Outputs: weights/task_centroids.pt (torch dict mapping type_name -> 512-d vector)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.config import CONFIG


def bow(text: str, dim: int = 512) -> np.ndarray:
    """Hashed bag-of-words vector."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.md5(tok.encode()).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def main():
    root = CONFIG.data_dir / "rsvqa_lr"
    if not root.exists():
        print(f"RSVQA-LR not found at {root}")
        return

    # Load training questions
    qs_file = root / "LR_split_train_questions.json"
    if not qs_file.exists():
        print(f"Questions file not found: {qs_file}")
        return

    qs = json.loads(qs_file.read_text(encoding="utf-8"))["questions"]

    # Collect questions by type
    type_questions: dict[str, list[str]] = {}
    for q in qs:
        if not q.get("active", True) or not q.get("answers_ids"):
            continue
        qtype = str(q.get("type", "other"))
        question = q.get("question", "")
        if question.strip():
            type_questions.setdefault(qtype, []).append(question)

    print("Question counts by type:")
    for t, qs_list in sorted(type_questions.items(), key=lambda x: -len(x[1])):
        print(f"  {t}: {len(qs_list)}")

    # Compute per-type centroids
    centroids = {}
    dim = 512
    for qtype, questions in type_questions.items():
        vecs = [bow(q, dim) for q in questions]
        mean = np.mean(vecs, axis=0)
        n = float(np.linalg.norm(mean))
        centroids[qtype] = mean / n if n > 0 else mean

    # Save
    import torch
    out_path = CONFIG.weights_dir / "task_centroids.pt"
    torch.save({k: torch.from_numpy(v) for k, v in centroids.items()}, out_path)
    print(f"\nSaved {len(centroids)} centroids to {out_path}")
    for name, vec in centroids.items():
        print(f"  {name}: norm={np.linalg.norm(vec):.4f}")


if __name__ == "__main__":
    main()
