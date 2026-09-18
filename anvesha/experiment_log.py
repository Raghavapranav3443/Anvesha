"""Lightweight experiment logging -- appends one JSON line per training run
to ``runs/experiments.jsonl``.  No MLflow/W&B dependency.

Usage from a training script::

    from anvesha.experiment_log import log_experiment
    log_experiment(
        script="train_vqa",
        args={"epochs": 20, "lr": 3e-4, "image_size": 192},
        metrics={"val_acc": 0.74},
        checkpoint="weights/vqa_head.pt",
    )
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def log_experiment(
    script: str,
    args: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Append one experiment record to ``runs/experiments.jsonl``.

    Returns the record dict for inspection.
    """
    from .config import CONFIG

    record: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": script,
        "args": args or {},
        "metrics": metrics or {},
    }
    if checkpoint:
        record["checkpoint"] = checkpoint
    if notes:
        record["notes"] = notes

    log_path = CONFIG.runs_dir / "experiments.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # best-effort -- never block training for logging

    return record


def recent_experiments(n: int = 20) -> list[Dict[str, Any]]:
    """Return the most recent *n* experiment records."""
    from .config import CONFIG

    log_path = CONFIG.runs_dir / "experiments.jsonl"
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()[-n:]
    records = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records
