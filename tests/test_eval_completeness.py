"""B10 eval completeness: VRSBench-VQA loader/protocol + SAC DRYRUN writer.
Model-inference runs are validated separately (bench needs torch + weights);
these tests stay fast and offline-safe."""
from __future__ import annotations

from pathlib import Path

import pytest

from anvesha.evaluate import (_norm_answer, _vrsbench_val_qa, _write_dryrun,
                               bench_vrsbench_vqa)


def test_norm_answer_protocol():
    assert _norm_answer("The Windmill") == "windmill"
    assert _norm_answer("  Windmill! ") == "windmill"
    assert _norm_answer("Built-up area") == "built up area"
    assert _norm_answer("AN  answer") == "answer"


def test_vrsbench_qa_loader_frozen_order():
    import json
    ann_dir = Path("data/vrsbench/Annotations_val")
    if not ann_dir.exists():
        pytest.skip("data/vrsbench not present on this machine")
    items = _vrsbench_val_qa(limit=25)
    assert items and len(items) == 25
    # independent re-derivation of the first item (frozen sorted order)
    for jf in sorted(ann_dir.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        qas = [qa for qa in d.get("qa_pairs", [])
               if qa.get("question") and qa.get("answer")]
        if qas and (ann_dir.parent / "Images_val" / f"{jf.stem}.png").exists():
            assert items[0]["image"].name == f"{jf.stem}.png"
            assert items[0]["gold"] == str(qas[0]["answer"])
            break
    assert all(it["question"] and it["gold"] for it in items)


def test_bench_vrsbench_vqa_guard_zero_n():
    assert bench_vrsbench_vqa(0) is None


def test_bench_vrsbench_vqa_loader_guards_missing_data(tmp_path, monkeypatch):
    from anvesha import evaluate as ev
    monkeypatch.setattr(ev.CONFIG, "data_dir", tmp_path)
    assert bench_vrsbench_vqa(10) is None


def test_dryrun_writer_rows_and_status():
    md = Path("runs/_test_dryrun.md")
    rows = [
        {"group": "siteA", "task": "change_analysis", "status": "ok",
         "answer": "Changed area ~1.2 ha", "confidence": 0.7,
         "mask_geotiff": "runs/x/visuals/change_mask.tif"},
        {"group": "siteB", "error": "ValueError: bad pair"},
    ]
    out = _write_dryrun(md, Path("runs/_sac_test"), rows, "q?")
    text = out.read_text(encoding="utf-8")
    assert "# SAC dry run log" in text
    assert "| siteA | change_analysis | PASS |" in text
    assert "| siteB |  | FAIL |" in text
    assert "withheld by ISRO/SAC" in text
    md.unlink(missing_ok=True)
