"""SAC-style batch mode smoke test: temp folder of pairs -> answers.csv."""
import sys
from pathlib import Path

import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.config import CONFIG
from anvesha.evaluate import sac_batch


def test_sac_batch(tmp_path):
    src = CONFIG.samples_dir
    pairs = [
        ("siteA_t1.tif", "demo_change_2020.tif"),
        ("siteA_t2.tif", "demo_change_2024.tif"),
        ("siteB_opt.tif", "demo_isroformat_optical.tif"),
        ("siteB_sar.tif", "demo_isroformat_sar.tif"),
    ]
    for name, src_name in pairs:
        shutil.copy(src / src_name, tmp_path / name)

    summary = sac_batch(tmp_path, query="Describe what changed between "
                                         "the two dates and where.")
    csv_path = Path(summary["csv"])
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8").lower()
    assert "sitea" in text and "siteb" in text
    assert "task" in text and "confidence" in text
    assert summary["pairs_processed"] == 2
