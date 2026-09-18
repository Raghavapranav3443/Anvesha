"""Download VRSBench validation images + annotations for the grounding head.

  python scripts/download_vrsbench.py [--images]
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anvesha.config import CONFIG

BASE = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/"
OUT = CONFIG.data_dir / "vrsbench"


def fetch(name: str):
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 1e6:
        print("have", name)
        return dest
    from scripts.download_datasets import _stream
    _stream(BASE + name, dest)
    return dest


def main(with_images: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    fetch("Annotations_val.zip")
    with zipfile.ZipFile(OUT / "Annotations_val.zip") as zf:
        zf.extractall(OUT)
        print("annotations:", [n for n in zf.namelist()][:8])
    if with_images:
        z = fetch("Images_val.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(OUT)
        print("images extracted")


if __name__ == "__main__":
    main(with_images="--images" in sys.argv)
