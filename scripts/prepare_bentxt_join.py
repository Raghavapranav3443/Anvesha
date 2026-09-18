"""Materialise BigEarthNet.txt annotations joined to our local 14K patches.

Produces data/bentxt_join/{captions.parquet, refs.parquet} used by
scripts/train_captioner.py and scripts/train_grounding.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).resolve().parents[1] / "data" / "bentxt_join"


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from anvesha.config import CONFIG

    local = {p.stem for p in CONFIG.data_dir.joinpath(
        "bigearthnet_14k", "BEN_14k", "BigEarthNet-S2").rglob("*.tif")}
    print("local patches:", len(local))

    pq = hf_hub_download("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
                         "BigEarthNet.txt.parquet", repo_type="dataset")
    df = pd.read_parquet(pq, columns=["patch_id", "type", "category",
                                      "split", "input", "output"])
    sub = df[df.patch_id.isin(local)]
    OUT.mkdir(parents=True, exist_ok=True)
    caps = sub[sub.type == "captioning"].reset_index(drop=True)
    refs = sub[(sub.type == "bounding box") &
               (sub.category == "reference")].reset_index(drop=True)
    caps.to_parquet(OUT / "captions.parquet")
    refs.to_parquet(OUT / "refs.parquet")
    print(f"captions: {len(caps)} -> {OUT/'captions.parquet'}")
    print(f"refs    : {len(refs)} -> {OUT/'refs.parquet'}")


if __name__ == "__main__":
    main()
