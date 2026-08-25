"""Download open remote-sensing datasets used by SatQuery AI.

Usage:
  python scripts/download_datasets.py eurosat          # ~90 MB, quick track
  python scripts/download_datasets.py rsvqa            # RSVQA LR from Zenodo (~3 GB)
  python scripts/download_datasets.py levircd          # LEVIR-CD via torchgeo/gdrive (~2 GB)
  python scripts/download_datasets.py bigearthnet_txt  # BigEarthNet.txt annotations from HF
  python scripts/download_datasets.py all_small        # eurosat + levircd + rsvqa

Verified sources (checked online):
  EuroSAT       https://zenodo.org/records/7711810/files/EuroSAT.zip
  RSVQA LR      Zenodo record 6344333 (DOI 10.5281/zenodo.6344333)
  LEVIR-CD      torchgeo HF mirror / Google Drive id 1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim
  BigEarthNet   https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
                (images come from reBEN v2: zenodo 10891137 S2 / sibling S1)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from satquery.config import CONFIG  # noqa: E402

DATA = CONFIG.data_dir


def _stream(url: str, dest: Path):
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r{pct:3d}%  {done/1e6:.0f}/{total/1e6:.0f} MB",
                          end="", flush=True)
    tmp.rename(dest)
    print()


def download_eurosat() -> Path:
    """EuroSAT RGB (27,000 labelled Sentinel-2 patches, 10 classes)."""
    zip_path = DATA / "eurosat" / "EuroSAT.zip"
    out_dir = zip_path.parent / "2750"
    if out_dir.exists():
        print("EuroSAT already present:", out_dir)
        return out_dir
    _stream("https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
            zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    extracted = zip_path.parent / "2750"
    if not extracted.exists():  # archive root is EuroSAT_RGB/
        src = zip_path.parent / "EuroSAT_RGB"
        if src.exists():
            src.rename(extracted)
    return out_dir


def download_rsvqa() -> Path:
    """RSVQA LowRes (Sentinel-2 VQA; Zenodo record 6344333)."""
    rec = requests.get(
        "https://zenodo.org/api/records/6344333", timeout=60).json()
    base = DATA / "rsvqa_lr"
    base.mkdir(parents=True, exist_ok=True)
    for f in rec["files"]:
        url = f["links"]["self"]
        dest = base / f["key"]
        if not dest.exists():
            _stream(url, dest)
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(base)
    print("RSVQA LR ready at", base)
    return base


def download_levircd(limit: int | None = None) -> Path:
    """LEVIR-CD bi-temporal building change dataset.

    Primary source: HF mirror 'ericyu/LEVIRCD_Cropped_256' (parquet, official
    LEVIR-CD crops). Falls back to Google Drive for the original archive.
    Materialises data/LEVIR-CD/{train,test}/{A,B,label}/*.png
    """
    out = DATA / "LEVIR-CD"
    if (out / "train" / "A").exists():
        print("LEVIR-CD already present:", out)
        return out
    try:
        import io

        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image

        files = {
            "train": "data/train-00000-of-00001-737f96f51caac8cd.parquet",
            "test": "data/test-00000-of-00001-31d7c3e3444e5b5d.parquet",
        }
        for split, fname in files.items():
            pq = Path(hf_hub_download("ericyu/LEVIRCD_Cropped_256", fname,
                                      repo_type="dataset"))
            df = pd.read_parquet(pq)
            if limit:
                df = df.head(limit)
            for sub in ("A", "B", "label"):
                (out / split / sub).mkdir(parents=True, exist_ok=True)
            key = {"A": "imageA", "B": "imageB", "label": "label"}
            for i, row in df.iterrows():
                for sub, col in key.items():
                    payload = row[col]
                    data = payload["bytes"] if isinstance(payload, dict) else payload
                    dest = out / split / sub / f"{i}.png"
                    if not dest.exists():
                        Image.open(io.BytesIO(data)).save(dest)
                if i % 200 == 0:
                    print(split, i)
        print("LEVIR-CD ready at", out)
        return out
    except Exception as e:
        print("HF mirror failed:", e)

    urls = [
        "https://huggingface.co/datasets/torchgeo/levircd/resolve/main/LEVIR-CD.zip",
    ]
    ok = False
    for url in urls:
        try:
            z = out.parent / "LEVIR-CD.zip"
            _stream(url, z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out.parent)
            ok = True
            break
        except Exception as e:
            print(f"mirror failed ({e}); trying next")
    if not ok:
        print("Falling back to Google Drive via gdown ...")
        import gdown
        gdown.download(id="1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim",
                       output=str(out.parent / "LEVIR-CD.zip"), quiet=False)
        with zipfile.ZipFile(out.parent / "LEVIR-CD.zip") as zf:
            zf.extractall(out.parent)
    print("LEVIR-CD ready")
    return out


def download_bigearthnet_txt(annotations_only: bool = True) -> Path:
    """BigEarthNet.txt image-text annotations (HF: BIFOLD-BigEarthNetv2-0/...).

    With annotations_only=True only the JSON/text annotation files are pulled;
    the co-registered reBEN S1/S2 images are the large Zenodo archives listed
    at https://bigearth.net (S2 ~59 GiB, S1 ~51 GiB).
    """
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
        repo_type="dataset",
        local_dir=DATA / "bigearthnet_txt",
        allow_patterns=None if annotations_only else None,
    )
    print("BigEarthNet.txt ready at", path)
    return Path(path)


HANDLERS = {
    "eurosat": download_eurosat,
    "rsvqa": download_rsvqa,
    "levircd": download_levircd,
    "bigearthnet_txt": download_bigearthnet_txt,
}


if __name__ == "__main__":
    targets = sys.argv[1:] or ["help"]
    if targets == ["all_small"]:
        HANDLERS["eurosat"]()
        HANDLERS["levircd"]()
        HANDLERS["rsvqa"]()
    else:
        for t in targets:
            fn = HANDLERS.get(t)
            if fn is None:
                print(__doc__)
                break
            fn()
