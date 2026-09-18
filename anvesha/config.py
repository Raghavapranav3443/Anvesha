"""Global configuration and label spaces for Anvesha AI."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tree(env: str, default: Path) -> Path:
    """A writable run-state tree, overridable by environment variable.

    Hosts disagree about which directories are writable. A Hugging Face Space
    replaces its code directory on every rebuild, and other platforms mount
    the image read-only, so pointing the two trees that are *written to* at a
    mount of the operator's choosing keeps the application working without a
    code change. Read-only trees (weights, artifacts) stay where they are.
    """
    raw = os.environ.get(env, "").strip()
    return Path(raw).expanduser().resolve() if raw else default

# BigEarthNet 19-class nomenclature (reBEN / BigEarthNet-19 labels)
BEN19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Agriculture with natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches dunes sands",
    "Inland waters",
    "Coastal wetlands",
    "Marine waters",
]

# EuroSAT classes used for the quick remote-sensing adaptation track
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Map coarse semantic concepts -> related BEN19 / EuroSAT labels (for captioning/VQA fallbacks)
CONCEPT_TO_BEN19 = {
    "built-up": ["Urban fabric", "Industrial or commercial units"],
    "water": ["Inland waters", "Marine waters", "Coastal wetlands"],
    "vegetation": ["Broad-leaved forest", "Coniferous forest", "Mixed forest",
                   "Natural grassland", "Moors and heathland",
                   "Transitional woodland/shrub", "Pastures"],
    "agriculture": ["Arable land", "Permanent crops", "Complex cultivation patterns",
                    "Agriculture with natural vegetation", "Agro-forestry areas"],
    "bare": ["Beaches dunes sands"],
}

CONCEPT_TO_EUROSAT = {
    "built-up": ["Industrial", "Residential"],
    "water": ["River", "SeaLake"],
    "vegetation": ["Forest", "HerbaceousVegetation", "Pasture"],
    "agriculture": ["AnnualCrop", "PermanentCrop"],
    "road": ["Highway"],
}


class Config:
    def __init__(self) -> None:
        self.repo_root = REPO_ROOT
        self.data_dir = _tree("ANVESHA_DATA_DIR", REPO_ROOT / "data")
        self.weights_dir = REPO_ROOT / "weights"
        self.runs_dir = _tree("ANVESHA_RUNS_DIR", REPO_ROOT / "runs")
        self.samples_dir = REPO_ROOT / "samples"
        self.artifacts_dir = REPO_ROOT / "artifacts"
        for d in (self.data_dir, self.weights_dir, self.runs_dir,
                  self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Model artefacts
        self.scene_encoder_weights = self.weights_dir / "scene_encoder.pt"
        self.vqa_weights = self.weights_dir / "vqa_head.pt"
        self.change_weights = self.weights_dir / "change_net.pt"
        self.fusion_weights = self.weights_dir / "optical_sar_fusion.pt"

        # Runtime limits (CPU/GPU friendly defaults)
        self.max_image_px = 2048          # longest edge downsampled to this
        self.device = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


    def artifact(self, name: str) -> Path:
        """Canonical, committed evidence path for ``name``.

        Benchmark and gate scripts write here, so re-running a gate updates the
        tracked evidence in place rather than scattering it into the ignored
        ``runs/`` scratch tree (see Decisions.md D25).
        """
        return self.artifacts_dir / name

    def evidence(self, name: str) -> Path:
        """Read path for committed evidence: ``artifacts/`` then ``runs/``.

        The ``runs/`` fallback keeps checkouts that predate the split working.
        """
        path = self.artifacts_dir / name
        return path if path.exists() else self.runs_dir / name


CONFIG = Config()
