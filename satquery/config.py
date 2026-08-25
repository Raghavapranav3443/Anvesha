"""Global configuration and label spaces for SatQuery AI."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

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
        self.data_dir = REPO_ROOT / "data"
        self.weights_dir = REPO_ROOT / "weights"
        self.runs_dir = REPO_ROOT / "runs"
        self.samples_dir = REPO_ROOT / "samples"
        for d in (self.data_dir, self.weights_dir, self.runs_dir):
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


CONFIG = Config()
