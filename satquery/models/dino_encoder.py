"""DINOv2 visual backbone wrapper (gated adoption — see scripts/gate_dinov2.py).

Duck-types the SceneEncoder call sites used by the VQA type heads:
`encoder(x)` with x = [0,1]-scaled RGB tensor (B,3,H,W) -> 256-d embedding.
Input size must be a multiple of 14 (patch size); we use 224.
Weights are bundled at weights/dinov2_vits14.pt so the system stays
offline-deployable. Apache-2.0 license (facebookresearch/dinov2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    FEATURE_DIM = 256
    INPUT_SIZE = 224

    def __init__(self, projected_dim: int = 256,
                 weights: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        w = Path(weights) if weights else None
        self.backbone = None
        if w is not None and w.exists():            # offline bundle first
            try:
                bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    pretrained=False, trust_repo=True)
                bb.load_state_dict(torch.load(w, map_location="cpu",
                                              weights_only=False))
                self.backbone = bb
            except Exception:
                self.backbone = None
        if self.backbone is None:                   # hub (cache or download)
            try:
                self.backbone = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
            except Exception:
                hub_file = Path(torch.hub.get_dir()) / "checkpoints" \
                    / "dinov2_vits14_pretrain.pth"
                bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    source="local", pretrained=False,
                                    trust_repo=True)
                bb.load_state_dict(torch.load(hub_file, map_location="cpu",
                                              weights_only=False))
                self.backbone = bb
        for p in self.backbone.parameters():        # frozen SSL features
            p.requires_grad_(False)
        self.proj = nn.Linear(384, projected_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept [0,1] RGB like the SceneEncoder call sites; resize to /14 grid
        if x.shape[-1] != self.INPUT_SIZE:
            x = nn.functional.interpolate(
                x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        cls = self.backbone(x)                    # B x 384
        return self.proj(cls)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                       # keep SSL tower frozen-mode
        return self
