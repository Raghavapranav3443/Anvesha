"""Density-map counting head (v5) - model definition.

Shared by scripts/train_count_density.py (training) and
satquery.models.vqa (inference). Kept inside the package so the runtime
import never depends on the scripts/ directory being importable.
"""
from __future__ import annotations

import torch.nn as nn


class DensityHead(nn.Module):
    """Predicts a stride-8 density map from the encoder's feature map, plus
    an auxiliary digit-classification head on globally pooled features.

    Trained with count-only supervision: the density map's sum is regressed
    to the total count (L1), with an ordinal-CE digit head as an auxiliary
    task. At inference the density sum and the digit head are combined:
    density wins when it agrees with the digit head within +-1.
    """

    def __init__(self, in_ch: int = 128, n_classes: int = 10):
        super().__init__()
        self.density = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))
        self.digit = nn.Sequential(
            nn.Linear(in_ch, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, fmap):
        dmap = self.density(fmap)                       # B x 1 x h x w
        pooled = fmap.mean(dim=(2, 3))                  # B x C
        return dmap, self.digit(pooled)