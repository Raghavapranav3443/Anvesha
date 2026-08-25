"""Remote-sensing adapted visual encoder (ResNet-18 backbone) shared by all
specialist heads. This is the component that is fine-tuned on remote-sensing
training data (EuroSAT quick track / BigEarthNet v2 full track) - see
scripts/train_scene_encoder.py and scripts/train_optical_sar.py."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapt a pretrained 3-channel stem to arbitrary band counts by averaging
    the pretrained RGB kernels (keeps ImageNet/RS priors)."""
    if conv.in_channels == in_channels:
        return conv
    out_ch = conv.out_channels
    new = nn.Conv2d(in_channels, out_ch, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.mean(dim=1, keepdim=True)          # O x 1 x k x k
        new.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels)
    return new


class SceneEncoder(nn.Module):
    """ResNet-18 based encoder producing a 256-d embedding for any band count."""

    FEATURE_DIM = 256

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            res = resnet18(weights=weights)
        except Exception:  # offline fallback: random init architecture only
            from torchvision.models import resnet18
            res = resnet18(weights=None)
        res.conv1 = adapt_first_conv(res.conv1, in_channels)
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1, self.layer2 = res.layer1, res.layer2
        self.layer3, self.layer4 = res.layer3, res.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, self.FEATURE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.relu(self.proj(self.pool(x).flatten(1)))

    def feature_map(self, x: torch.Tensor, stride: int = 16) -> torch.Tensor:
        """Spatial feature map BxCxh'xw' for dense heads (grounding/change)."""
        x = self.stem(x)
        x = self.layer1(x)
        if stride <= 8:
            return self.layer2(x)   # stride 8, 128 ch
        x = self.layer2(x)
        x = self.layer3(x)          # stride 16, 256 ch
        if stride >= 32:
            x = self.layer4(x)      # stride 32, 512 ch
        return x


def resize_np(arr: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize preserving float precision for any band count."""
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).numpy().astype(np.float32)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()
    return t.unsqueeze(0)


def normalise_for_encoder(arr: np.ndarray, modality: str) -> np.ndarray:
    a = arr.astype(np.float32)
    if modality == "sar":
        if float(np.median(a)) < 0:
            # already logarithmic (dB) scale - e.g. reBEN/RISAT products
            mu = a.mean(axis=(0, 1), keepdims=True)
            sd = a.std(axis=(0, 1), keepdims=True) + 1e-6
            return ((a - mu) / sd).astype(np.float32)
        a = np.log1p(np.clip(a, 0, None))
        mu = a.mean(axis=(0, 1), keepdims=True)
        sd = a.std(axis=(0, 1), keepdims=True) + 1e-6
        return ((a - mu) / sd).astype(np.float32)
    # optical: scale reflectance-ish values into [0,1]
    hi = np.percentile(a, 99.0)
    if hi > 1.5:  # raw reflectance *10000
        a = a / hi
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _rgb3_compat(img) -> np.ndarray:
    """Collapse any band configuration to an HxWx3 display array."""
    a = img.array
    c = a.shape[2]
    if c >= 4:
        return a[..., [2, 1, 0]]
    if c == 3:
        return a
    if c == 2:
        return np.stack([a[..., 0], a[..., 1], a.mean(axis=2)], axis=2)
    return np.repeat(a, 3, axis=2)
