"""Backbone -> logits model for the Phase 3 CNN baseline.

Architecture (intentionally small):
    MobileNetV2 (ImageNet pretrained, frozen-ish by default)
        -> frame-level features (B*T, 1280, 1, 1) avgpool'd to 1280
    temporal aggregation: mean-pool over T (configurable later)
        -> 1280-d vector per clip
    Linear classifier: 1280 -> num_classes

The temporal head is deliberately *mean* in Phase 3. Phase 4 will replace
the head with GRU/BiLSTM, optionally fuse landmarks, etc. So the structure
here — `frame features -> temporal pool -> linear` — keeps one swap point.

`num_classes` is configurable because Phase 3 will eventually train on all
50 (or 263) INCLUDE classes; the Working-Set 22-clip run is only 2 classes.
The model accepts whatever you give it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as M


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v2"     # "mobilenet_v2" | "resnet18" — extend in Phase 4 if needed
    num_classes: int = 50              # INCLUDE-50 default; we override per training run
    pretrained: bool = True            # download ImageNet weights the first time
    freeze_backbone: bool = True       # Phase-3 default; unfreeze in Phase 4 if it helps
    dropout: float = 0.2


class FrameCNN(nn.Module):
    """Per-video classifier: frame features -> temporal mean -> linear logits."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        feat_dim = self._build_backbone(cfg)
        self.temporal = TemporalPool()
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, cfg.num_classes),
        )

    def _build_backbone(self, cfg: ModelConfig) -> int:
        """Returns the per-frame feature dim; strips the classifier."""
        if cfg.backbone == "mobilenet_v2":
            w = M.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            net = M.mobilenet_v2(weights=w)
            net.classifier = nn.Identity()  # -> 1280-d
            if cfg.freeze_backbone:
                for p in net.parameters():
                    p.requires_grad = False
            self.backbone = net
            return 1280
        if cfg.backbone == "resnet18":
            w = M.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            net = M.resnet18(weights=w)
            net.fc = nn.Identity()  # -> 512-d
            if cfg.freeze_backbone:
                for p in net.parameters():
                    p.requires_grad = False
            self.backbone = net
            return 512
        raise ValueError(f"unknown backbone: {cfg.backbone}")

    def forward(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """frames: (B, T, 3, H, W).  mask: (B, T) in {0,1} (optional, unused for mean)."""
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feats = self.backbone(x)              # (B*T, feat_dim)
        feats = feats.view(B, T, -1)
        if mask is not None:
            # Mask-aware mean: ignore padding frames.
            m = mask.float().unsqueeze(-1)   # (B, T, 1)
            denom = m.sum(dim=1).clamp(min=1.0)
            agg = (feats * m).sum(dim=1) / denom
        else:
            agg = feats.mean(dim=1)
        return self.head(agg)


class TemporalPool(nn.Module):
    """Phase-3 default = mean pool. Class kept so Phase-4 swaps are localised."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x.mean(dim=1)
