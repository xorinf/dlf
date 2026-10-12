"""Phase 4 — Spatiotemporal head on the same frozen MobileNetV2 backbone.

Reuses everything in `model.py` (the frozen feature extractor) and only swaps
the temporal aggregator: mean-pool -> GRU -> linear. The training loop in
`train.py` is unchanged because it takes any `nn.Module` whose `forward`
takes `(frames, mask)` and returns `(B, num_classes)` logits.

# ponytail: single-layer GRU; deeper / bidirectional added only if Phase-9
#            metrics show the lift justifies the cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import FrameCNN, ModelConfig


class StemporalCNN(nn.Module):
    """MobileNetV2 features -> 1-layer GRU -> linear logits.

    Defaults: hidden=128, num_layers=1, batch_first=True, dropout=0.2.
    """

    def __init__(self, cfg: ModelConfig, *, gru_hidden: int = 128, gru_layers: int = 1,
                 gru_dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.cfg = cfg
        self.frame_cnn = FrameCNN(cfg)             # backbone + mean-pool we replace
        # Reuse the frozen backbone from the inner FrameCNN so we don't re-load it.
        feat_dim = self._backbone_feature_dim(cfg)
        out_dim = gru_hidden * (2 if bidirectional else 1)
        self.temporal = nn.GRU(
            input_size=feat_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, cfg.num_classes),
        )

    @staticmethod
    def _backbone_feature_dim(cfg: ModelConfig) -> int:
        return 1280 if cfg.backbone == "mobilenet_v2" else 512 if cfg.backbone == "resnet18" else 0

    def forward(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feats = self.frame_cnn.backbone(x).view(B, T, -1)   # (B, T, feat_dim)
        out, _ = self.temporal(feats)                       # (B, T, out_dim)
        # Last-step pooling. GRU naturally encodes ordering; if a mask is given,
        # we take the last non-padded index.
        if mask is not None:
            lengths = mask.long().sum(dim=1).clamp(min=1) - 1   # (B,)
            idx = lengths.view(B, 1, 1).expand(-1, 1, out.size(-1))
            pooled = out.gather(1, idx).squeeze(1)              # (B, out_dim)
        else:
            pooled = out[:, -1, :]
        return self.head(pooled)