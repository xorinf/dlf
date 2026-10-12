"""Tiny checkpoint round-trip.

State-dict + the ModelConfig + the integer -> name label map, optionally with
the per-epoch training history. Saved as a single torch.save blob. Nothing
else: no optimizer state, no DDP wrapping, no EMA. Add them when something
actually needs them.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .model import ModelConfig


def save_checkpoint(path: Path, *, model: torch.nn.Module, cfg: ModelConfig,
                    idx_to_class: dict[int, str], history: list[dict[str, Any]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "history": history or [],
        "format_version": 1,
    }
    torch.save(blob, path)


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, ModelConfig, dict[int, str], list[dict[str, Any]]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["config"])
    # Detect GRU checkpoints (Phase 4): keys carry the `frame_cnn.*` prefix
    # that `StemporalCNN` uses, and `head.1.weight` has the GRU's hidden
    # dim rather than the backbone's feature dim.
    state = blob["model_state"]
    has_gru_keys = any(k.startswith("frame_cnn.") for k in state.keys()) or \
                   any(k.startswith("temporal.") for k in state.keys())
    # ponytail: heuristic dispatch. If we add a third model variant, swap to
    #            a `model_name` field in the checkpoint.
    if has_gru_keys:
        from .stemporal import StemporalCNN
        # StemporalCNN stores its head after the backbone, with input = gru_hidden
        # (or 2*gru_hidden if bidirectional). We can't always recover the exact
        # GRU hyperparams from state_dict, but the head's input dim tells us.
        head_w = state.get("head.1.weight")
        if head_w is not None:
            # head.1 is the linear; input dim == gru_hidden (or 2*).
            gru_hidden = head_w.shape[1]      # may be wrong if bidirectional
            model = StemporalCNN(cfg, gru_hidden=gru_hidden)
        else:
            model = StemporalCNN(cfg)
        model.load_state_dict(state)
    else:
        from .model import FrameCNN
        model = FrameCNN(cfg)
        model.load_state_dict(state)
    idx_to_class = {int(k): v for k, v in blob["idx_to_class"].items()}
    return model, cfg, idx_to_class, blob.get("history", [])
