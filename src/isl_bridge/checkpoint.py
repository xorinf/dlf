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
    from .model import FrameCNN   # imported here to avoid circular at module load
    model = FrameCNN(cfg)
    model.load_state_dict(blob["model_state"])
    idx_to_class = {int(k): v for k, v in blob["idx_to_class"].items()}
    return model, cfg, idx_to_class, blob.get("history", [])
