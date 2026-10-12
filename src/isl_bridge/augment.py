"""Phase 5 — Minimal spatial augmentations applied to the training set only.

Two augmentations that are cheap and well-known to help sign recognition:

* `RandomResizedCrop` — aspect-preserving crop + resize (224 -> 224).
* `RandomHorizontalFlip` — handedness flip; safe because INCLUDE clips don't
  encode absolute left/right world coords.

We deliberately skip temporal augmentation here (random clip speed, reverse).
Those can be added in Phase 9 if the eval shows a real lift. Augmentation only
fires on the *training* DataLoader — val and test see the cached deterministic
frames.

# ponytail: torchvision's built-in transforms; no custom kernel. If we need
#            OpticalFlow/Scale/aggressive colour jitter later, swap here.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from . import dataloaders


class _AugClipDataset(Dataset):
    """Wraps a `ClipDataset` and applies augmentations on the float32 tensor."""

    def __init__(self, base: dataloaders.ClipDataset, *, hflip_p: float = 0.5,
                       crop_scale: tuple[float, float] = (0.7, 1.0)):
        self.base = base
        # Delegate `items` so `_RemappedDataset` (in train.py) can read the
        # same (relpath, label_id) tuples without re-scanning the cache.
        self.items = base.items
        self.hflip_p = hflip_p
        self.crop_scale = crop_scale

    def __len__(self) -> int:
        return len(self.base)

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (T, 3, H, W) float32, ImageNet-normalised.
        T, C, H, W = frames.shape
        # Random horizontal flip (per-clip, not per-frame — sign kinematics
        # would be destroyed otherwise).
        if torch.rand(1).item() < self.hflip_p:
            frames = frames.flip(-1)
        # Random spatial crop + resize: pick a square crop of `s` fraction of
        # the smaller side, then resize back to HxW.
        s_lo, s_hi = self.crop_scale
        s = float(torch.empty(1).uniform_(s_lo, s_hi))
        side = int(round(min(H, W) * s))
        if side < H and side < W:
            top = int(torch.randint(0, H - side + 1, (1,)).item())
            left = int(torch.randint(0, W - side + 1, (1,)).item())
            cropped = frames[:, :, top:top + side, left:left + side]
            # Nearest is fine on normalised floats; INTER_AREA isn't defined in
            # pure torch. Bilinear for a smoother resize.
            frames = torch.nn.functional.interpolate(
                cropped, size=(H, W), mode="bilinear", align_corners=False)
        return frames

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        return {**item, "frames": self._augment(item["frames"])}


def wrap_train(base: dataloaders.ClipDataset, *, hflip_p: float = 0.5,
               crop_scale: tuple[float, float] = (0.7, 1.0)) -> _AugClipDataset:
    """Convenience: build the augmented wrapper from a base ClipDataset."""
    return _AugClipDataset(base, hflip_p=hflip_p, crop_scale=crop_scale)