"""Shared video-prep helpers used by `scripts/infer_cnn.py` and `scripts/api.py`.

Keeps the decode -> resample -> resize -> normalise pipeline in one place so
the CLI, the API, and (later) the webcam loop don't drift.

# ponytail: one place to edit. If we swap backbones, this is the spot.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataloaders import IMAGENET_MEAN, IMAGENET_STD
from .preprocess import DEFAULT_FRAMES, DEFAULT_FPS, DEFAULT_SIZE, _read_all_frames, _resample


def _resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w == size:
        return img
    import cv2
    scale = size / max(h, w)
    new = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=new.dtype)
    out[:new.shape[0], :new.shape[1]] = new
    return out


def _prep_video(path, n_frames: int = DEFAULT_FRAMES,
                size: int = DEFAULT_SIZE, fps_out: float = DEFAULT_FPS,
                seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode + resample + normalise a video into a (1, T, 3, H, W) tensor + mask."""
    rng = np.random.default_rng(seed)
    frames, fps, _ = _read_all_frames(str(path))
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    pick = _resample(frames, n_frames, fps, fps_out, rng)
    rgb = pick[:, :, :, ::-1]
    rgb = np.stack([_resize_keep_aspect(rgb[i], size) for i in range(len(rgb))])
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    mask = torch.ones((n_frames,), dtype=torch.uint8)
    return x.unsqueeze(0), mask.unsqueeze(0)


def _prep_npz(path, size: int = DEFAULT_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a preprocessed .npz cache file (frames uint8 (T,H,W,3))."""
    z = np.load(path)
    f = z["frames"]
    rgb = torch.from_numpy(f).float() / 255.0
    rgb = rgb.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = ((rgb - mean) / std).unsqueeze(0)
    mask = torch.from_numpy(z["mask"]).unsqueeze(0)
    return x, mask