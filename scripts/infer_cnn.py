"""Stand-alone inference for a single clip — Phase 3 deliverable.

Loads a saved checkpoint (model + label map) and runs it on either:
  - a video file path (.MOV / .MP4) — decoded + resized on the fly
  - a preprocessed .npz cache (data/processed/<split>/.../*.npz)
  - a directory of clips to score at once (--batch mode)

Usage:
    PYTHONPATH=src python3 scripts/infer_cnn.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --video path/to/clip.MOV

Returns the top-k (word, confidence) predictions. Threshold-based abstention
is exposed as `--abstain-threshold`; predictions below the threshold are
labelled "uncertain" rather than emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.checkpoint import load_checkpoint  # noqa: E402
from isl_bridge.dataloaders import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD,
)
from isl_bridge.preprocess import (  # noqa: E402
    DEFAULT_FRAMES, DEFAULT_FPS, DEFAULT_SIZE, _read_all_frames, _resample,
)
from isl_bridge.model import FrameCNN  # noqa: E402


def _prep_video(path: Path, *, size: int = DEFAULT_SIZE,
                n_frames: int = DEFAULT_FRAMES, fps_out: float = DEFAULT_FPS,
                rng: np.random.Generator | None = None,
                seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode + resample + normalise a video into a (1, T, 3, H, W) tensor."""
    rng = rng or np.random.default_rng(seed)
    frames, fps, n = _read_all_frames(str(path))
    if not frames: raise RuntimeError(f"could not decode {path}")
    pick = _resample(frames, n_frames, fps, fps_out, rng)
    rgb = pick[:, :, :, ::-1]
    rgb = np.stack([_resize_keep_aspect(rgb[i], size) for i in range(len(rgb))])
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
    x = (x - mean) / std
    mask = torch.ones((n_frames,), dtype=torch.uint8)
    return x.unsqueeze(0), mask.unsqueeze(0)


def _resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w == size:
        return img
    scale = size / max(h, w)
    new = __import__("cv2").resize(img, (int(round(w * scale)), int(round(h * scale))),
                                   interpolation=__import__("cv2").INTER_AREA if scale < 1 else __import__("cv2").INTER_LINEAR)
    out = np.zeros((size, size, 3), dtype=new.dtype)
    out[:new.shape[0], :new.shape[1]] = new
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None,
                   help="A single .MOV or .MP4 to score.")
    p.add_argument("--npz", type=Path, default=None,
                   help="A preprocessed .npz clip to score.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="If > 0, label low-confidence predictions as 'uncertain'.")
    args = p.parse_args()

    if not args.video and not args.npz:
        p.error("must supply --video or --npz")

    model, cfg, idx_to_class, _ = load_checkpoint(args.checkpoint)
    model.eval()
    with torch.no_grad():
        if args.video is not None:
            x, mask = _prep_video(args.video)
        else:
            z = np.load(args.npz)
            f = z["frames"]                         # uint8 (T, H, W, 3)
            rgb = torch.from_numpy(f).float() / 255.0
            rgb = rgb.permute(0, 3, 1, 2)
            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
            x = ((rgb - mean) / std).unsqueeze(0)
            mask = torch.from_numpy(z["mask"]).unsqueeze(0)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    order = np.argsort(probs)[::-1][:args.top_k]
    out = []
    for i in order:
        out.append({"class": idx_to_class.get(int(i), str(int(i))),
                   "confidence": round(float(probs[i]), 4)})

    top1_conf = out[0]["confidence"] if out else 0.0
    uncertain = (args.abstain_threshold > 0 and top1_conf < args.abstain_threshold)
    if uncertain:
        result = {"prediction": "uncertain", "top1_confidence": top1_conf,
                  "abstain_threshold": args.abstain_threshold, "top_k": out}
    else:
        result = {"prediction": out[0]["class"] if out else "n/a",
                  "top1_confidence": top1_conf,
                  "top_k": out}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
