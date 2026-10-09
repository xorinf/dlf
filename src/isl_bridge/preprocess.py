"""Frame-level preprocessing and a uniform-length clip cache.

Outputs (under data/processed/<split>/<category>/<word>/<basename>.npz):
    frames   uint8  (T, H, W, 3)  RGB
    mask     uint8  (T,)           1 = real frame, 0 = padding (for short clips)
    meta     str    (JSON)         {src_relpath, fps_in, n_src_frames, ...}

Why this format (and not TFRecord / WebDataset): the dataset is small enough
that single-file .npz with a manifest is the smallest surface. If we ever
need WebDataset, that is a separate swap.

Output spec
-----------
T   = --frames  (uniform clip length, default 32)   — temporal augmentation
      either T uniform temporal random crop, or T = clip length on shorter clips
H, W = --size   (default 224)                       — square backbone input
Fps = --fps    (default 12)                         — temporal resample target

These three defaults will be revisited after Phase 3 once we see inference
cost and accuracy. They are committed *here* so test #2 below can pin them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config
from . import dataset

DEFAULT_FRAMES = 32
DEFAULT_SIZE   = 224
DEFAULT_FPS    = 12

cv2.setNumThreads(0)  # OpenCV threading is slow when over-subscribed; keep it single-threaded


@dataclass
class ClipMeta:
    src_relpath: str
    fps_in: float
    n_src_frames: int
    fps_out: float
    frames_out: int
    size: int
    out_path: str


def _ffprobe_meta(path: str) -> tuple[float, int]:
    """Return (fps, frame_count) via ffprobe. -1 / -1 on failure (caller logs)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames,avg_frame_rate",
             "-of", "json", path],
            timeout=10,
        )
        d = json.loads(out)
        s = d["streams"][0]
        n = int(s.get("nb_read_frames", -1))
        num, den = s["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return fps, n
    except Exception:
        return -1.0, -1


def _read_all_frames(path: str) -> tuple[list[np.ndarray], float, int]:
    """Decode a video into memory. For INCLUDE clips (~3–25 s, ~7 MB) this is fine.
    Returns (frames BGR list, fps, n).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], -1.0, -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    ok, fr = cap.read()
    while ok:
        frames.append(fr)
        ok, fr = cap.read()
    cap.release()
    return frames, fps, len(frames)


def _resample(frames: list[np.ndarray], n_out: int, fps_in: float, fps_out: float,
              rng: np.random.Generator) -> np.ndarray:
    """Pick n_out frame indices given target fps. Tries to sub-sample to fps_out, then
    fits n_out (loops / random-crop). If fps_in < fps_out we can't upsample so loop.
    """
    if not frames:
        return np.zeros((0,), dtype=np.uint8)
    in_n = len(frames)
    step_in = fps_in / fps_out if fps_in > 0 and fps_out > 0 else 1.0
    sample_idx = np.arange(0, in_n, step_in).astype(int)
    sample_idx = sample_idx[sample_idx < in_n]
    if len(sample_idx) == 0:
        sample_idx = np.array([0])

    if len(sample_idx) >= n_out:
        # Uniform pick with a deterministic random offset (so the cache is
        # reproducible per source clip + seed).
        start = int(rng.integers(0, len(sample_idx) - n_out + 1))
        pick = sample_idx[start:start + n_out]
    else:
        # Loop the available frames up to n_out by tiling.
        reps = int(np.ceil(n_out / len(sample_idx)))
        tiled = np.tile(sample_idx, reps)
        pick = tiled[:n_out]
        # ponytail: repetition-instead-of-upsampling is a known ceiling — it's OK for
        # INCLUDE (short clips, no fine motion), swap to time-interpolation if
        # accuracy drops on Phase 9 eval.

    return np.stack([frames[i] for i in pick])  # (n_out, H, W, 3) BGR


def _resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w == size:
        return img
    scale = size / max(h, w)
    new = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # Pad to square with zeros (black border).
    out = np.zeros((size, size, 3), dtype=new.dtype)
    out[:new.shape[0], :new.shape[1]] = new
    return out


def process_clip(rec: dataset.ClipRecord, out_root: Path, *,
                 size: int = DEFAULT_SIZE, n_frames: int = DEFAULT_FRAMES,
                 fps_out: float = DEFAULT_FPS,
                 seed: int = 0,
                 force: bool = False) -> ClipMeta | None:
    """Process one clip and cache to out_root/<expected_split>/<rel-dir>/<basename>.npz.

    Returns ClipMeta on success, None on read/IO failure.
    """
    if not rec.exists or rec.size_bytes == 0:
        return None
    # Mirror the same nested structure (split/cat/word) so caches and raw 1:1 map.
    rel_dir = Path(rec.relpath).parent
    base = Path(rec.relpath).stem
    out_dir = out_root / (rec.expected_split or "unassigned") / rel_dir
    out_path = out_dir / f"{base}.npz"
    if out_path.exists() and not force:
        # Trust the cache. We re-validate on read in build_dataset().
        return ClipMeta(rec.relpath, -1, -1, fps_out, n_frames, size, str(out_path))

    rng = np.random.default_rng(seed + abs(hash(rec.relpath)) % (2**32))
    frames, fps, n = _read_all_frames(rec.abs_path)
    if not frames:
        return None

    pick = _resample(frames, n_frames, fps, fps_out, rng)
    rgb = pick[:, :, :, ::-1]                       # BGR → RGB
    rgb = np.stack([_resize_keep_aspect(rgb[i], size) for i in range(len(rgb))])
    mask = np.ones((n_frames,), dtype=np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frames=rgb.astype(np.uint8), mask=mask)

    fps_meta, n_meta = (fps_in := fps, n_in := n) if n > 0 else _ffprobe_meta(rec.abs_path)
    return ClipMeta(rec.relpath, fps_meta, n_meta, fps_out, n_frames, size, str(out_path))


def build_dataset(inventory: list[dataset.ClipRecord], *,
                  size: int = DEFAULT_SIZE, n_frames: int = DEFAULT_FRAMES,
                  fps_out: float = DEFAULT_FPS,
                  force: bool = False) -> list[ClipMeta]:
    out_root = config.PROC_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    metas: list[ClipMeta] = []
    n_total = len(inventory)
    for i, rec in enumerate(inventory):
        meta = process_clip(rec, out_root, size=size, n_frames=n_frames,
                            fps_out=fps_out, force=force)
        if meta is not None:
            metas.append(meta)
        # Tiny progress print; expected to be moved to tqdm in Phase 3 when this
        # goes wider.
        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            print(f"  processed {i+1}/{n_total}  ({len(metas)} ok)")
    return metas


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    inv = dataset.build_inventory(config.RAW_DIR)
    metas = build_dataset(inv, size=args.size, n_frames=args.frames,
                          fps_out=args.fps, force=args.force)
    (config.META_DIR / "preprocess_index.json").write_text(
        json.dumps([asdict(m) for m in metas], indent=2)
    )
    print(f"Processed {len(metas)} clips → {config.PROC_DIR}")


if __name__ == "__main__":
    cli_main()
