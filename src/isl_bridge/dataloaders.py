"""PyTorch Dataset / DataLoader for INCLUDE.

Inputs come from data/processed/<split>/<Category>/<word>/<stem>.npz. We follow
the cache layout produced by src.isl_bridge.preprocess, not raw .MOV files,
because that's what makes CPU training tractable on a laptop (Phase 3 gate
verification on this machine).

Returned per item: dict with
    frames : float32, shape (T, 3, H, W)   — normalised for the backbone
    mask   : uint8,   shape (T,)            — 1 = real, 0 = pad
    label  : int                            — INCLUDE-50 label id
    relpath: str                            — for error reporting

Splits are loaded from data/metadata/include50_{train,val,test}.txt — exactly
the upstream AI4Bharat/INCLUDE pinning. A row whose word segment does not
match a canonical label is *skipped*, mirroring the upstream reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config, dataset_meta, dataset as ds

# ImageNet mean/std — the MobileNetV2 / ResNet expectation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _load_split_manifest(
    split: str,
    *,
    raw_dir: Path | None = None,
    proc_dir: Path | None = None,
    meta_dir: Path | None = None,
) -> list[tuple[str, int]]:
    """Return [(relpath, label_id)] for the given split.

    Skips rows that:
      - don't have an on-disk cache (under `proc_dir`)
      - have a word segment that doesn't resolve to a known label (orphan rows)

    Tests can pass raw_dir / proc_dir / meta_dir to point at synthetic datasets
    without monkey-patching module globals.
    """
    raw_dir  = raw_dir  or config.RAW_DIR
    proc_dir = proc_dir or config.PROC_DIR
    meta_dir = meta_dir or dataset_meta.META_DIR
    label_map = dataset_meta.load_label_map("include50", meta_dir=meta_dir)
    inv = ds.build_inventory(raw_dir, meta_dir=meta_dir)
    inv_by_rel = {r.relpath: r for r in inv}

    fp = dataset_meta.split_paths("include50", meta_dir=meta_dir)[split]
    out: list[tuple[str, int]] = []
    for line in fp.read_text().splitlines():
        rel = line.strip()
        if not rel: continue
        r = inv_by_rel.get(rel)
        if r is None or not (r.exists and r.size_bytes > 0): continue
        lid = ds.label_for(r.word, label_map)
        if lid is None: continue
        # Confirm the cache file exists too. Match preprocess layout:
        #   <proc_dir>/<split>/<rel_dir>/<stem>.npz
        stem = Path(rel).stem
        rel_dir = Path(rel).parent
        cache = proc_dir / split / rel_dir / f"{stem}.npz"
        if not cache.is_file(): continue
        out.append((rel, lid))
    return out


class ClipDataset(Dataset):
    """One item per npz cache."""

    def __init__(self, split: str, *, raw_dir: Path | None = None,
                 proc_dir: Path | None = None, meta_dir: Path | None = None,
                 label_map: Optional[dict[str, int]] = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.split = split
        self.raw_dir  = raw_dir  or config.RAW_DIR
        self.proc_dir = proc_dir or config.PROC_DIR
        self.items = _load_split_manifest(split, raw_dir=self.raw_dir,
                                          proc_dir=self.proc_dir, meta_dir=meta_dir)
        if label_map is None:
            label_map = dataset_meta.load_label_map("include50",
                                                   meta_dir=meta_dir)
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.items)

    def _npz_path(self, rel: str) -> Path:
        # rel is "<Cat>/<idx>. <word>[/<sub>]/<file>" — strip "<file>".
        rel_dir = Path(rel).parent
        stem = Path(rel).stem
        return self.proc_dir / self.split / rel_dir / f"{stem}.npz"

    def __getitem__(self, idx: int) -> dict:
        rel, label = self.items[idx]
        cache = self._npz_path(rel)
        z = np.load(cache)
        frames = z["frames"]     # uint8 (T,H,W,3)
        mask = z["mask"]         # uint8 (T,)
        T, H, W, _ = frames.shape

        # uint8 (T,H,W,3) -> float32 (T,3,H,W), ImageNet-normalised.
        x = torch.from_numpy(frames).float() / 255.0
        x = x.permute(0, 3, 1, 2)              # (T, 3, H, W)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
        x = (x - mean) / std

        return {
            "frames":  x,                       # (T, 3, H, W)
            "mask":    torch.from_numpy(mask),  # (T,)
            "label":   torch.tensor(label, dtype=torch.long),
            "relpath": rel,
        }
