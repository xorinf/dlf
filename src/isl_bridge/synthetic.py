"""Synthetic mini-INCLUDE for Phase-3 smoke tests.

Why: the on-disk working subset is 22 videos across 2 classes — explicitly
insufficient for a model-evaluation metric. But we still want a
realistic-enough dataset to exercise the model end-to-end (forward pass, loss,
backward, optimizer step, checkpoint round-trip, eval). We synthesise TINY
random clips that look exactly like what preprocess.py writes — so dataloader,
model, and training can all be exercised without touching the real labels.

The synthetic dataset is exclusively for tests. It is never evaluated against
the real clips. It is generated fresh each run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from . import dataloaders, config


def make_synthetic_clips(out_root: Path, *, n_train_per_class: int = 2,
                         n_val_per_class: int = 1, n_test_per_class: int = 1,
                         T: int = 32, H: int = 224, W: int = 224,
                         seed: int = 0) -> list[dict]:
    """Write random uint8 clips under <out_root>/<split>/<Cat>/<word>/<stem>.npz.

    Layout (matches make_fake_splits_metadata exactly):
        Train:  indices 0..n_train_per_class-1        -> class a (synthetica)
                indices n_train_per_class..2*n_train_per_class-1 -> class b (syntheticb)
        Val:    next n_val_per_class per class
        Test:   next n_test_per_class per class

    Total clips = 2 * (n_train_per_class + n_val_per_class + n_test_per_class).
    """
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root.parent / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    word_pool = ["synthetica", "syntheticb", "syntheticc", "syntheticd"]
    n_classes = 2  # this fixture is intentionally 2-class; wider sets would need
                   # another helper.

    blocks: list[tuple[str, int, int]] = []   # (split, start_idx, count) — repeated per class
    blocks.append(("train", 0, n_train_per_class))
    blocks.append(("val",   n_train_per_class, n_val_per_class))
    blocks.append(("test",  n_train_per_class + n_val_per_class, n_test_per_class))

    written: list[dict] = []
    # Index assignment walks split → class → within-class so the manifest
    # writer and disk layout agree exactly. Per-class counts are concatenated
    # within each split.
    idx = 0
    for split, start, count in blocks:
        for cls_idx in range(n_classes):
            for k in range(count):
                word = word_pool[cls_idx]
                rel = f"SyntheticCat/{idx:03d}. {word}/synth_{idx:03d}.MOV"
                idx += 1
                frames = rng.integers(20, 200, size=(T, H, W, 3), dtype=np.uint8)
                mask = np.ones((T,), dtype=np.uint8)
                rel_dir = Path(rel).parent
                stem = Path(rel).stem
                out = out_root / split / rel_dir / f"{stem}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, frames=frames, mask=mask)
                raw = raw_root / rel
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(b"\x00" * 8)
                written.append({"split": split, "rel": rel, "class_idx": cls_idx,
                                "word": word, "npz": str(out), "raw": str(raw)})
    return written


def make_fake_splits_metadata(fake_dir: Path, *, n_train_per_class: int = 2,
                              n_val_per_class: int = 1, n_test_per_class: int = 1) -> None:
    """Write a fake split manifest + label map. Layout matches make_synthetic_clips exactly:
    1. n_train_per_class items in synthetica (indices 0..T-1)
    2. n_train_per_class items in syntheticb (indices T..2T-1)
    3. n_val_per_class items in synthetica (indices 2T..2T+V-1)
    4. n_val_per_class items in syntheticb (indices 2T+V..2T+2V-1)
    5. n_test_per_class in synthetica
    6. n_test_per_class in syntheticb
    where T = n_train_per_class, V = n_val_per_class.
    """
    fake_dir.mkdir(parents=True, exist_ok=True)
    word_a = "synthetica"
    word_b = "syntheticb"

    def block(start: int, count: int, word: str) -> list[str]:
        return [f"SyntheticCat/{i:03d}. {word}/synth_{i:03d}.MOV" for i in range(start, start + count)]

    T = n_train_per_class
    V = n_val_per_class
    Te = n_test_per_class
    train = block(0,           T,  word_a) + block(T,                T,  word_b)
    val   = block(2*T,         V,  word_a) + block(2*T + V,          V,  word_b)
    test  = block(2*T + 2*V,   Te, word_a) + block(2*T + 2*V + Te,   Te, word_b)

    (fake_dir / "include50_train.txt").write_text("\n".join(train) + "\n")
    (fake_dir / "include50_val.txt").write_text("\n".join(val) + "\n")
    (fake_dir / "include50_test.txt").write_text("\n".join(test) + "\n")
    (fake_dir / "label_map_include50.json").write_text(
        '{"synthetica": 0, "syntheticb": 1}')
    (fake_dir / "label_map_include.json").write_text(
        '{"synthetica": 0, "syntheticb": 1}')
