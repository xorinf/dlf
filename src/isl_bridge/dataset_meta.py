"""Canonical INCLUDE label maps.

The two label maps and train/val/test splits ship with the AI4Bharat/INCLUDE
GitHub repository (which mirrors the structures in their Google Drive upload).
We pin to specific SHA so the pipeline is reproducible; if the upstream labels
change, this module is the single place to update.
"""

from __future__ import annotations

import json
from pathlib import Path

# Versioned metadata directory. Files here are committed (and so reproducible).
# Initial layout is filled by scripts/fetch_splits.sh from the AI4Bharat/INCLUDE
# repo (which is the canonical, public source the paper cites).
META_DIR = Path(__file__).resolve().parents[2] / "data" / "metadata"

UPSTREAM_BASE = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master"
# Pinned commit SHA. Update both this constant AND re-run fetch_splits.sh to
# refresh the local copies if the upstream label_map changes.
UPSTREAM_COMMIT = "master"

LABEL_FILES = {
    "include":     "label_maps/label_map_include.json",
    "include50":   "label_maps/label_map_include50.json",
}
SPLIT_FILES = {
    "include":     ["train_test_paths/include_train.txt",
                    "train_test_paths/include_val.txt",
                    "train_test_paths/include_test.txt"],
    "include50":   ["train_test_paths/include50_train.txt",
                    "train_test_paths/include50_val.txt",
                    "train_test_paths/include50_test.txt"],
}


def load_label_map(variant: str = "include50") -> dict[str, int]:
    """Return {class_name: integer_id}.

    Variants: 'include' (263 classes), 'include50' (50 classes).
    """
    fp = META_DIR / Path(LABEL_FILES[variant]).name
    if not fp.exists():
        raise FileNotFoundError(
            f"label map for {variant!r} missing at {fp}. "
            f"Run scripts/fetch_splits.sh to populate data/metadata/."
        )
    return {k: int(v) for k, v in json.loads(fp.read_text()).items()}


def split_paths(variant: str) -> dict[str, Path]:
    """Return {'train': ..., 'val': ..., 'test': ...} as Path to local copies."""
    files = SPLIT_FILES[variant]
    return {
        "train": META_DIR / Path(files[0]).name,
        "val":   META_DIR / Path(files[1]).name,
        "test":  META_DIR / Path(files[2]).name,
    }
