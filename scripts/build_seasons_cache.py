"""Build the Seasons cache from on-disk raw videos.

The standard preprocess script (`isl_bridge.preprocess.build_dataset`) walks
the INCLUDE-50 manifest, which only knows about 2 of the 6 Season classes
(Fall, Summer). This script walks the 263-class `include` manifest, picks
the 6 Seasons rows that have actual video bytes, and caches them to
`data/processed_seasons/` so the seasons trainer can use them.

The Seasons working subset on disk (Seasons_1of1.zip extracted) is 66
videos across 6 classes: fall, summer, spring, winter, season, exmonsoon.
The official AI4Bharat/INCLUDE 263-class manifest lists all 6 classes with
train/val/test splits, so this is the smallest, most reproducible way to
get a 6-class working set without re-downloading data.

Usage:
    PYTHONPATH=src python3 scripts/build_seasons_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge import config  # noqa: E402
from isl_bridge import dataset  # noqa: E402
from isl_bridge import dataset_meta  # noqa: E402
from isl_bridge import preprocess  # noqa: E402

# Six Season words present in the on-disk archive. Map raw cache-folder
# name -> canonical label-map key.
SEASON_WORDS = {
    "61. Summer": "summer",
    "62. Spring": "spring",
    "63. Winter": "winter",
    "64. Fall":   "fall",
    "65. Season": "season",
    "Ex. Monsoon": "exmonsoon",
}


def main() -> int:
    out_root = Path("data/processed_seasons")
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config.RAW_DIR)
    label_map = dataset_meta.load_label_map("include", meta_dir=dataset_meta.META_DIR)
    splits = dataset_meta.split_paths("include", meta_dir=dataset_meta.META_DIR)

    inv: list[dataset.ClipRecord] = []
    for split_name, fp in splits.items():
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line or not line.startswith("Seasons/"):
                continue
            # line: "Seasons/<class_dir>/<file>.MOV"
            parts = line.split("/")
            if len(parts) != 3:
                continue
            class_dir_name = parts[1]
            if class_dir_name not in SEASON_WORDS:
                continue
            word = SEASON_WORDS[class_dir_name]
            if word not in label_map:
                continue
            abs_path = raw_dir / line
            if not abs_path.is_file() or abs_path.stat().st_size == 0:
                continue
            inv.append(dataset.ClipRecord(
                relpath=line, abs_path=str(abs_path),
                size_bytes=abs_path.stat().st_size, exists=True,
                word=word, category="Seasons", expected_split=split_name,
            ))
    if not inv:
        print("ERROR: no Season rows with on-disk bytes found. "
              "Did you run scripts/extract_include.sh?")
        return 1

    by_split: dict[str, int] = {}
    by_word: dict[str, int] = {}
    for r in inv:
        by_split[r.expected_split] = by_split.get(r.expected_split, 0) + 1
        by_word[r.word] = by_word.get(r.word, 0) + 1
    print(f"Seasons inventory: {len(inv)} clips")
    print(f"  by split: {by_split}")
    print(f"  by class: {by_word}")

    # The shared preprocess code reads its output root from config.PROC_DIR.
    # Temporarily swap to the seasons cache so build_dataset writes there.
    orig = config.PROC_DIR
    config.PROC_DIR = out_root
    try:
        # ClipRecord vs our duck-typed inv: we use the same fields preprocess
        # reads (relpath, abs_path, size_bytes, exists, expected_split).
        metas = preprocess.build_dataset(cast("list[dataset.ClipRecord]", inv), force=False)
    finally:
        config.PROC_DIR = orig
    print(f"cached {len(metas)} clips under {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
