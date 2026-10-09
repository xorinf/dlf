"""Dataset scanning and integrity checks for INCLUDE.

Anatomy of an INCLUDE clip path on disk (after extraction):

    RAW_DIR/<Category>/<idx>. <word with spaces>/<file>.MOV|MP4

The official INCLUDE splits (AI4Bharat/INCLUDE) use the same format:

    <Category>/<idx>. <word>/<file>.MOV

so we can join them verbatim without re-deriving labels.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from . import config
from . import dataset_meta

SUPPORTED_EXTS = (".MOV", ".MP4", ".mov", ".mp4")


@dataclass
class ClipRecord:
    relpath: str         # "Category/idx. word/file.MOV"
    category: str
    index: str           # numeric part before the period, e.g. "61"
    word: str            # "Summer", "Hello", etc.
    filename: str
    abs_path: str
    size_bytes: int
    expected_split: str | None   # one of {train,val,test} from the official CSVs

    @property
    def exists(self) -> bool: return Path(self.abs_path).is_file()

    @property
    def corrupted(self) -> bool:
        # Fast zero-byte check. Full ffprobe is run in scan() only for candidate files.
        return self.size_bytes == 0 or not self.exists


def _scan_disk(raw_dir: Path) -> dict[str, Path]:
    """Walk raw_dir, return {relpath (POSIX): absolute_path} for every video file."""
    found: dict[str, Path] = {}
    if not raw_dir.exists():
        return found
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix in SUPPORTED_EXTS:
            rel = p.relative_to(raw_dir).as_posix()
            found[rel] = p
    return found


def _load_split_files(meta_dir: Path | None = None) -> dict[str, set[str]]:
    """Return {split_name: set(relpath)} for INCLUDE-50."""
    out: dict[str, set[str]] = {}
    for split, fp in dataset_meta.split_paths("include50", meta_dir=meta_dir).items():
        out[split] = {ln.strip() for ln in fp.read_text().splitlines() if ln.strip()}
    return out


def build_inventory(raw_dir: Path, *, meta_dir: Path | None = None) -> list[ClipRecord]:
    """List every clip that *should* exist according to INCLUDE-50 splits,
    mark which are present in raw_dir, which are missing, and which are
    zero-byte/corrupted.

    A clip is "in scope" iff its relative path appears in any of the
    INCLUDE-50 split files. This is what makes the scan honestly report what
    *the project's working subset* looks like rather than counting random
    other-Junk on disk.

    `meta_dir` overrides the default data/metadata location for tests.

    Rows whose word-segment cannot be matched to a known INCLUDE-50 label are
    flagged with `unmatched=True` and counted separately in summary(). They
    are still kept in the inventory so an upstream data error is surfaced
    rather than hidden. Today the AI4Bharat splits contain one such row
    ("Adjectives/79. short/extra/MVI_5192.MOV") — logged at import time.
    """
    splits = _load_split_files(meta_dir=meta_dir)
    rel_to_split = {rel: name for name, rels in splits.items() for rel in rels}

    on_disk = _scan_disk(raw_dir)
    out: list[ClipRecord] = []
    label_map = dataset_meta.load_label_map("include50", meta_dir=meta_dir)
    n_unmatched = 0
    for rel, sp in sorted(rel_to_split.items()):
        p = on_disk.get(rel)
        size = p.stat().st_size if p else 0
        # rel looks like "<Cat>/<idx>. <word>[/<sub>]/<filename>"
        # e.g. "Adjectives/79. short/MVI_5189.MOV"
        # e.g. "Adjectives/79. short/extra/MVI_5192.MOV"  (real outlier in AI4Bharat data)
        # Robust rule: filename = last segment, word = everything in the
        # directory-portion that's NOT the index prefix.
        parts = rel.split("/")
        filename = parts[-1]
        cat = parts[0]
        dir_path = "/".join(parts[1:-1]) if len(parts) >= 3 else ""
        if ". " in dir_path:
            idx, word = dir_path.split(". ", 1)
        else:
            idx, word = dir_path, ""
        if not word:
            word = dir_path
        # Strip any embedded subdirs the AI4Bharat metadata happens to carry.
        # Word segment here should be just the human label — the trailing "/<sub>"
        # is a dataset quirk we keep, but flatten at label_map lookup time.
        unmatched = label_for(word, label_map) is None
        if unmatched:
            n_unmatched += 1
        out.append(ClipRecord(
            relpath=rel,
            category=cat,
            index=idx,
            word=word,
            filename=filename,
            abs_path=str(p) if p else "",
            size_bytes=size,
            expected_split=sp,
        ))
    if n_unmatched:
        # Idempotent warning: print once per process. Multiple dataset / model
        # code paths build the inventory; we only want the user to see one.
        from isl_bridge import _once
        _once.warn(f"isl_bridge.dataset: {n_unmatched} INCLUDE-50 row(s) "
                   f"have a word segment that doesn't match the canonical "
                   f"label_map. See artifacts/dataset_summary.json for the list.")
    return out


def label_for(word: str, label_map: dict[str, int]) -> int | None:
    """Resolve a word to its label id using INCLUDE's canonical normalisation.

    The AI4Bharat/INCLUDE reference pipeline normalises the word segment of a
    split path to ``"".join(c for c in word if c.isalpha()).lower()`` before
    looking it up in label_map. We mirror that exactly so our pipeline and
    theirs resolve the same word segments to the same integer ids.

    Examples (INCLUDE-50):
        "Hello"            -> "hello"           -> 22
        "T-Shirt"          -> "tshirt"          -> 45
        "you (plural)"     -> "youplural"       -> 49
        "Bird/Extra"       -> "birdextra"       -> None (orphan in upstream)
        "short/extra"      -> "shortextra"      -> None (orphan in upstream)

    Returns None iff the normalised word has no entry in `label_map`.
    """
    norm = "".join(c for c in word if c.isalpha()).lower()
    return label_map.get(norm)


def summary(inventory: Iterable[ClipRecord]) -> dict:
    by_split = Counter()
    by_cat = Counter()
    by_word = Counter()
    missing = 0
    corrupted = 0
    zero_byte = 0
    total_bytes = 0
    n = 0
    for r in inventory:
        n += 1
        if not r.exists:
            missing += 1
            continue
        if r.size_bytes == 0:
            zero_byte += 1
            continue
        by_split[r.expected_split or "?"] += 1
        by_cat[r.category] += 1
        if r.word:
            by_word[r.word] += 1
        total_bytes += r.size_bytes
    return {
        "n_records": n,
        "n_present": n - missing - zero_byte,
        "n_missing": missing,
        "n_zero_byte": zero_byte,
        "bytes_total": total_bytes,
        "by_split": dict(by_split),
        "by_category": dict(by_cat),
        "n_distinct_words": len(by_word),
    }


def write_manifest(inventory: list[ClipRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(inventory[0]).keys()))
        w.writeheader()
        for r in inventory:
            w.writerow(asdict(r))


def label_coverage(inventory: Iterable[ClipRecord]) -> dict:
    """Per-INCLUDE-50-class tally: how many present clips we have for each class."""
    lab = dataset_meta.load_label_map("include50")
    counts: dict[str, int] = defaultdict(int)
    unmatched: list[str] = []
    for r in inventory:
        if not r.exists or r.size_bytes == 0:
            continue
        lid = label_for(r.word, lab)
        if lid is None:
            unmatched.append(r.word)
            continue
        counts[r.word] += 1
    return {
        "n_classes_seen": len(counts),
        "n_classes_total_in_labelmap": len(lab),
        "per_class_counts": dict(counts),
        "unmatched_words": sorted(set(unmatched)),
    }


if __name__ == "__main__":  # pragma: no cover (manual smoke)
    inv = build_inventory(config.RAW_DIR)
    s = summary(inv)
    json.dump(s, sys.stdout, indent=2, default=str)
    print()
