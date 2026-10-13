"""Phase-2 visualisations.

Tiny matplotlib + opencv-only figures; written into artifacts/figures/ so the
report can cite them. Nothing pretty — just the deterministic evidence the
spec asks for: a sample-frame grid and a per-class histogram for whatever is
on disk.

Run with `PYTHONPATH=src python3 scripts/make_figures.py`.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isl_bridge import config, dataset, dataset_meta  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sample_frame(video_path: str) -> "np.ndarray | None":
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0: cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 4))   # middle-ish, not first frame
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def sample_grid(inv, n_per_class: int = 3, classes: list[str] | None = None) -> None:
    label_map = dataset_meta.load_label_map("include50")
    by_word: dict[str, list] = {}
    for r in inv:
        if not (r.exists and r.size_bytes > 0): continue
        if dataset.label_for(r.word, label_map) is None: continue
        by_word.setdefault(r.word, []).append(r)

    chosen = classes or sorted(by_word)
    rows = len(chosen); cols = max(1, n_per_class)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4),
                             squeeze=False)
    for i, cls in enumerate(chosen):
        # Pick n distinct records for visual diversity.
        pool = by_word.get(cls, [])
        picks = random.sample(pool, min(cols, len(pool)))
        for j in range(cols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(picks):
                ax.set_facecolor("white"); continue
            img = _sample_frame(picks[j].abs_path)
            if img is None:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            else:
                ax.imshow(img)
            if j == 0:
                ax.set_ylabel(cls, fontsize=9)
    fig.suptitle("INCLUDE sample frames (one per class per column)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "sample_frames.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def class_histogram(inv) -> None:
    label_map = dataset_meta.load_label_map("include50")
    counts = {}
    for r in inv:
        if not (r.exists and r.size_bytes > 0): continue
        if dataset.label_for(r.word, label_map) is None: continue
        counts[r.word] = counts.get(r.word, 0) + 1

    if not counts:
        print("No present classes — skipping histogram.")
        return
    labels = sorted(counts)
    values = [counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("present videos on disk")
    ax.set_title(f"INCLUDE working subset — per-class present clip counts "
                 f"({len(labels)} classes, {sum(values)} clips)")
    fig.tight_layout()
    out = OUT_DIR / "class_histogram.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def _seasons_rows(inv):
    """Filter the inventory to Season rows; tolerate both '61. Summer' and
    'Ex. Monsoon' folder shapes."""
    out = []
    for r in inv:
        if not r.relpath.startswith("Seasons/"): continue
        if not (r.exists and r.size_bytes > 0): continue
        out.append(r)
    return out


def seasons_grid(inv, n_per_class: int = 3) -> None:
    seasons = _seasons_rows(inv)
    by_class: dict[str, list] = {}
    for r in seasons:
        parts = r.relpath.split("/")
        if len(parts) < 2: continue
        by_class.setdefault(parts[1], []).append(r)
    chosen = sorted(by_class)
    if not chosen:
        print("No Seasons classes on disk — skipping seasons grid.")
        return
    rows, cols = len(chosen), max(1, n_per_class)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4),
                             squeeze=False)
    for i, cls in enumerate(chosen):
        pool = by_class[cls]
        picks = random.sample(pool, min(cols, len(pool)))
        for j in range(cols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(picks):
                ax.set_facecolor("white"); continue
            img = _sample_frame(picks[j].abs_path)
            if img is None:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            else:
                ax.imshow(img)
            if j == 0:
                ax.set_ylabel(cls, fontsize=9)
    fig.suptitle(f"Seasons sample frames ({len(chosen)} classes, "
                 f"{sum(len(v) for v in by_class.values())} clips)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "seasons_sample_frames.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def seasons_histogram(inv) -> None:
    seasons = _seasons_rows(inv)
    by_class: dict[str, int] = {}
    for r in seasons:
        parts = r.relpath.split("/")
        if len(parts) < 2: continue
        by_class[parts[1]] = by_class.get(parts[1], 0) + 1
    if not by_class:
        print("No Seasons classes — skipping seasons histogram.")
        return
    labels = sorted(by_class)
    values = [by_class[l] for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("present videos on disk")
    ax.set_title(f"Seasons working subset — per-class counts "
                 f"({len(labels)} classes, {sum(values)} clips)")
    fig.tight_layout()
    out = OUT_DIR / "seasons_class_histogram.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def duration_histogram(inv) -> None:
    """Side-effect: a quick ffprobe-driven duration distribution for honest reporting."""
    import subprocess
    durs = []; n_failed = 0
    for r in inv:
        if not (r.exists and r.size_bytes > 0): continue
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 r.abs_path],
                timeout=10,
            )
            d = float(out.strip())
            if d > 0: durs.append(d)
            else: n_failed += 1
        except Exception:
            n_failed += 1
    if not durs:
        print("No durations measured — skipping histogram.")
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(durs, bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("video duration (s)")
    ax.set_ylabel("count")
    ax.set_title(f"Duration distribution — {len(durs)} clips, "
                 f"min={min(durs):.1f}s median={sorted(durs)[len(durs)//2]:.1f}s "
                 f"max={max(durs):.1f}s (failed={n_failed})")
    fig.tight_layout()
    out = OUT_DIR / "duration_histogram.png"
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


def main() -> None:
    inv = dataset.build_inventory(config.RAW_DIR)
    random.seed(0)
    sample_grid(inv)
    class_histogram(inv)
    duration_histogram(inv)
    # Seasons-specific figures: uses the 263-class manifest so the extra
    # classes (Spring, Winter, Season, Ex. Monsoon) show up too.
    seasons_grid(inv)
    seasons_histogram(inv)


if __name__ == "__main__":
    main()
