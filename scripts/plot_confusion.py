"""Phase 9 — Generate a confusion-matrix PNG from a saved evaluation.json.

Reads `artifacts/<run>/evaluation.json`, draws the confusion matrix with
matplotlib, and writes it next to the source. No axes-styling debates —
single function, default matplotlib colours, labelled rows/cols.

# ponytail: matplotlib only. No seaborn. The numbers come from the same
#            evaluation.json we already produced, so this is a one-shot plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # no display required
import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Defaults to <eval-json parent>/confusion_matrix.png")
    p.add_argument("--title", default="Confusion Matrix (test)")
    args = p.parse_args()

    blob = json.loads(args.eval_json.read_text())
    cm = np.array(blob["confusion_matrix"], dtype=np.int64)
    names = blob.get("stats", {}).get("class_names") or \
            [f"class_{i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(4 + cm.shape[0] * 0.6, 4 + cm.shape[0] * 0.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=colour)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(args.title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = args.out or (args.eval_json.parent / "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(json.dumps({"out": str(out), "shape": list(cm.shape)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())