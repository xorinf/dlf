"""Evaluation: confusion matrix + macro P/R/F1 + per-class table.

Tiny, dependency-free (just numpy + sklearn-free P/R/F1 via counts). We
deliberately don't pull in torchmetrics/sklearn — both are unneeded for
these four numbers and would add a download and a venv for Phase 3.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], n: int) -> np.ndarray:
    """`n` is the number of classes. Out-of-range predictions are clipped to 0."""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p) % n] += 1
    return cm


def macro_metrics(cm: np.ndarray) -> dict:
    """Macro precision/recall/F1 across classes. Skips classes with zero support."""
    p_list, r_list, f1_list = [], [], []
    per_class = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        per_class.append({
            "class": k,
            "support": int(cm[k, :].sum()),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        })
        if cm[k, :].sum() > 0:        # macro averages skip absent classes
            p_list.append(prec); r_list.append(rec); f1_list.append(f1)
    return {
        "macro_precision": round(float(np.mean(p_list)), 4) if p_list else 0.0,
        "macro_recall":    round(float(np.mean(r_list)), 4) if r_list else 0.0,
        "macro_f1":        round(float(np.mean(f1_list)), 4) if f1_list else 0.0,
        "per_class": per_class,
    }


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    t = list(y_true); p = list(y_pred)
    return _safe_div(sum(1 for a, b in zip(t, p) if a == b), len(t))


def write_report(out_dir: Path, *, model_name: str, split: str, cm: np.ndarray,
                 metrics: dict, idx_to_class: dict[int, str]) -> None:
    """Write: confusion_matrix.csv, classification_report.json, summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = cm.shape[0]
    cm_csv = out_dir / f"confusion_matrix_{model_name}_{split}.csv"
    with cm_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [idx_to_class.get(i, str(i)) for i in range(n)])
        for k in range(n):
            w.writerow([idx_to_class.get(k, str(k))] + list(map(int, cm[k])))
    (out_dir / f"classification_report_{model_name}_{split}.json").write_text(
        json.dumps({"model": model_name, "split": split, "metrics": metrics}, indent=2)
    )
    (out_dir / f"summary_{model_name}_{split}.json").write_text(json.dumps({
        "model": model_name, "split": split,
        "confusion_matrix_path": cm_csv.name,
        "report_path": f"classification_report_{model_name}_{split}.json",
        **metrics,
    }, indent=2))
