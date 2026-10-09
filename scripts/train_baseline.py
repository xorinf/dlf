"""Train the Phase-3 CNN baseline against the on-disk working subset.

Honest disclosure printed at the top of every run:

    Working subset is N=22 clips across K=2 classes — too small for any
    meaningful recognition metric. The resulting numbers are reported for
    pipeline completeness only and should not be cited as performance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge import dataloaders, dataset_meta, config  # noqa: E402
from isl_bridge.model import ModelConfig  # noqa: E402
from isl_bridge.train import train, evaluate_checkpoint  # noqa: E402


def _working_subset_stats() -> dict:
    n_train = sum(1 for _ in dataloaders.ClipDataset("train").items)
    n_val   = sum(1 for _ in dataloaders.ClipDataset("val").items)
    n_test  = sum(1 for _ in dataloaders.ClipDataset("test").items)
    classes = sorted({int(it[1]) for it in dataloaders.ClipDataset("train").items})
    label_map_name = dataset_meta.load_label_map("include50")  # str -> int
    int_to_name = {int(v): k for k, v in label_map_name.items()}
    return {"n_train": n_train, "n_val": n_val, "n_test": n_test,
            "n_classes": len(classes),
            "class_names": sorted(int_to_name[c] for c in classes if c in int_to_name)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/baseline_run"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unfreeze", action="store_true",
                   help="Phase-4 style: don't freeze the backbone. Adds cost; "
                        "kept off by default in Phase 3.")
    args = p.parse_args()

    stats = _working_subset_stats()
    banner = (
        f"Working subset: train={stats['n_train']}, val={stats['n_val']}, "
        f"test={stats['n_test']}, classes={stats['n_classes']} ({stats['class_names']}).\n"
        "Phase 3 baseline on this subset is reported for pipeline completeness\n"
        "ONLY — the subset is too small for any meaningful recognition metric."
    )
    print(banner)
    if stats["n_train"] < 5:
        print("ERROR: too few train clips to train a meaningful baseline. "
              "Use the synthetic fixture for end-to-end pipeline tests instead.")
        sys.exit(2)

    cfg = ModelConfig(num_classes=stats["n_classes"], pretrained=True,
                      freeze_backbone=not args.unfreeze, backbone=args.backbone)
    if args.out.exists():
        shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)

    report = train(
        cfg, out_dir=args.out,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
        device=None, train_workers=0,
    )
    (args.out / "subset_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nTRAIN REPORT:", json.dumps(report, indent=2, default=str))

    print("\nEvaluating on the on-disk test split ...")
    ev = evaluate_checkpoint(
        args.out / "best.pt",
        split="test",
        out_dir=args.out / "reports",
        batch_size=args.batch_size, device=None,
    )
    (args.out / "evaluation.json").write_text(json.dumps({
        "warning": "Computed on Phase-3 working subset only "
                   f"({stats['n_test']} test clips, {stats['n_classes']} classes). "
                   "Numbers reported for pipeline completeness.",
        "stats": stats,
        "report": ev["report_path"] if isinstance(ev.get("report_path"), str) else None,
        "metrics": ev["metrics"],
        "n_eval": ev["n_eval"],
        "confusion_matrix": ev["confusion_matrix"],
    }, indent=2))
    print("EVAL:", json.dumps({"metrics": ev["metrics"], "n_eval": ev["n_eval"]}, indent=2, default=str))
    print("\nOutputs under:", args.out)


if __name__ == "__main__":
    main()
