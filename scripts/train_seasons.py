"""Train the GRU on the Seasons working subset.

Why this script exists separately from `train_stemporal.py`:
the 50-class INCLUDE manifest only knows about 2 of the 6 Season classes
(Fall, Summer). The Seasons zip on disk has all 6 — Fall, Summer, Spring,
Winter, Season, Ex. Monsoon — but those extra 4 classes aren't in the
50-class splits, so the standard trainer ignores them.

This trainer uses the 263-class `include` manifest (which lists all 6
classes) against a dedicated cache at `data/processed_seasons/`. It is the
smallest possible change that gets us a real 6-class, 66-clip model.

Usage:
    PYTHONPATH=src python3 scripts/train_seasons.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge import augment  # noqa: E402
from isl_bridge import config  # noqa: E402
from isl_bridge import dataset_meta  # noqa: E402
from isl_bridge.checkpoint import save_checkpoint  # noqa: E402
from isl_bridge.model import ModelConfig  # noqa: E402
from isl_bridge.stemporal import StemporalCNN  # noqa: E402
from isl_bridge.eval import write_report, macro_metrics, confusion_matrix  # noqa: E402
from isl_bridge.train import _RemappedDataset, _default_collate  # noqa: E402


# Ponytail: hardcoded Seasons roster. Six classes, all on disk under
# data/raw/include/Seasons/<NN. <word>/*.MOV. The label map comes from the
# 263-class INCLUDE manifest; the per-split membership comes from the same
# manifest's include_train/val/test.txt files.
SEASONS_CLASSES = ["fall", "summer", "spring", "winter", "season", "exmonsoon"]


def _build_seasons_inventory(proc_dir: Path) -> tuple[list[tuple[str, int]], dict[int, int], dict[int, str], dict[str, str]]:
    """Walk the Seasons cache and produce the (rel, label) pairs the trainer
    expects, plus a label remap that compresses 6 season word-strings into
    contiguous integer ids 0..5.

    Returns (kept_items, label_to_idx, idx_to_class). `kept_items` is a list
    of (relpath, orig_label_id) tuples where orig_label_id is the integer
    label from the 263-class label map.
    """
    label_map = dataset_meta.load_label_map("include", meta_dir=dataset_meta.META_DIR)
    word_to_orig = {w: label_map[w] for w in SEASONS_CLASSES if w in label_map}
    splits = dataset_meta.split_paths("include", meta_dir=dataset_meta.META_DIR)
    rel_to_split: dict[str, str] = {}
    for split_name, fp in splits.items():
        for line in fp.read_text().splitlines():
            line = line.strip()
            if line and line.startswith("Seasons/"):
                rel_to_split[line] = split_name

    # Walk the cache; each npz is at proc_dir/<split>/<rel_dir>/<stem>.npz
    kept: list[tuple[str, int]] = []
    for split in ("train", "val", "test"):
        split_dir = proc_dir / split / "Seasons"
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            # class_dir.name is "<NN>. <word>" for canonical Seasons rows OR
            # "Ex. Monsoon" / "Ex. <word>" for the extra category the
            # upstream team added. Strip the numeric prefix if present and
            # lower-case; "Ex. Monsoon" -> "exmonsoon" via direct lookup.
            raw = class_dir.name
            if ". " in raw and raw.split(". ", 1)[0].isdigit():
                word = raw.split(". ", 1)[1].strip().lower()
            else:
                # "Ex. Monsoon" -> "exmonsoon"
                word = raw.replace(".", "").replace(" ", "").strip().lower()
            if word not in word_to_orig:
                continue
            for npz in sorted(class_dir.glob("*.npz")):
                # npz.stem is the original filename (e.g. MVI_4567).
                # relpath = "Seasons/<class_dir.name>/<stem>.MOV"
                rel = f"Seasons/{class_dir.name}/{npz.stem}.MOV"
                if rel not in rel_to_split or rel_to_split[rel] != split:
                    continue
                kept.append((rel, word_to_orig[word]))

    train_orig = sorted({lid for (rel, lid) in kept if rel_to_split[rel] == "train"})
    label_to_idx = {orig: new for new, orig in enumerate(train_orig)}
    idx_to_class = {new: next(w for w, o in word_to_orig.items() if o == orig)
                    for orig, new in label_to_idx.items()}
    return kept, label_to_idx, idx_to_class, rel_to_split


class _SeasonsDataset:
    """Lightweight dataset matching the trainer's expectations: __len__,
    __getitem__ that returns the cached (1,T,3,224,224) float32 frames plus
    a remapped integer label."""

    def __init__(self, items: list[tuple[str, int]], label_to_idx: dict,
                 split: str, proc_dir: Path, rel_to_split: dict):
        # Filter rows to the requested split AND drop any class the model
        # wasn't trained on (so val/test don't error when a class is absent).
        self.items = [(rel, label_to_idx[orig]) for (rel, orig) in items
                      if rel_to_split.get(rel) == split
                      and orig in label_to_idx]
        self.split = split
        self.proc_dir = proc_dir
        # Mirror ClipDataset's npz lookup: proc_dir/<split>/<rel_dir>/<stem>.npz
        from isl_bridge.dataloaders import IMAGENET_MEAN, IMAGENET_STD
        self._mean = IMAGENET_MEAN
        self._std = IMAGENET_STD

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np
        import torch
        rel, label = self.items[idx]
        rel_dir = str(Path(rel).parent)
        stem = Path(rel).stem
        npz_path = self.proc_dir / self.split / rel_dir / f"{stem}.npz"
        z = np.load(npz_path)
        f = z["frames"]   # uint8 (T,H,W,3)
        mask = z["mask"]  # uint8 (T,)
        x = torch.from_numpy(f).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        mean = torch.tensor(self._mean).view(1, 3, 1, 1)
        std = torch.tensor(self._std).view(1, 3, 1, 1)
        x = (x - mean) / std
        return {"frames": x,
                "mask": torch.from_numpy(mask),
                "label": torch.tensor(label, dtype=torch.long),
                "relpath": rel}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/phase4_seasons"))
    p.add_argument("--proc-dir", type=Path, default=Path("data/processed_seasons"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gru-hidden", type=int, default=128)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.proc_dir.is_dir():
        print(f"ERROR: cache not found at {args.proc_dir}. "
              "Run scripts/build_seasons_cache.py first.")
        return 1

    kept, label_to_idx, idx_to_class, rel_to_split = _build_seasons_inventory(args.proc_dir)
    if not kept:
        print("ERROR: no Seasons rows found in cache.")
        return 1
    print(f"working subset: {len(kept)} clips across {len(idx_to_class)} classes: "
          f"{[idx_to_class[i] for i in sorted(idx_to_class)]}")

    train_ds = _SeasonsDataset(kept, label_to_idx, "train", args.proc_dir, rel_to_split)
    val_ds = _SeasonsDataset(kept, label_to_idx, "val", args.proc_dir, rel_to_split)
    test_ds = _SeasonsDataset(kept, label_to_idx, "test", args.proc_dir, rel_to_split)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    if len(train_ds) < 5:
        print("ERROR: too few train clips; need >= 5.")
        return 1

    cfg = ModelConfig(backbone=args.backbone, num_classes=len(idx_to_class),
                      pretrained=True, freeze_backbone=True, dropout=0.2)

    # Wrap train with augmentation. The augment wrapper only requires the
    # duck-typed surface (`__len__`, `__getitem__`, `.items`) that our
    # `_SeasonsDataset` already provides; the type mismatch with ClipDataset
    # is type-checker only.
    from typing import cast
    from isl_bridge import dataloaders as _dl
    aug_train = augment.wrap_train(cast(_dl.ClipDataset, train_ds))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(aug_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=_default_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_default_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=_default_collate)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StemporalCNN(cfg, gru_hidden=args.gru_hidden).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    from isl_bridge.train import _run_epoch
    history = []
    best_val = -1.0
    best_path = args.out / "best.pt"
    t0 = time.time()
    for ep in range(args.epochs):
        tl, ta, _, _ = _run_epoch(model, train_loader, device=device, optimizer=optimizer, train=True)
        vl, va, _, _ = _run_epoch(model, val_loader, device=device, optimizer=optimizer, train=False)
        rec = {"epoch": ep, "train_loss": round(tl, 4), "train_acc": round(ta, 4),
               "val_loss": round(vl, 4), "val_acc": round(va, 4),
               "lr": args.lr, "n_train": len(aug_train), "n_val": len(val_ds),
               "classes": cfg.num_classes}
        history.append(rec)
        print(json.dumps(rec))
        if va > best_val:
            best_val = va
            save_checkpoint(best_path, model=model, cfg=cfg,
                            idx_to_class=idx_to_class, history=history)
    elapsed = round(time.time() - t0, 2)

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "label_map.json").write_text(json.dumps({
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "label_to_idx": {str(k): v for k, v in label_to_idx.items()},
    }, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "model_config": cfg.__dict__, "lr": args.lr, "batch_size": args.batch_size,
        "epochs": args.epochs, "seed": args.seed, "device": device.type,
        "elapsed_sec": elapsed, "gru_hidden": args.gru_hidden,
        "proc_dir": str(args.proc_dir),
    }, indent=2))

    # Evaluate on test.
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].to(device)
            logits = model(batch["frames"].to(device), batch["mask"].to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            yt.extend(labels.cpu().numpy().tolist())
            yp.extend(preds.tolist())
    n = cfg.num_classes
    cm = confusion_matrix(yt, yp, n=n)
    m = macro_metrics(cm)
    m["accuracy"] = round(sum(1 for a, b in zip(yt, yp) if a == b) / max(1, len(yt)), 4)
    m["n_eval"] = len(yt)
    rep_dir = args.out / "reports"
    write_report(rep_dir, model_name="stemporal_gru_seasons", split="test",
                 cm=cm, metrics=m, idx_to_class=idx_to_class)
    eval_blob = {
        "stats": {"n_train": len(train_ds), "n_val": len(val_ds),
                  "n_test": len(test_ds), "n_classes": cfg.num_classes,
                  "class_names": [idx_to_class[i] for i in range(cfg.num_classes)]},
        "metrics": m, "confusion_matrix": cm.tolist(),
        "warning": "Computed on the Seasons working subset only (66 clips, 6 classes). "
                   "Numbers are pipeline-completeness on a small set, not a benchmark.",
    }
    (args.out / "evaluation.json").write_text(json.dumps(eval_blob, indent=2))
    print(json.dumps({"test_acc": m["accuracy"], "macro_f1": m["macro_f1"],
                      "n_test": len(yt), "classes": cfg.num_classes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
