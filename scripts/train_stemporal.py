"""Train the Phase-4 spatiotemporal model on the on-disk working subset.

Usage:
    PYTHONPATH=src python3 scripts/train_stemporal.py

This is intentionally a thin wrapper around `train.py`. The training loop
doesn't care whether the model is `FrameCNN` (mean pool) or
`StemporalCNN` (GRU) — both have the same `forward(frames, mask) -> logits`
signature. We just inject `augment.wrap_train(...)` on the training loader.

The same honest banner from `train_baseline.py` prints at the top: this
working subset (2 classes, 16 train clips) is too small for a meaningful
recognition metric, so we report the run for pipeline completeness only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge import augment, dataloaders, dataset_meta, train  # noqa: E402
from isl_bridge.checkpoint import save_checkpoint  # noqa: E402
from isl_bridge.model import ModelConfig  # noqa: E402
from isl_bridge.stemporal import StemporalCNN  # noqa: E402
from isl_bridge.eval import write_report, macro_metrics, confusion_matrix  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/phase4_real_run"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gru-hidden", type=int, default=128)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # 1. Build the labelled datasets using the shared dataloader.
    raw_dir = Path("data/raw/include")
    proc_dir = Path("data/processed")
    meta_dir = dataset_meta.META_DIR
    base_train = dataloaders.ClipDataset("train", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
    base_val = dataloaders.ClipDataset("val", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
    base_test = dataloaders.ClipDataset("test", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)

    if len(base_train) < 5:
        print("ERROR: too few train clips; need >= 5.")
        return 1

    train_classes = sorted({int(it[1]) for it in base_train.items})
    label_to_idx = {orig: new for new, orig in enumerate(train_classes)}
    int_to_name = {int(v): k for k, v in dataset_meta.load_label_map("include50", meta_dir=meta_dir).items()}
    idx_to_class = {new: str(int_to_name.get(orig, f"<id {orig}>"))
                    for new, orig in enumerate(train_classes)}
    (args.out / "label_map.json").write_text(json.dumps({
        "new_to_orig_idx": {str(k): v for k, v in enumerate(train_classes)},
        "orig_to_new_idx": {str(k): v for k, v in label_to_idx.items()},
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
    }, indent=2))

    cfg = ModelConfig(backbone=args.backbone, num_classes=len(train_classes),
                      pretrained=True, freeze_backbone=True, dropout=0.2)

    # 2. Wrap train with augmentation; remap labels.
    from isl_bridge.train import _RemappedDataset, _default_collate
    aug_train = _RemappedDataset(
        augment.wrap_train(base_train), label_to_idx)
    val_plain = _RemappedDataset(base_val, label_to_idx)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(aug_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=_default_collate)
    val_loader = DataLoader(val_plain, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_default_collate)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StemporalCNN(cfg, gru_hidden=args.gru_hidden).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    history = []
    best_val = -1.0
    best_path = args.out / "best.pt"
    t0 = time.time()
    for ep in range(args.epochs):
        tl, ta, _, _ = train._run_epoch(model, train_loader, device=device, optimizer=optimizer, train=True)
        vl, va, _, _ = train._run_epoch(model, val_loader, device=device, optimizer=optimizer, train=False)
        rec = {"epoch": ep, "train_loss": round(tl, 4), "train_acc": round(ta, 4),
               "val_loss": round(vl, 4), "val_acc": round(va, 4),
               "lr": args.lr, "n_train": len(aug_train), "n_val": len(val_plain),
               "classes": cfg.num_classes}
        history.append(rec)
        print(json.dumps(rec))
        if va > best_val:
            best_val = va
            save_checkpoint(best_path, model=model, cfg=cfg,
                            idx_to_class=idx_to_class, history=history)
    elapsed = round(time.time() - t0, 2)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "model_config": cfg.__dict__, "lr": args.lr, "batch_size": args.batch_size,
        "epochs": args.epochs, "seed": args.seed, "device": device.type,
        "elapsed_sec": elapsed, "gru_hidden": args.gru_hidden,
    }, indent=2))

    # 3. Evaluate on test.
    model.eval()
    yt, yp = [], []
    test_plain = _RemappedDataset(base_test, label_to_idx)
    test_loader = DataLoader(test_plain, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=_default_collate)
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
    write_report(rep_dir, model_name="stemporal_gru", split="test", cm=cm,
                 metrics=m, idx_to_class=idx_to_class)
    eval_blob = {
        "stats": {"n_train": len(aug_train), "n_val": len(val_plain),
                  "n_test": len(base_test), "n_classes": cfg.num_classes,
                  "class_names": [idx_to_class[i] for i in range(cfg.num_classes)]},
        "metrics": m, "confusion_matrix": cm.tolist(),
        "warning": "Computed on Phase-4 working subset only (2 classes). Pipeline completeness, not a benchmark.",
    }
    (args.out / "evaluation.json").write_text(json.dumps(eval_blob, indent=2))
    print(json.dumps({"test_acc": m["accuracy"], "macro_f1": m["macro_f1"], "n_test": len(yt)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())