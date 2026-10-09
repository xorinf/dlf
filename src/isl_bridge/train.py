"""Training loop for the Phase-3 CNN baseline.

Single-process, single-GPU-or-CPU. AMP optional. Best-by-val checkpointing.
History persisted as JSON so we can plot or audit it without rerunning.

This is intentionally one file, not a class hierarchy. Phase 4 swaps the
model (model.FrameCNN -> spatiotemporal.Stemporal) and reuses this loop
unchanged — that's the test of whether the abstraction is at the right
level of indirection.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config, dataloaders, dataset_meta, eval as ev
from .checkpoint import save_checkpoint
from .model import FrameCNN, ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_collate(batch: list[dict]) -> dict:
    """Stack a Phase-3 batch: B frames (each T,3,H,W) into (B,T,3,H,W)."""
    return {
        "frames":  torch.stack([b["frames"]  for b in batch]),
        "mask":    torch.stack([b["mask"]    for b in batch]),
        "label":   torch.stack([b["label"]   for b in batch]),
        "relpath": [b["relpath"] for b in batch],
    }


def _forward_batch(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    return model(batch["frames"].to(device), batch["mask"].to(device))


def _run_epoch(model, loader, *, device, optimizer, train: bool) -> tuple[float, float, np.ndarray, np.ndarray]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total, correct = 0.0, 0, 0
    yt, yp = [], []
    for batch in loader:
        labels = batch["label"].to(device)
        logits = _forward_batch(model, batch, device)
        loss = F.cross_entropy(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        total += labels.size(0)
        correct += int((preds == labels.detach().cpu().numpy()).sum())
        yt.extend(labels.detach().cpu().numpy().tolist())
        yp.extend(preds.tolist())
    yt = np.asarray(yt); yp = np.asarray(yp)
    return total_loss / max(1, total), correct / max(1, total), yt, yp


def train(
    cfg: ModelConfig,
    *,
    out_dir: Path,
    lr: float = 1e-3,
    batch_size: int = 4,
    epochs: int = 10,
    seed: int = 0,
    train_workers: int = 0,
    device: str | None = None,
    progress_cb: Callable[[dict], None] | None = None,
    # If supplied, every path lookup is rooted at fake_root so the trainer can
    # be exercised against a synthetic dataset (tests). Production callers
    # leave this None.
    fake_root: Path | None = None,
) -> dict:
    """Train a FrameCNN on the on-disk working subset. Returns the final report dict."""
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    meta_dir  = (fake_root / "metadata") if fake_root else dataset_meta.META_DIR
    raw_dir   = (fake_root / "raw")      if fake_root else config.RAW_DIR
    proc_dir  = (fake_root / "processed") if fake_root else config.PROC_DIR

    label_map_name = dataset_meta.load_label_map("include50", meta_dir=meta_dir)
    # Restrict class-space to classes actually present in the *training* split
    # of the working subset. This keeps `cfg.num_classes` honest (model head
    # matches the data we actually train on); full-set training in Phase 4
    # resets this to 50.
    train_ds = dataloaders.ClipDataset("train", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
    train_classes = sorted({int(it[1]) for it in train_ds.items})
    cfg.num_classes = len(train_classes)
    label_to_idx: dict[int, int] = {orig: new for new, orig in enumerate(train_classes)}
    # label_map_name is {name: int}; we need {int: name} to print/save per-class.
    int_to_name: dict[int, str] = {int(v): k for k, v in label_map_name.items()}
    pairs: list[tuple[int, str]] = [(new, str(int_to_name.get(orig, f"<id {orig}>")))
                                    for new, orig in enumerate(train_classes)]
    idx_to_class: dict[int, str] = dict(pairs)
    (out_dir / "label_map.json").write_text(json.dumps({
        "new_to_orig_idx": {str(k): v for k, v in enumerate(train_classes)},
        "orig_to_new_idx": {str(k): v for k, v in label_to_idx.items()},
        "idx_to_class":    {str(k): v for k, v in idx_to_class.items()},
    }, indent=2))

    # Re-map the train labels to the new compact space; test/val skip if class missing.
    train_ds_remapped = _RemappedDataset(train_ds, label_to_idx)
    val_ds = dataloaders.ClipDataset("val", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
    val_ds_remapped  = _RemappedDataset(val_ds,  label_to_idx)

    train_loader = DataLoader(train_ds_remapped, batch_size=batch_size, shuffle=True,
                              num_workers=train_workers, collate_fn=_default_collate)
    val_loader   = DataLoader(val_ds_remapped,   batch_size=batch_size, shuffle=False,
                              num_workers=0,        collate_fn=_default_collate)

    model = FrameCNN(cfg).to(device_t)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    history: list[dict] = []
    best_val_acc = -1.0
    best_path = out_dir / "best.pt"
    t0 = time.time()
    for ep in range(epochs):
        tl, ta, _, _ = _run_epoch(model, train_loader, device=device_t, optimizer=optimizer, train=True)
        vl, va, _, _ = _run_epoch(model, val_loader,   device=device_t, optimizer=optimizer, train=False)
        rec = {"epoch": ep, "train_loss": round(tl, 4), "train_acc": round(ta, 4),
               "val_loss": round(vl, 4),   "val_acc":   round(va, 4),
               "lr": lr, "n_train": len(train_ds_remapped), "n_val": len(val_ds_remapped),
               "classes": cfg.num_classes}
        history.append(rec)
        if va > best_val_acc:
            best_val_acc = va
            save_checkpoint(best_path, model=model, cfg=cfg,
                            idx_to_class=idx_to_class, history=history)
        if progress_cb:
            progress_cb(rec)
        else:
            print(json.dumps(rec))
    elapsed = round(time.time() - t0, 2)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model_config": asdict(cfg), "lr": lr, "batch_size": batch_size,
        "epochs": epochs, "seed": seed, "device": device, "elapsed_sec": elapsed,
    }, indent=2))
    return {
        "best_val_acc": best_val_acc,
        "epochs_ran": epochs,
        "n_train": len(train_ds_remapped),
        "n_val": len(val_ds_remapped),
        "n_classes_trained": cfg.num_classes,
        "device": device,
        "elapsed_sec": elapsed,
        "best_checkpoint": str(best_path),
    }


class _RemappedDataset:
    """Wrap a ClipDataset and remap integer labels via a dict. Drop items whose
    label is not in the map (i.e. val samples for a class the model wasn't
    trained on)."""
    def __init__(self, base: "dataloaders.ClipDataset", label_to_idx: dict[int, int]):
        self.base = base
        self.label_to_idx = label_to_idx
        # Cache the remapped (rel, new_lid) so __getitem__ stays cheap.
        self.kept: list[tuple[int, int]] = [(i, label_to_idx[base.items[i][1]])
                                            for i in range(len(base.items))
                                            if base.items[i][1] in label_to_idx]

    def __len__(self) -> int:
        return len(self.kept)

    def __getitem__(self, i: int) -> dict:
        idx, _ = self.kept[i]
        out = self.base[idx]
        out["label"] = torch.tensor(self.kept[i][1], dtype=torch.long)
        return out


def evaluate_checkpoint(checkpoint_path: Path, *, split: str = "test",
                        out_dir: Path | None = None, batch_size: int = 4,
                        device: str | None = None,
                        label_to_idx: dict[int, int] | None = None,
                        idx_to_class: dict[int, str] | None = None,
                        fake_root: Path | None = None) -> dict:
    """Run a checkpoint on a split; write reports iff out_dir is given."""
    from .checkpoint import load_checkpoint
    model, cfg, ckpt_idx_to_class, history = load_checkpoint(checkpoint_path)
    model.eval()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_t)

    meta_dir = (fake_root / "metadata")  if fake_root else dataset_meta.META_DIR
    raw_dir  = (fake_root / "raw")       if fake_root else config.RAW_DIR
    proc_dir = (fake_root / "processed") if fake_root else config.PROC_DIR

    ds_obj = dataloaders.ClipDataset(split, raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
    # If the caller didn't supply the remap dict, infer it from the train-set
    # working subset — same code path training used.
    if label_to_idx is None:
        train_ds = dataloaders.ClipDataset("train", raw_dir=raw_dir, proc_dir=proc_dir, meta_dir=meta_dir)
        train_classes = sorted({int(it[1]) for it in train_ds.items})
        label_to_idx = {orig: new for new, orig in enumerate(train_classes)}
    if idx_to_class is None:
        idx_to_class = {new: ckpt_idx_to_class.get(new, str(new)) for new in label_to_idx.values()}

    remapped = _RemappedDataset(ds_obj, label_to_idx)
    loader = DataLoader(remapped, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_default_collate)

    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            logits = _forward_batch(model, batch, device_t)
            preds = logits.argmax(dim=-1).cpu().numpy()
            yt.extend(batch["label"].numpy().tolist())
            yp.extend(preds.tolist())
    n = max(cfg.num_classes, max(yt + yp, default=0) + 1)
    cm = ev.confusion_matrix(yt, yp, n)
    metrics = ev.macro_metrics(cm)
    metrics["accuracy"] = round(ev.accuracy(yt, yp), 4)
    metrics["n_eval"] = len(yt)
    if out_dir is not None:
        ev.write_report(out_dir, model_name=f"cnn_{cfg.backbone}", split=split,
                        cm=cm, metrics=metrics, idx_to_class=idx_to_class)
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "history_loaded": history,
        "n_eval": len(yt),
        "n_classes_trained": cfg.num_classes,
    }
