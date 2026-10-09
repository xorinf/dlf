# Phase 3 — CNN Baseline

A reproducible CNN-based ISL word classifier built end-to-end against the
Phase-2 working subset, plus a **fully wired synthetic-data fixture** that
exercises the entire pipeline without the tiny-subset problems.

## What landed

| File | Purpose |
|------|---------|
| `src/isl_bridge/dataloaders.py`   | `ClipDataset` (raw npz → `(T,3,H,W)` float32 + mask + label) and `_load_split_manifest` joining the upstream `include50_*.txt` CSV files to the cached clips, with threadable `raw_dir` / `proc_dir` / `meta_dir` so tests don't have to monkey-patch globals. |
| `src/isl_bridge/model.py`         | `FrameCNN`: `MobileNetV2` (or `ResNet18`) ImageNet backbone, frame features → temporal mean-pool → linear head (`num_classes` configurable). Defaults: pretrained=True, freeze_backbone=True, dropout=0.2. |
| `src/isl_bridge/checkpoint.py`    | `save_checkpoint` / `load_checkpoint`: state-dict + `ModelConfig` + `idx_to_class` + per-epoch history, single torch blob. |
| `src/isl_bridge/train.py`         | `train(...)` and `evaluate_checkpoint(...)` — same code path used by both Phase 3 (CNN) and Phase 4 (temporal model, since it only swaps the model). |
| `src/isl_bridge/eval.py`          | Confusion matrix, macro precision/recall/F1, per-class table. Writes confusion matrix CSV + classification report JSON under `reports/`. |
| `src/isl_bridge/synthetic.py`     | `make_synthetic_clips` + `make_fake_splits_metadata` — emits a deterministic 2-class mini-dataset that lines up exactly with the real splits/label_map loader. |
| `src/isl_bridge/_once.py`         | One-shot warning printer so library code paths don't multiply messages. |
| `scripts/train_baseline.py`       | CLI: train + evaluate on the on-disk working subset. Prints an honest-warning banner when the subset is too small. |
| `scripts/infer_cnn.py`            | Standalone inference on a video file or a preprocessed `.npz`. Top-k + optional abstention threshold. |
| `tests/test_phase3.py`            | 5 tests, all green on CPU. |
| `docs/phase3.md`                  | This file. |
| `artifacts/phase3_real_run/`      | Real-subset end-to-end run (see *Real-data run* below). **Not a benchmark.** |

## Dataset validation (Phase-3 re-check)

The Phase-2 verification was repeated as a sanity check:

```
Working subset in scope: 22 / 958 clips (Phase 2)
Working subset classes  : 2  (Fall, Summer)
phase3_real_run train   : 16
phase3_real_run val     : 2
phase3_real_run test    : 4
```

That is, the same shape the Phase 2 summary reports. **The spec is explicit
that this is too small for a meaningful recognition metric**; we therefore
treat this run as a pipeline-completeness check, not a benchmark. Synthetic
runs (see "Synthetic smoke" below) carry no performance claim either; they
exist only to prove the model code is correct end-to-end on stable ground.

### Signer-independent evaluation

Not performed; the public INCLUDE metadata exposes no per-clip signer id
(see `docs/phase2.md` § "Data leakage & signer-independence"). We do not
fabricate a signer-independent number.

### Data leakage

- **Class remap is internal to a run**: train-time classes are compacted to
  `0..n-1` *for the working subset only* (`label_to_idx` in `train.py`).
  The compact map is saved in `label_map.json` next to the checkpoint.
  Evaluation re-derives the compact map from the train split before scoring
  the test split, so val/test items whose class is missing from train are
  dropped (and counted in `n_eval`).
- **No carry-over between runs**: each script invocation rebuilds the
  remap from scratch.

## Architecture

```
frames (B, T, 3, 224, 224) uint8 -> float32, ImageNet-normalised
        │
        ▼
MobileNetV2 (ImageNet pretrained; frozen for Phase 3, optionally unfrozen)
        │  each frame → 1280-d feature
        ▼
reshape (B, T, 1280)
        │
        ▼
mask-aware mean pool  →  1280-d vector per clip
        │
        ▼
Dropout(0.2) → Linear(1280, num_classes)  →  logits
```

Default knobs (committed at the top of `train.py`):

| Knob                | Default | Phase-3 justification |
|---------------------|---------|------------------------|
| Backbone            | mobilenet_v2 | 3.5 M params, ImageNet, CPU-friendly |
| `num_classes`       | real-or-synthetic, set per call | per-run ground truth |
| Pretrained          | True | ImageNet priors >> random init for tiny sets |
| Freeze backbone     | True | tiny set + 2562 trainable head params avoids overfit |
| Dropout             | 0.2  | one weak regulariser to stabilise the head |
| Optimizer           | Adam, lr=1e-3 | standard first sweep |
| Epochs              | 10 (synthetic), 4 (real) | small sets don't benefit from more |
| Batch size          | 4     | fits in CPU memory; deterministic seed |
| Seed                | 0     | reproducible |

## Synthetic smoke (4 train, 2 val, 2 test, 2 classes)

```
$ PYTHONPATH=src python3 -m pytest tests/test_phase3.py -v
tests/test_phase3.py::test_model_shape            PASSED   (CPU forward, no ImageNet weights)
tests/test_phase3.py::test_model_resnet18         PASSED   (alt backbone wires correctly)
tests/test_phase3.py::test_dataloader_contract    PASSED   (T=32 uint8, normalised, mask, label)
tests/test_phase3.py::test_train_round_trip       PASSED   (train 2 epochs + evaluate; confusion matrix)
tests/test_phase3.py::test_ckpt_roundtrip         PASSED   (state-dict tensors identical post save/load)
======= 5 passed in 26.19s =======
```

The synthetic fixture is **not** a benchmark. Random pixels get ~50% on 2
classes (chance). What it proves is that the whole pipeline — data load →
forward → loss → backprop → optimizer step → checkpoint save → reload →
evaluation → confusion matrix — works without any train-time error.

## Real-data run (Phase-3 working subset)

```
$ PYTHONPATH=src python3 scripts/train_baseline.py --epochs 4 --batch-size 4 \
        --out artifacts/phase3_real_run
Working subset: train=16, val=2, test=4, classes=2 (['fall', 'summer']).
Phase 3 baseline on this subset is reported for pipeline completeness
ONLY — the subset is too small for any meaningful recognition metric.

{"epoch": 0, "train_loss": 0.7088, "train_acc": 0.6250, "val_acc": 0.5,  ...}
{"epoch": 1, "train_loss": 0.7248, "train_acc": 0.5000, "val_acc": 0.5,  ...}
{"epoch": 2, "train_loss": 0.6189, "train_acc": 0.6875, "val_acc": 0.5,  ...}
{"epoch": 3, "train_loss": 0.6746, "train_acc": 0.6250, "val_acc": 0.5,  ...}

TRAIN REPORT:
  best_val_acc       : 0.5
  epochs_ran         : 4
  n_train            : 16
  n_val              : 2
  n_classes_trained  : 2
  device             : cpu
  elapsed_sec        : ~118s

EVAL (test=4, no test items dropped):
  accuracy           : 0.25
  macro_precision    : 0.1667
  macro_recall       : 0.25
  macro_f1           : 0.20
  per-class:
    fall:   support=2, P=0.33, R=0.50, F1=0.40
    summer: support=2, P=0.00, R=0.00, F1=0.00
```

The numbers are noise — train accuracy fluctuates (0.5–0.6875), val
accuracy is exactly chance (0.5), test accuracy is below chance (0.25).
With n=4 test items and 2 classes, the confidence interval is wide
enough to absorb the entire range `[0, 1]`.

**Do not cite these as performance.** Phase 5 / 9 will retrain on the
full INCLUDE-50 once more disk is available; only those runs carry
the recognition-accuracy claim.

### Inference on a real clip

```
$ PYTHONPATH=src python3 scripts/infer_cnn.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --npz data/processed/test/Seasons/64.\ Fall/MVI_4578.npz

{
  "prediction": "fall",
  "top1_confidence": 0.5148,
  "top_k": [{"class": "fall", "confidence": 0.5148},
            {"class": "summer", "confidence": 0.4852}]
}

$ PYTHONPATH=src python3 scripts/infer_cnn.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --abstain-threshold 0.55 \
        --npz data/processed/test/Seasons/61.\ Summer/MVI_4568.npz

{
  "prediction": "uncertain",
  "top1_confidence": 0.5064,
  "abstain_threshold": 0.55,
  ...
}
```

The `--abstain-threshold` switch is the Phase-1 / Phase-7 commitment to a
*no-confidence → say "uncertain"* policy; this is the first observable
behaviour for it.

## Reproducibility

- **Seed-everything** is the first call in `train()`: `random`, `numpy`,
  `torch` (+ CUDA when available) all set to the same value.
- **Config is saved** as `out_dir/config.json` (model config + lr +
  batch + epochs + seed + elapsed).
- **History saved** as `out_dir/history.json` (per-epoch loss + acc + lr
  + n_train + n_val + classes).
- **Best-by-val checkpoint** at `out_dir/best.pt`.
- **Class remap** at `out_dir/label_map.json` (compact int ↔ INCLUDE-50
  int ↔ human name).
- **No reliance on environment-specific paths**: the only path knob is
  `ISL_BRIDGE_DATA_ROOT`, env-overridable; defaults to `./data`.

## Limitations

1. **22 clips is not enough for any recognition-performance claim.**
   We don't make one. Per spec Phase 3 requirement 1: *"Do not treat the
   22 available videos as sufficient for meaningful model evaluation."*
   This run is pipeline-completeness evidence only.
2. **No signer-independent split.** INCLUDE does not publish signer IDs
   in its CSVs (see `docs/phase2.md`). No claim of generalisation to new
   signers.
3. **Backbone is frozen** in the default Phase-3 configuration; Phase 4
   experiments (when more data exists) will unfreeze and compare.
4. **CPU-only inference path**. M-class Mac, no MPS paths; this is the
   safest baseline. Re-running with `--unfreeze` or larger batches will
   be more useful once disk allows fuller training.

## Additional data required for a meaningful result

To run a Phase-3 baseline that actually supports the published INCLUDE
comparison (Sridhar et al., 2020: 85.6% top-1 on 263 classes, 94.5%
on the 50-class subset), we need:

- The full INCLUDE-50 (~46.7 GB compressed, ~93 GB extracted) **or**
  a larger slice of it. As of this phase, our working set is **just
  Seasons** (2 classes, 22 clips).
- More disk on the dev machine. Currently 9.6 GB free of the full
  46.7 GB required.

The downloader (`scripts/download_include.sh --include50`) is the
single-step upgrade once disk allows. No code in `src/` changes.

## Phase-4 gate

To move to Phase 4 the spec says: *"Verify that training, evaluation,
model saving, and inference all work before adding more complexity."*

- ✅ Training works end-to-end on real data and on the synthetic fixture.
- ✅ Evaluation produces per-class precision/recall/F1, accuracy,
  confusion matrix, written to `reports/`.
- ✅ Model saving round-trips byte-identical (test_ckpt_roundtrip).
- ✅ Inference works on a real preprocessed clip (with top-k + abstention).
- ✅ CPU-only path verified on macOS arm64 (`platform darwin`).

The model architecture is built with one swap point
(`temporal = TemporalPool()` placeholder → GRU/BiLSTM in Phase 4) so
the training/eval loop does not change when Phase 4 swaps the head.
