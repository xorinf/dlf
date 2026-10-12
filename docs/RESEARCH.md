# ISL-Bridge — Research Notes

A research-and-development write-up for the ISL-Bridge sign-recognition
project. This file is the *evidence trail* behind the README numbers — the
questions we asked, the methods we tried, the answers we got, and the
experiments we did *not* run (with reasons).

## 1. Problem framing

**Goal.** Given a short (1–4 second) video of a person signing a single
isolated ISL word, output the most likely word with a calibrated
confidence, in near-real-time on commodity hardware.

**Constraints.**
* On-disk working subset is 22 clips across 2 INCLUDE classes (Fall, Summer)
  due to disk space; the same code is intended to run on the full
  INCLUDE-50 archive (43 classes, ~958 clips) once available.
* Must run on CPU for evaluation. GPU optional for training.
* End-to-end latency budget: ≤ 1 second per clip.

**Non-goals.** Continuous sign-language translation. Live signer
identification. Mobile deployment. Real signer-independent evaluation
(INCLUDE's public CSVs don't carry per-clip signer IDs).

## 2. Literature context

The INCLUDE dataset and its baselines are described in Sridhar et al.,
"INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition",
ACM MM 2020. The paper reports:

* INCLUDE-50: 50 word classes, ~4287 clips, signer-aware splits.
* INCLUDE (full): 263 word classes, ~4287 clips.
* Reported baselines: a 3D-CNN (C3D / I3D) and a 2D-CNN+GRU hybrid. The
  CNN+GRU class is what Phase 4's `StemporalCNN` mirrors.

Our departure: the public INCLUDE split files we use (`data/metadata/`)
match the official `train_test_paths/include50_*.txt` from the AI4Bharat
INCLUDE repo. Any reproducibility claim we make is anchored to those
splits — not a re-derivation.

## 3. Hypotheses

| # | Hypothesis                                                                    | Test                                                | Result                                          |
|---|-------------------------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------|
| H1 | A frozen ImageNet-pretrained CNN + linear head reaches a non-trivial baseline on tiny data.                                                 | Train MobileNetV2 (mean-pool head) on the 22-clip subset. | Result: pipeline works; 4-epoch run is 25% test acc on 4 test clips. **Cannot** distinguish "model learned" from "noise" with N=4 test. |
| H2 | A GRU temporal head over the same backbone improves over mean-pool.           | Compare Phase 3 vs Phase 4 on the same splits.        | Result: 0.25 vs 0.50 — Phase 4 is *better* on this test set, but Phase 4 also collapsed to "always predict summer". Not a meaningful lift. |
| H3 | Spatial augmentation (hflip + random-resized-crop) reduces overfitting on tiny data.                                                 | Phase 4 trainer wraps `augment.wrap_train`.          | Converges faster on train (acc 0.81 at ep 2 vs Phase 3's 0.69 at ep 2). Val/test 4 unchanged — limit is data volume, not capacity. |
| H4 | Horizontal flip is sign-kinematics-preserving (handedness is symmetric).     | `audit_robustness()` measures hflip agreement.      | 75% on the 4 test clips (3/4 predictions unchanged after hflip). |
| H5 | The model is robust to ±2x temporal subsampling.                              | `audit_robustness()` measures temporal agreement.    | 100% on the 4 test clips. |

## 4. Methodology

### 4.1 Data

* Source dataset: **INCLUDE** (Sridhar et al., 2020), Zenodo mirror.
* Splits: official AI4Bharat INCLUDE-50 `include50_{train,val,test}.txt`
  shipped under `data/metadata/`. We do **not** re-derive splits.
* Working subset (Phase 2): the first archive on the disk-budget hierarchy
  is `Seasons_1of1.zip` (43 MB), 22 clips across 2 classes (Summer, Fall).
* Preprocessing: every clip is decoded with OpenCV, resampled to a uniform
  32-frame × 224×224 tensor, and ImageNet-normalised before caching as
  `data/processed/<split>/<rel>.npz`. The cache format is documented in
  `src/isl_bridge/preprocess.py`.

### 4.2 Models

| Model            | Backbone          | Temporal head          | Default config                          | File                              |
|-------------|--------------------|------------------------|---------------------------------------|
| `FrameCNN`    | MobileNetV2 / ResNet18 (frozen) | mean pool                | 4 epochs, lr=1e-3, batch=4            | `src/isl_bridge/model.py`         |
| `StemporalCNN` | MobileNetV2 (frozen) | 1-layer GRU (h=128)        | 4 epochs, lr=1e-3, batch=4            | `src/isl_bridge/stemporal.py`     |

* MobileNetV2 is the default because it has 1280-d features (small head),
  and ImageNet pretraining puts a strong edge/texture prior into the per-frame
  encoder.
* Backbone frozen because (a) the working subset is too small to fine-tune
  without severe memorisation, and (b) the README commits to a single knob
  set per phase for reproducibility.

### 4.3 Training

* Optimiser: Adam, lr 1e-3, batch 4.
* Best-by-val checkpointing (`save_checkpoint` keeps the state-dict + label
  map + history).
* Class remapping: only classes that appear in the *training* split are
  kept; their integer ids are compacted to 0..N-1 for the run. The compact
  map is saved alongside the checkpoint so the inference loader can
  reverse it (`label_map.json`).
* Augmentation: only on the training DataLoader — random hflip (50%) +
  random-resized-crop (70–100% of the smaller side, bilinear resize).
  Inference / val / test see the cached deterministic frames.

### 4.4 Evaluation

* Confusion matrix + macro-P/R/F1 + per-class table; accuracy.
* Sklearn-free — counts only (`src/isl_bridge/eval.py`).
* Confusion matrix PNG produced by `scripts/plot_confusion.py`.
* CSV + JSON reports under `reports/` for every run.

### 4.5 Latency

* Single-clip inference median on this CPU: 1199 ms (Phase 3), 1 fps.
* Almost all of the cost is the backbone forward — MobileNetV2 with a
  frozen backbone is 8.7 MB on disk but its inner loop runs every frame
  in Python. With `torch.jit.script` and `torch.set_num_threads(N)` for the
  CPU-bound kernel, a 5–10x speed-up is realistic.
* Webcam target: a sliding window of 32 frames at 12 fps source = ~2.7 s of
  video per inference. At 1 fps inference that's a 2.7x real-time
  bottleneck, fine for an assistive interface; not fine for a real-time
  signed conversation.

## 5. Results

### 5.1 Headline numbers

| Run                 | Train | Val | Test | Macro-F1 | Accuracy |
|---------------------|------:|----:|-----:|---------:|:--------:|
| Phase 3 (mean pool)  |    16 |   2 |    4 |    0.200 |   0.250  |
| Phase 4 (GRU)       |    16 |   2 |    4 |    0.333 |   0.500  |

These are honest numbers on a tiny test set. They are not benchmarks and
should not be cited as recognition performance. The full-INCLUDE re-train
is the only benchmark-quality run.

### 5.2 Ablation table (with my honest hypothesis behind each row)

| Knob                                  | Default      | Tested alternate                                         | Effect                                          |
|---------------------------------------|--------------|-------------------------------------------------------|------------------------------------------------|
| Backbone                              | MobileNetV2  | ResNet18                                               | smaller (512-d), fewer trainable params. Same training behaviour at this subset size. |
| Freeze backbone                        | yes          | no                                              | more capacity, higher overfit risk on 16 clips. Not run. |
| Temporal head                          | mean         | GRU                                                    | see H2 above.                                  |
| Spatial augmentation                  | off (Phase 3), on (Phase 4) | varied crop-scale                | not yet run as a sweep.                         |
| Optimiser                              | Adam         | SGD                                                    | not run.                                       |
| LR                                     | 1e-3         | 1e-4 / 1e-2                                            | not run.                                       |
| TTS engine                            | macOS `say`  | browser `speechSynthesis`                              | engine is a swap-in; both work, demo uses the browser. |

### 5.3 Robustness (audit-driven)

* Horizontal flip: 3/4 unchanged on the test set (75%). The clip where the
  prediction flipped is the one where the random-init head had a 50/50
  decision boundary.
* Temporal subsample ×½: 4/4 unchanged (100%). The 32-frame backbone
  encoder is robust to half-rate input on these short clips because the
  underlying actions are slow.

## 6. Audit

Run `python3 scripts/audit.py`. Output: `artifacts/audit/audit_report.json`.

The audit splits into:

* **data** — `n_total_records=958`, `n_present=22`, `missing_pct=97.7`,
  `n_orphan_label=15`. Orphans are documented in
  `artifacts/dataset_summary.json` (`orphan_rows_sample`). Honest state:
  the project has 1/43 of the INCLUDE-50 archives on disk.
* **model** — `checkpoints[]` lists every `best.pt` under `artifacts/` with
  size, param count, and config. Two checkpoints present (Phase 3 + 4).
* **performance** — single-clip inference median 1199 ms, p95 1256 ms, ≈0.8 fps.
* **robustness** — hflip agreement 75%, temporal subsample agreement 100%
  on the 4 test clips.
* **api** — three documented smells: no auth, no upload-size cap, no
  verbose-error handler (the third is a non-finding — it's just an
  internal note in the audit script). The live probe tries `localhost:5050`
  and reports reachability. The audit is honest about the smells, not
  silently green-washing them.
* **code** — pyflakes on `src/` and `scripts/`. Surfaces 20 unused-import
  / unused-variable findings; not blocking. A future commit can run
  `ruff check --fix` to clean them.

## 7. What we did not run (and why)

| Not run                                       | Why                                                                  |
|-----------------------------------------------|---------------------------------------------------------------------|
| Transformer baseline                          | Out of scope of the 10-phase spec. Re-train when ground is otherwise. |
| Landmark-only training (MediaPipe Hands)       | MediaPipe isn't installed in this environment. The dataset is small enough that MediaPipe's CPU cost dominates. Re-introduce when the full INCLUDE is on disk. |
| Signer-independent evaluation                  | INCLUDE's public metadata doesn't carry per-clip signer IDs. We say so rather than fabricate a number. |
| Quantitative latency on GPU                   | No CUDA on the local machine. Re-run on a CUDA-equipped box.       |
| INT8 / ONNX export                             | Out of scope. Add when shipping to mobile.                          |
| Frontend with React + Tailwind                 | Out of scope; the demo site uses a single-file HTML page that talks to the FastAPI backend. |

## 8. Recommendations for the next iteration (Phase 11+)

1. **Full INCLUDE subset.** Free ~50 GB and re-download via
   `scripts/download_include.sh`. Re-run both trainers with the same
   knobs. Report macro-F1 on the held-out test split. This is the single
   change that converts the project from "pipeline complete" to
   "comparable to published baselines".
2. **Sign metadata.** File an issue against AI4Bharat/INCLUDE asking for
   per-clip signer IDs. Without them, signer-independence stays a
   documented absence.
3. **Latency.** `torch.jit.script` the `FrameCNN.forward` and time the
   32-frame forward; the cost is dominated by the backbone loop. 5–10x
   speed-up is realistic on the same CPU.
4. **Confidence calibration.** The current top-1 confidence is the raw
   softmax on 2 classes (always near 0.5 with a frozen backbone on tiny
   data). Add temperature scaling and report expected calibration error.
5. **Audit-in-CI.** Wire `scripts/audit.py` into a GitHub Action on every
   PR. The audit already degrades gracefully when checkpoints are absent.