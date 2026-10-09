# Phase 1 — Research & Project Design

## 1. Problem statement

Build a **real-time Indian Sign Language (ISL) isolated-word recognizer** that:

1. Accepts a live webcam stream.
2. Recognizes a single signed ISL word per gesture.
3. Emits the predicted gloss with a calibrated confidence score and an explicit
   *abstain* state when uncertain.
4. Surfaces the result in an accessible UI (text transcript, optional speech).

### Out of scope (explicit)

- **Continuous / sentence-level translation.** Isolated gloss recognition is the
  unit of evaluation. We do not synthesize grammar or sentence meaning.
- **Signer-independent production claims beyond what the dataset supports.**
- **Two-way dialogue, named-entity grounding, ISL→English syntax.** Those would
  change the problem class.

### Why this problem class

ISL has an estimated 5–15M users in South Asia (FdMSE-ISL, 2024 — arXiv:2407.14224)
and limited real-time tooling compared to ASL. A measured, isolated-word
recognizer with an *abstain* policy is a defensible contribution: the published
baselines (Sridhar et al., 2020) report 85.6% top-1 across 263 classes, leaving
room to ask what a spatiotemporal model + landmark features improve, and whether
the system can refuse low-confidence predictions.

## 2. Research gaps & limitations of existing work

What the literature does well and where it stops short:

| Existing work                      | Strength                                         | Limitation                                                       |
|------------------------------------|-------------------------------------------------|------------------------------------------------------------------|
| **INCLUDE** (Sridhar et al., 2020) | Largest open ISL dataset; 263 classes; paper numbers (85.6% / 94.5%) are reproducible | Captured with 7 senior students at one school; signer-independence evaluated post-hoc only; static-image augmentations, no temporal smoothing for live use |
| **CISLR** (Joshi et al., EMNLP 2022) | Larger signer pool | Not as widely re-used for isolated-word recognition; gated access |
| **FDMSE-ISL** (arXiv 2407.14224)   | 2002 classes, 20 deaf signers, true signer-independence | **Available on request only** (cs.rkmvu.ac.in/~isl) — not safe for a reproducible pipeline |
| **iSign** (arXiv 2407.05404)       | Continuous translation                           | Gated; poses a translation problem, not isolated recognition     |
| **ISLRTC dictionary**              | Authoritative vocab (~10–15K terms)              | Videos hosted but not redistributable in bulk as a clean labeled set for ML |

**Research gap I can actually address here:** most offline benchmarks stop at
overall accuracy. Real-time SLR systems need *calibrated confidence + abstain*
behaviour to be safe to use. I treat the model's confidence as a first-class
deliverable, not an afterthought. That is the project's measurable contribution
alongside the baseline-vs-temporal comparison.

## 3. Dataset comparison

| Dataset        | Size                | Classes | Signers | Public download | License      | Practical for this project |
|----------------|---------------------|---------|---------|------------------|--------------|----------------------------|
| **INCLUDE**    | 4,292 videos / 56.8 GB across 18 zip parts | 263 | 7 (school) | Yes — Zenodo 4010759 with author bash script | CC-BY-4.0 | **Yes — primary.** Manageable subset, official code, paper numbers. |
| INCLUDE-50     | 958 videos          | 50      | 7       | Same source      | CC-BY-4.0    | Smoke-test / ablation subset |
| **FDMSE-ISL**  | 40,033 videos       | 2002    | 20 (deaf, gender-balanced, signer-indep split) | Upon request only | Research-only | **No** — gated access breaks reproducibility |
| **ISLRTC Dictionary** | ~10–15K videos, ~200 GB re-encoded | thousands | n/a | Catalog page lists links to per-word HTML pages; bulk download not sanctioned | Government of India terms | **No** — fetcher would scrape a running website; license ambiguous for bulk redistribution |
| iSign          | 118K video-sentence pairs | continuous | n/a | Gated (HF accept terms) | Research-only | **No** — wrong task class (continuous), access gated |

### Decision: **INCLUDE** (full + INCLUDE-50 subset for fast iteration).

- It is the only open, large, explicitly-labeled ISL isolated-word dataset I can
  download and reproduce against without violating access terms.
- Has an official reference code (`AI4Bharat/INCLUDE`) so published numbers can
  be sanity-checked.
- Includes the authors' official train/test CSV splits.

### Disk / bandwidth handling

The full INCLUDE archive is **~56.8 GB**. We will not assume that fits the
working environment. The dataset loader (Phase 2) will:

1. Default to the **INCLUDE-50** subset (≈ 958 videos, far smaller) for the
   baseline + temporal ablation so the pipeline can be iterated end-to-end on a
   single laptop.
2. Make the full 263-class INCLUDE an *opt-in* download via the author script,
   gated on enough disk.
3. Fail loudly if INCLUDE's Zenodo links change — never silently substitute
   another dataset.

### License, attribution, restrictions

- License: **CC-BY-4.0** — preserve attribution, indicate changes.
- Citation: Sridhar, Ganesan, Kumar, Khapra. *INCLUDE: A Large Scale Dataset
  for Indian Sign Language Recognition.* ACM MM 2020. DOI 10.1145/3394171.3413528.
- Videos were recorded at St. Louis School for the Deaf, Adyar, Chennai.

## 4. Scope and objectives

### In scope

- Isolated-word recognition on INCLUDE.
- Two models: (a) lightweight per-frame CNN baseline, (b) spatiotemporal
  CNN+GRU/BiLSTM, optionally fused with MediaPipe hand/pose landmarks.
- A real-time webcam demo with a *neutral / abstain* state.
- UI: live transcript, undo, optional speech output.
- Rigorous metrics: top-1 accuracy, macro precision/recall/F1, confusion
  matrix, signer-independent accuracy where the data supports it, latency,
  effective FPS, model size.

### Out of scope (re-stated)

- Continuous / sentence ISL translation.
- Face / facial-expression modelling beyond what MediaPipe Holistic exposes by
  default (used only because the landmark pipeline is free).
- Mobile deployment beyond an exportable ONNX weight.

### Success criteria (Phase 10 demonstrator)

| Metric                                | Target                                 |
|---------------------------------------|----------------------------------------|
| Top-1 on INCLUDE-50 (signer-aware split) | ≥ 90% (paper baseline 94.5%)         |
| Top-1 on full INCLUDE (signer-aware)  | ≥ 80% (paper baseline 85.6%)           |
| Effective webcam FPS on dev machine   | ≥ 10 FPS on M-class CPU; ≥ 20 on GPU   |
| Real-time abstain precision@recall=.9 | ≥ 0.85 — measured, not asserted        |

These are **targets**, not claimed in advance. Phase 9 reports what we actually
get, honest.

## 5. System architecture (proposed)

```
┌──────────────┐   frames (MJPEG)
│  Webcam      │ ────────────────────┐
└──────────────┘                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │  Frame producer (OpenCV)    │
                       │  - decode / resize          │
                       │  - temporal buffer (window) │
                       └────────────┬────────────────┘
                                    │ windowed clip
                                    ▼
                       ┌─────────────────────────────┐
                       │  Preprocess                 │
                       │  - normalize, optional      │
                       │    MediaPipe Holistic       │
                       │    -> 2D keypoints          │
                       └────────────┬────────────────┘
                                    │ RGB tensor [+ landmarks]
                                    ▼
                       ┌─────────────────────────────┐
                       │  Model (PyTorch)            │
                       │  - CNN backbone            │
                       │  - temporal head (GRU)     │
                       │  - softmax logits          │
                       └────────────┬────────────────┘
                                    │ logits + confidence
                                    ▼
                       ┌─────────────────────────────┐
                       │  Smoothing / abstain       │
                       │  - EMA, vote, threshold    │
                       │  - return "uncertain"      │
                       └────────────┬────────────────┘
                                    │ word (or abstain)
                                    ▼
        ┌──────────────────────────────────────────────┐
        │  App layer (FastAPI + React/Tailwind)        │
        │  - transcript                                │
        │  - undo / clear                              │
        │  - optional TTS (pyttsx3 / Web Speech API)   │
        └──────────────────────────────────────────────┘
```

Two model flavours will live under `src/models/`:

- `cnn_baseline.py` — single-frame classifier on a MobileNetV2/ResNet18 backbone,
  used to confirm the data pipeline and produce a confusion matrix.
- `spatiotemporal.py` — CNN backbone + GRU/BiLSTM. Optional MediaPipe
  landmark features concatenated before the temporal head.

Inference path is shared (`src/infer.py` → model → smoothing → output). Training
and evaluation live under `src/{train,eval}`.

## 6. Baseline models, evaluation, experiments

### Baselines

- **B0 — frame-level CNN.** ResNet-18 (ImageNet pretrained) head over the
  single most-confident frame per clip. Cheap sanity baseline.
- **B1 — spatiotemporal CNN+GRU.** ResNet-18 frame features → 2-layer GRU →
  softmax. The lift over B0 measures what *temporal modelling* buys us.

### Optional fused variant

- **B2 — spatiotemporal + landmarks.** Same as B1 with MediaPipe Holistic
  keypoints concatenated before the GRU. Used only if the comparison is
  meaningful (similar training set, similar compute budget).

### Metrics

Per split, per model:
- Top-1 accuracy
- Top-5 accuracy
- Macro precision, recall, F1
- Confusion matrix (saved as CSV + PNG)
- Inference latency (ms / clip) and effective FPS (clips / sec on webcam)
- Model size (MB, FP32 + INT8 if quantized)

### Splits

INCLUDE's official `train.csv` / `test.csv` (Sridhar et al., 2020) for the
**signer-aware** numbers — i.e., the same comparison the paper itself reports.

A **signer-independent** holdout evaluation is reported only if signer IDs are
useable in the metadata. INCLUDE does not currently expose a per-video signer
field in the public CSVs; if absent, we honestly say so and avoid claiming
signer-independence results.

### Experiments (planned)

| E# | Question                                                        | What changes                   |
|----|-----------------------------------------------------------------|-------------------------------|
| E1 | Does the temporal head help over a frame baseline?              | B0 vs B1                      |
| E2 | Does RGB-only match a landmark-only model?                      | B1 RGB vs landmark-only head  |
| E3 | Does fusing landmarks help?                                     | B1 vs B2                      |
| E4 | Do augmentations (spatial + temporal) help generalize?          | train with/without            |
| E5 | Does the abstention threshold behave as designed?               | precision/recall vs threshold |

Every "improvement" claim must come out of one of these runs.

## 7. Why these decisions

- **PyTorch over TensorFlow**: torch 2.x is already installed; `python -c
  "import torch"` works. Avoids a fresh setup.
- **FastAPI + React (Phase 8)**: FastAPI is in the env; React + Tailwind is the
  stack called out in the spec. No rewrite needed if a Frontend skeleton
  already exists.
- **MediaPipe for landmarks**: small dep, robust on CPU, makes the *RGB vs
  landmark vs fusion* comparison tractable.
- **No backbone that needs special gating (3D-CNN, transformer over 32 frames)
  as the default**: keep compute honest on a laptop. A transformer baseline is
  mentioned in the INCLUDE paper and can be added later (Phase 4 gate) only if
  it shows a real lift and fits the training budget.
- **Abstain by default**: many sign-recognition demos appear to work because
  every frame emits *something*. A confidence gate + smoothing turns noise
  into honest "I don't know", which is the right behaviour for an assistive
  interface.

## 8. Gate (Phase 1 → Phase 2)

Moving on requires:

1. ✅ Selected primary dataset and confirmed public availability — **INCLUDE**
   on Zenodo, with the official bash script and the official train/test CSV.
2. ✅ Out-of-scope items written down so we don't drift.
3. ✅ Experiment plan written before any training job starts.

If later phases show INCLUDE can not be downloaded as described, the fallback
is INCLUDE-50 only — announced loudly, not silent.
