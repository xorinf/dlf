# ISL-Bridge — Real-Time Indian Sign Language Recognition

Real-time isolated-word **Indian Sign Language** recognition that converts webcam
video into recognized words, with an optional text-to-speech output.

This repository accompanies the *ISL-Bridge* university project: a 10-phase build
of a research-grounded, measurable, reproducible recognition system.

> **Status:** Phase 1 — Research & project design.

## What this project is

A working app where a user signs an ISL word in front of the webcam; the system
outputs the predicted word and its confidence, appends it to a transcript, and
optionally speaks it. The contributions are measurable: a baseline CNN vs. a
spatiotemporal model vs. landmarks, evaluated under signer-independent splits
where the dataset supports it.

## What this project is NOT

- Not a translation system. Phase 7 explicitly separates isolated-word recognition
  from continuous SL translation. We do not invent grammar.
- Not a synthetic demo. Every reported number comes from an actual evaluation
  run that another developer can reproduce from this repo.

## Phase checklist

| #  | Phase                                | Status   | Notes                                                                  |
|----|--------------------------------------|----------|------------------------------------------------------------------------|
| 1  | Research & project design            | **current** | Dataset, scope, architecture, evaluation plan. See `docs/phase1.md`. |
| 2  | Dataset preparation                  | pending  | INCLUDE downloader, EDA, splits, augmentation.                          |
| 3  | CNN baseline                         | pending  | MobileNetV2 / ResNet18 frame-level baseline.                            |
| 4  | Spatiotemporal model                 | pending  | CNN backbone + GRU/BiLSTM; optional MediaPipe landmarks.               |
| 5  | Improvement & robustness             | pending  | Augmentation, class imbalance, calibration, ablations.                  |
| 6  | Real-time recognition                | pending  | Webcam capture, sliding window, smoothing, latency.                     |
| 7  | Text & speech interface              | pending  | Transcript UI, optional TTS, abstain on low confidence.                 |
| 8  | Application development              | pending  | FastAPI backend + React/Tailwind frontend.                              |
| 9  | Testing & evaluation                 | pending  | Confusion matrix, signer-independence, FPS, model size.                 |
| 10 | Deployment & final submission        | pending  | README, repo, demo, report.                                             |

Phases are unlocked sequentially — the gate before each phase is verified before
the next begins (see `docs/phase1.md` § "Gates").

## Repository layout

```
.
├── README.md
├── docs/                  # per-phase design notes
├── src/                   # training & inference code (populated from phase 2 onward)
├── scripts/               # data download & utility scripts (populated from phase 2 onward)
├── data/                  # dataset artifacts (NOT committed — see .gitignore)
└── .hermes/               # internal agent notes (per-session)
```

## Quickstart

Not applicable until Phase 2 lands. See `docs/phase1.md` for the dataset decision
and the planned reproduction pipeline.

## License

Project code: MIT (placeholder, finalised at Phase 10).
Training data: INCLUDE — CC-BY-4.0 (Sridhar et al., ACM MM 2020).
