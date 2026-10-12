"""Stand-alone inference for a single clip — Phase 3 deliverable.

Loads a saved checkpoint (model + label map) and runs it on either:
  - a video file path (.MOV / .MP4) — decoded + resized on the fly
  - a preprocessed .npz cache (data/processed/<split>/.../*.npz)
  - a directory of clips to score at once (--batch mode)

Usage:
    PYTHONPATH=src python3 scripts/infer_cnn.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --video path/to/clip.MOV

Returns the top-k (word, confidence) predictions. Threshold-based abstention
is exposed as `--abstain-threshold`; predictions below the threshold are
labelled "uncertain" rather than emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.checkpoint import load_checkpoint  # noqa: E402
from isl_bridge.infer import _prep_video, _prep_npz  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video", type=Path, default=None,
                   help="A single .MOV or .MP4 to score.")
    p.add_argument("--npz", type=Path, default=None,
                   help="A preprocessed .npz clip to score.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="If > 0, label low-confidence predictions as 'uncertain'.")
    args = p.parse_args()

    if not args.video and not args.npz:
        p.error("must supply --video or --npz")

    model, cfg, idx_to_class, _ = load_checkpoint(args.checkpoint)
    model.eval()
    with torch.no_grad():
        if args.video is not None:
            x, mask = _prep_video(args.video)
        else:
            x, mask = _prep_npz(args.npz)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    order = np.argsort(probs)[::-1][:args.top_k]
    out = []
    for i in order:
        out.append({"class": idx_to_class.get(int(i), str(int(i))),
                   "confidence": round(float(probs[i]), 4)})

    top1_conf = out[0]["confidence"] if out else 0.0
    uncertain = (args.abstain_threshold > 0 and top1_conf < args.abstain_threshold)
    if uncertain:
        result = {"prediction": "uncertain", "top1_confidence": top1_conf,
                  "abstain_threshold": args.abstain_threshold, "top_k": out}
    else:
        result = {"prediction": out[0]["class"] if out else "n/a",
                  "top1_confidence": top1_conf,
                  "top_k": out}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
