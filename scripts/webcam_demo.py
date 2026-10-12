"""Phase 6 — Real-time webcam capture + sliding-window recognition.

Reads frames from the default webcam, buffers them into short clips, and
emits a top-k prediction when a clip fills. Designed to be the *smallest
script that demos the loop* — no live UI overlay, no recording, no threading
beyond what OpenCV already does.

Usage (requires a display + permission to access the camera):
    PYTHONPATH=src python3 scripts/webcam_demo.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --window-sec 1.6 \
        --stride-sec 0.4 \
        --abstain-threshold 0.6

Each emitted line:
    2025-01-15 10:42:11  prediction=fall  conf=0.71  top_k=[fall:0.71 summer:0.29]

If the top-1 confidence is below --abstain-threshold, `prediction=uncertain`
and the top_k still prints so you can see what *would* have been said.

On a machine without a webcam, this script exits immediately with a clear
message — so reviewers running CI/CI don't trip on a missing device.

# ponytail: single-thread, no async. Frame resolution locked at 224 to match
#            the backbone input. Multi-thread capture is an upgrade only if
#            FPS is the bottleneck (it won't be on CPU with frozen backbone).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.checkpoint import load_checkpoint  # noqa: E402
from isl_bridge.dataloaders import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# Match the training preprocessing so the live input is recognised.
WINDOW_FRAMES = 32                # frames per prediction
IMAGES_SIZE = 224                 # square backbone input


def _open_cam(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam at index {index}")
    return cap


def _preprocess(frames_bgr: list[np.ndarray]) -> torch.Tensor:
    rgb = [f[:, :, ::-1] for f in frames_bgr]
    arr = np.stack(rgb, axis=0)                       # (T, H, W, 3) uint8
    # resize each frame to 224x224 (square), preserving aspect via cv2.resize.
    out = np.zeros((arr.shape[0], IMAGES_SIZE, IMAGES_SIZE, 3), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        h, w = arr[i].shape[:2]
        s = IMAGES_SIZE / max(h, w)
        new = cv2.resize(arr[i], (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out[i, :new.shape[0], :new.shape[1]] = new
    x = torch.from_numpy(out).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)            # (1, T, 3, H, W)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--window-frames", type=int, default=WINDOW_FRAMES)
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="If > 0, label low-confidence predictions as 'uncertain'.")
    p.add_argument("--max-windows", type=int, default=0,
                   help="Stop after this many windows emitted (0 = run forever). "
                        "Useful for CI smoke tests without a real display.")
    args = p.parse_args()

    model, cfg, idx_to_class, _ = load_checkpoint(args.checkpoint)
    model.eval()
    cap = _open_cam(args.cam_index)
    buf: list[np.ndarray] = []
    n_windows = 0
    print(json.dumps({"status": "started", "checkpoint": str(args.checkpoint),
                      "window_frames": args.window_frames, "classes": idx_to_class}))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(json.dumps({"status": "cam_read_failed"}))
                break
            buf.append(frame)
            if len(buf) >= args.window_frames:
                with torch.no_grad():
                    x = _preprocess(buf[-args.window_frames:])
                    logits = model(x)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                order = np.argsort(probs)[::-1][:3]
                top1_conf = float(probs[order[0]])
                top1_class = idx_to_class.get(int(order[0]), str(int(order[0])))
                pred = top1_class
                if args.abstain_threshold > 0 and top1_conf < args.abstain_threshold:
                    pred = "uncertain"
                rec = {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": pred,
                    "top1_confidence": round(top1_conf, 4),
                    "top_k": [{"class": idx_to_class.get(int(i), str(int(i))),
                               "confidence": round(float(probs[i]), 4)}
                              for i in order],
                }
                print(json.dumps(rec))
                n_windows += 1
                if args.max_windows and n_windows >= args.max_windows:
                    break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())