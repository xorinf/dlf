"""Phase 8 — FastAPI backend serving the trained CNN.

Single endpoint: `POST /predict` accepting an uploaded `.MOV` / `.MP4`,
running the same preprocess pipeline as `scripts/infer_cnn.py`, and
returning the top-k predictions.

A second endpoint: `GET /health` returns `{"status": "ok", "classes": [...]}`.
A third: `GET /classes` lists the model's known words.

The frontend (Phase-8 backfill) is out of scope for this repo: a single-page
React/Tailwind UI is a separate stack. The API is shaped so a UI can plug in
without server changes.

Run locally:
    PYTHONPATH=src python3 scripts/api.py \
        --checkpoint artifacts/phase3_real_run/best.pt \
        --host 127.0.0.1 --port 5050

# ponytail: single uvicorn process, no auth, no async batch endpoint. A real
#            deployment would add HTTPS + auth — keep it local until Phase 10.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.checkpoint import load_checkpoint  # noqa: E402
from isl_bridge.infer import _prep_video  # noqa: E402

log = logging.getLogger("isl_bridge.api")

_STATE: dict[str, Any] = {}

# ponytail: per-process max upload. ~64MB — generous for a short ISL clip,
# rejects the obvious abuse case (gigabytes of arbitrary bytes) before they
# hit RAM. Bump if the teacher demos 4K sources.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@asynccontextmanager
async def _lifespan(app: FastAPI):
    ckpt = Path(_STATE["checkpoint"])
    model, cfg, idx_to_class, history = load_checkpoint(ckpt)
    model.eval()
    _STATE["model"] = model
    _STATE["cfg"] = cfg
    _STATE["idx_to_class"] = idx_to_class
    _STATE["history"] = history
    log.info("loaded %s, classes=%d", ckpt, len(idx_to_class))
    yield
    _STATE.clear()


app = FastAPI(title="ISL-Bridge API", version="0.1", lifespan=_lifespan)

# ponytail: open CORS only for localhost/127.0.0.1 on any port — same-machine
#            demo. Tighten to an allow-list before any deploy beyond this Mac.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost", "http://127.0.0.1",
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    if "model" not in _STATE or "idx_to_class" not in _STATE:
        # Lifespan didn't run or the model load raised. Surface a real 503 so
        # the UI's status bar reads "API offline" instead of lying.
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok",
            "classes": sorted(_STATE["idx_to_class"].values()),
            "model": _STATE["model"].__class__.__name__}


@app.get("/classes")
def classes() -> dict:
    return {"classes": _STATE.get("idx_to_class", {})}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5,
                  abstain_threshold: float = 0.0) -> JSONResponse:
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    if file.content_type not in ("video/mp4", "video/quicktime", "video/webm",
                                   "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"unsupported content-type {file.content_type}")
    if file.content_type == "video/quicktime":
        suffix = ".MOV"
    elif file.content_type == "video/webm":
        suffix = ".webm"
    else:
        suffix = ".mp4"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload too large ({len(raw)} > {MAX_UPLOAD_BYTES})")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
        fp.write(raw)
        tmp = Path(fp.name)
    try:
        try:
            x, mask = _prep_video(tmp)
        except RuntimeError as e:
            # _prep_video raises when cv2 can't decode — most common path for
            # a bad .mov. Surface as 400, not 500.
            raise HTTPException(status_code=400, detail=f"decode failed: {e}")
        return _predict_tensor(x, top_k=top_k, abstain_threshold=abstain_threshold)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/predict_frames")
async def predict_frames(
    files: list[UploadFile] = File(...),
    top_k: int = 5,
    abstain_threshold: float = 0.0,
) -> JSONResponse:
    """Live webcam path: browser sends a batch of JPEG frames (default 32, ~1.6s
    at 20 fps). Decoded + normalised with the same pipeline as /predict, then
    the shared post-processor runs. Ponytail ceiling: no batching across
    concurrent users — single-process, request-per-batch. If two browsers
    stream at once we serve them serially. Add an inference queue if it bites.
    """
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(files) < 8:
        raise HTTPException(status_code=400, detail=f"need >=8 frames, got {len(files)}")
    # Decode each JPEG -> BGR uint8. Cap total bytes early.
    bgrs: list[np.ndarray] = []
    total = 0
    for f in files:
        data = await f.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"upload too large (>{MAX_UPLOAD_BYTES})")
        if not data:
            raise HTTPException(status_code=400, detail=f"empty frame: {f.filename}")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"could not decode frame {f.filename}")
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            # Defensive: cv2.imdecode can return a 0x0 ndarray on malformed
            # input that the None check above didn't catch. s = SIZE/max(h,w)
            # would divide by zero.
            raise HTTPException(status_code=400, detail=f"empty image: {f.filename}")
        bgrs.append(img)
    # Build (1, T, 3, H, W) tensor with the same resize + normalise as /predict.
    from isl_bridge.preprocess import DEFAULT_FRAMES, DEFAULT_SIZE
    from isl_bridge.dataloaders import IMAGENET_MEAN, IMAGENET_STD
    n_frames = DEFAULT_FRAMES
    # If the browser sent more, take the most recent n_frames; if fewer, tile.
    if len(bgrs) >= n_frames:
        bgrs = bgrs[-n_frames:]
    else:
        reps = int(np.ceil(n_frames / len(bgrs)))
        bgrs = (bgrs * reps)[:n_frames]
    rgb = np.stack([b[:, :, ::-1] for b in bgrs], axis=0)            # (T,H,W,3) RGB
    # Resize-aspect-pad to DEFAULT_SIZE (matches _resize_keep_aspect).
    h, w = rgb.shape[1], rgb.shape[2]
    scale_denom = max(h, w)
    if scale_denom == 0:
        raise HTTPException(status_code=400, detail="all frames are 0x0")
    s = DEFAULT_SIZE / scale_denom
    new_w, new_h = int(round(w * s)), int(round(h * s))
    out = np.zeros((n_frames, DEFAULT_SIZE, DEFAULT_SIZE, 3), dtype=rgb.dtype)
    for i in range(n_frames):
        rs = cv2.resize(rgb[i], (new_w, new_h),
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out[i, :new_h, :new_w] = rs
    x = torch.from_numpy(out).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0)                                # (1,T,3,H,W)
    return _predict_tensor(x, top_k=top_k, abstain_threshold=abstain_threshold)


@app.get("/sample_predict")
async def sample_predict(top_k: int = 5, abstain_threshold: float = 0.0) -> JSONResponse:
    """Predict on the first npz in data/processed/test — useful when no upload
    is available. Lets the demo site exercise the model end-to-end without the
    user picking a file."""
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    # Resolve test-dir relative to the repo root, not the current CWD, so
    # `python -m uvicorn` from anywhere still finds the cache.
    test_dir = ROOT / "data" / "processed" / "test"
    if not test_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"no test cache at {test_dir}; "
                                "run the include-style prep first")
    test_files = sorted(test_dir.rglob("*.npz"))
    if not test_files:
        raise HTTPException(status_code=404, detail="no test npz files on disk; "
                                "run the include-style prep first")
    from isl_bridge.infer import _prep_npz
    x, mask = _prep_npz(test_files[0])
    return _predict_tensor(x, top_k=top_k, abstain_threshold=abstain_threshold)


def _predict_tensor(x: torch.Tensor, mask: torch.Tensor | None = None,
                    *, top_k: int, abstain_threshold: float) -> JSONResponse:
    """Shared post-processor used by /predict, /sample_predict, /predict_frames."""
    # Cap top_k so a malicious / curious client can't ask for top_k=10000.
    n_classes = _STATE["model"].cfg.num_classes if hasattr(_STATE["model"], "cfg") \
        else len(_STATE["idx_to_class"])
    top_k = max(1, min(int(top_k), int(n_classes)))
    with torch.no_grad():
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.uint8)
        logits = _STATE["model"](x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if probs.size == 0:
        return JSONResponse({"prediction": "n/a", "top1_confidence": 0.0,
                             "top_k": []})
    order = np.argsort(probs)[::-1][:top_k]
    top_k_list = [{"class": _STATE["idx_to_class"].get(int(i), str(int(i))),
                   "confidence": round(float(probs[i]), 4)} for i in order]
    top1_conf = float(probs[order[0]])
    pred = top_k_list[0]["class"] if top_k_list else "n/a"
    if abstain_threshold > 0 and top1_conf < abstain_threshold:
        pred = "uncertain"
    return JSONResponse({"prediction": pred, "top1_confidence": round(top1_conf, 4),
                         "top_k": top_k_list})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    args = p.parse_args()
    _STATE["checkpoint"] = str(args.checkpoint)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())