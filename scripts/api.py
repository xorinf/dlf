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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.checkpoint import load_checkpoint  # noqa: E402
from isl_bridge.infer import _prep_video  # noqa: E402

log = logging.getLogger("isl_bridge.api")

_STATE: dict[str, Any] = {}


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
    return {"status": "ok",
            "classes": sorted(_STATE.get("idx_to_class", {}).values())}


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
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
        fp.write(await file.read())
        tmp = Path(fp.name)
    try:
        x, mask = _prep_video(tmp)
        return _predict_tensor(x, top_k=top_k, abstain_threshold=abstain_threshold)
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/sample_predict")
async def sample_predict(top_k: int = 5, abstain_threshold: float = 0.0) -> JSONResponse:
    """Predict on the first npz in data/processed/test — useful when no upload
    is available. Lets the demo site exercise the model end-to-end without the
    user picking a file."""
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    test_files = sorted((Path("data/processed/test")).rglob("*.npz"))
    if not test_files:
        raise HTTPException(status_code=404, detail="no test clips on disk; "
                                "run the include-style prep first")
    from isl_bridge.infer import _prep_npz
    x, mask = _prep_npz(test_files[0])
    return _predict_tensor(x, top_k=top_k, abstain_threshold=abstain_threshold)


def _predict_tensor(x: torch.Tensor, mask: torch.Tensor | None = None,
                    *, top_k: int, abstain_threshold: float) -> JSONResponse:
    """Shared post-processor used by both /predict and /sample_predict."""
    with torch.no_grad():
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.uint8)
        logits = _STATE["model"](x, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
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