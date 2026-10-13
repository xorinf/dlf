"""End-to-end smoke for the demo stack.

Boots the FastAPI app in-process (uvicorn programmatically) with the real
phase-4 checkpoint, waits for /health, then exercises every public endpoint
with a tiny batch of synthetic JPEG frames. No real network, no subprocesses
beyond what uvicorn already does, no real webcam.

Ponytail ceiling: one happy-path test per endpoint. If you need negative
tests (bad mime, missing file, model not loaded), they're nice-to-have but
not blocking the demo.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Point the API at the real phase-4 checkpoint for this test only.
CHECKPOINT = ROOT / "artifacts" / "phase4_real_run" / "best.pt"
if not CHECKPOINT.is_file():
    pytest.skip(f"checkpoint missing: {CHECKPOINT}", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    # We import the API lazily so the skip above can short-circuit without
    # pulling in torch for a missing-checkpoint environment.
    import scripts.api as api  # type: ignore
    api._STATE["checkpoint"] = str(CHECKPOINT)
    # The lifespan loads the model. We use the TestClient context manager so
    # that happens exactly once for the whole module.
    with TestClient(api.app) as c:
        # /health may take a beat to come up while the model loads.
        for _ in range(40):
            r = c.get("/health")
            if r.status_code == 200:
                break
            time.sleep(0.5)
        else:
            pytest.fail("API /health never returned 200")
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert "classes" in d and len(d["classes"]) >= 1


def test_classes(client):
    r = client.get("/classes")
    assert r.status_code == 200
    d = r.json()
    assert "classes" in d
    # Every value in idx_to_class should be a non-empty string.
    assert all(isinstance(v, str) and v for v in d["classes"].values())


def test_predict_frames_synthetic(client):
    """Build 16 random JPEG frames, POST them as multipart, expect a top-k
    JSON with at least one entry. We don't assert on the *value* (the model
    is too small for that to mean anything on noise); we assert on the
    *shape* of the response — that's the contract the UI relies on."""
    rng = np.random.default_rng(0)
    files = []
    for i in range(16):
        img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok, f"cv2.imencode failed at frame {i}"
        files.append(("files", (f"f{i:03d}.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")))
    r = client.post("/predict_frames?top_k=3&abstain_threshold=0", files=files)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d.keys()) == {"prediction", "top1_confidence", "top_k"}
    assert isinstance(d["top_k"], list) and len(d["top_k"]) >= 1
    for entry in d["top_k"]:
        assert "class" in entry and "confidence" in entry
        assert 0.0 <= entry["confidence"] <= 1.0
    assert 0.0 <= d["top1_confidence"] <= 1.0


def test_predict_video_synthetic(client):
    """Encode a tiny synthetic .mp4 in-memory and POST it. Same contract as
    /predict_frames but exercises the ffmpeg/cv2 decode path."""
    rng = np.random.default_rng(1)
    # 32 frames at 20fps to match the training default.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 240, 320
    writer = cv2.VideoWriter("/tmp/_isl_e2e.mp4", fourcc, 20.0, (w, h))
    try:
        for _ in range(32):
            frame = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    with open("/tmp/_isl_e2e.mp4", "rb") as f:
        data = f.read()
    r = client.post(
        "/predict?top_k=3&abstain_threshold=0",
        files={"file": ("clip.mp4", io.BytesIO(data), "video/mp4")},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert "prediction" in d and "top_k" in d and len(d["top_k"]) >= 1
