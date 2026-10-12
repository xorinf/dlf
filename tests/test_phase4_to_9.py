"""Phase 4–9 tests.

Four small tests, in the same Ponytail-friendly style as test_phase3.py:

  test_stemporal_shape        : StemporalCNN runs a CPU forward pass.
  test_stemporal_ckpt_roundtrip : save/load a GRU checkpoint round-trips.
  test_augment_shape_preserved : augment wrapper keeps (T,3,H,W) shape + dtype.
  test_load_dispatches_gru    : load_checkpoint() detects GRU checkpoints.

These tests use synthetic data only — they exist to prove the code paths
wire together, not to measure recognition performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isl_bridge.augment import wrap_train  # noqa: E402
from isl_bridge.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from isl_bridge.model import FrameCNN, ModelConfig  # noqa: E402
from isl_bridge.synthetic import (  # noqa: E402
    make_synthetic_clips, make_fake_splits_metadata,
)
from isl_bridge.stemporal import StemporalCNN  # noqa: E402
from isl_bridge import dataloaders  # noqa: E402


def _make_fake_root(tmp_path: Path) -> Path:
    fr = tmp_path
    fp = fr / "processed"
    fm = fr / "metadata"
    make_synthetic_clips(fp, n_train_per_class=2, n_val_per_class=1, n_test_per_class=1)
    make_fake_splits_metadata(fm, n_train_per_class=2, n_val_per_class=1, n_test_per_class=1)
    return fr


def test_stemporal_shape():
    cfg = ModelConfig(num_classes=3, pretrained=False, freeze_backbone=True, backbone="mobilenet_v2")
    m = StemporalCNN(cfg, gru_hidden=32).eval()
    B, T, C, H, W = 2, 4, 3, 32, 32
    x = torch.randn(B, T, C, H, W)
    mask = torch.ones(B, T, dtype=torch.uint8)
    y = m(x, mask)
    assert y.shape == (B, 3)
    assert y.dtype == torch.float32


def test_stemporal_ckpt_roundtrip(tmp_path: Path):
    cfg = ModelConfig(num_classes=2, pretrained=False, freeze_backbone=True, backbone="mobilenet_v2")
    model = StemporalCNN(cfg, gru_hidden=32).eval()
    idx = {0: "summer", 1: "fall"}
    ck = tmp_path / "gru.pt"
    save_checkpoint(ck, model=model, cfg=cfg, idx_to_class=idx, history=[{"epoch": 0}])
    loaded, cfg_loaded, idx_loaded, history = load_checkpoint(ck)
    # Dispatch should have built a StemporalCNN, not a FrameCNN.
    assert isinstance(loaded, StemporalCNN), f"expected StemporalCNN, got {type(loaded).__name__}"
    assert idx_loaded == idx
    a = dict(model.named_parameters())
    b = dict(loaded.named_parameters())
    assert set(a) == set(b), "parameter sets differ"
    for k in a:
        assert torch.equal(a[k].data, b[k].data), f"parameter {k!r} drifted"


def test_augment_shape_preserved(tmp_path: Path):
    fr = _make_fake_root(tmp_path)
    base = dataloaders.ClipDataset(
        "train", raw_dir=fr / "raw", proc_dir=fr / "processed", meta_dir=fr / "metadata",
    )
    aug = wrap_train(base)
    it = aug[0]
    assert it["frames"].dtype == torch.float32
    assert it["frames"].dim() == 4 and it["frames"].shape[1] == 3, \
        "frames should be (T, 3, H, W)"
    assert it["mask"].dtype == torch.uint8


def test_load_dispatches_gru(tmp_path: Path):
    # Build a FrameCNN checkpoint, save it, and confirm the loader doesn't
    # mistakenly try to load it as a GRU checkpoint.
    cfg = ModelConfig(num_classes=2, pretrained=False, freeze_backbone=True, backbone="mobilenet_v2")
    model = FrameCNN(cfg).eval()
    ck = tmp_path / "cnn.pt"
    save_checkpoint(ck, model=model, cfg=cfg, idx_to_class={0: "a", 1: "b"})
    out, _, _, _ = load_checkpoint(ck)
    assert isinstance(out, FrameCNN), f"expected FrameCNN, got {type(out).__name__}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))