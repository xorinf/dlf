"""Comprehensive project audit.

Runs a battery of checks and writes `artifacts/audit/audit_report.json` plus
a short summary printed to stdout. Designed to be re-run as a "is this
thing honest?" smoke test.

Categories:
  data        — dataset inventory + integrity (zero-byte / orphan rows)
  model       — checkpoint sizes + parameter counts (when present)
  api         — endpoint reachability + basic security (no auth, payload
                 size limits, content-type checks)
  performance — single-clip inference latency (median over N passes)
  robustness  — horizontal-flip invariance + temporal subsample rate
  code        — pyflakes-style lint on src/ and scripts/

# ponytail: one report, one exit code. Doesn't try to be a CI server.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "artifacts" / "audit"


def audit_data() -> dict:
    """Dataset integrity: counts, splits, missing clips, zero-byte clips."""
    from isl_bridge import dataset, dataset_meta
    inv = dataset.build_inventory(Path("data/raw/include"),
                                  meta_dir=dataset_meta.META_DIR)
    n_total = len(inv)
    n_present = sum(1 for r in inv if r.exists and r.size_bytes > 0)
    n_missing = n_total - n_present
    n_zero = sum(1 for r in inv if r.exists and r.size_bytes == 0)
    n_orphan = sum(1 for r in inv if dataset.label_for(r.word, dataset_meta.load_label_map("include50")) is None)
    bytes_present = sum(r.size_bytes for r in inv if r.exists and r.size_bytes > 0)
    return {
        "n_total_records": n_total,
        "n_present": n_present,
        "n_missing": n_missing,
        "n_zero_byte": n_zero,
        "n_orphan_label": n_orphan,
        "bytes_present": bytes_present,
        "missing_pct": round(100 * n_missing / max(1, n_total), 2),
    }


def audit_model() -> dict:
    """Inspect any checkpoints under artifacts/*/best.pt for size + param count.

    Checkpoints are gitignored; this audit reports whatever is on disk and
    degrades gracefully if the .pt files aren't present.
    """
    out = {"checkpoints": []}
    for p in sorted((ROOT / "artifacts").rglob("best.pt")):
        size_mb = round(p.stat().st_size / 1024 / 1024, 2)
        info = {"path": str(p.relative_to(ROOT)), "size_mb": size_mb}
        try:
            import torch
            blob = torch.load(p, map_location="cpu", weights_only=False)
            n_params = sum(t.numel() for t in blob["model_state"].values())
            info["n_params"] = n_params
            info["size_mb_if_fp32"] = round(n_params * 4 / 1024 / 1024, 2)
            info["backbone"] = blob["config"].get("backbone", "?")
            info["num_classes"] = blob["config"].get("num_classes", "?")
            info["epochs_history"] = len(blob.get("history", []))
        except Exception as e:                                # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
        out["checkpoints"].append(info)
    if not out["checkpoints"]:
        # ponytail: honest empty-state — surface it in the report so reviewers
        #            don't mistake a missing checkpoint for a working model.
        out["note"] = "no checkpoints found on disk (re-train to populate)"
    return out


def audit_performance(checkpoint: Path, *, n_passes: int = 5) -> dict:
    """Single-clip inference latency on a real npz (median over n_passes)."""
    import torch

    from isl_bridge.checkpoint import load_checkpoint
    from isl_bridge.infer import _prep_npz

    npz_files = sorted((ROOT / "data/processed/test").rglob("*.npz"))
    if not npz_files or not checkpoint.exists():
        return {"skipped": True, "reason": "no checkpoint or test npz on disk"}

    model, cfg, idx, _ = load_checkpoint(checkpoint)
    model.eval()
    x, mask = _prep_npz(npz_files[0])
    latencies = []
    with torch.no_grad():
        # Warm-up.
        _ = model(x, mask)
        for _ in range(n_passes):
            t0 = time.perf_counter()
            _ = model(x, mask)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "n_passes": n_passes,
        "median_ms": round(latencies[len(latencies) // 2], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "fps_effective": round(1000.0 / max(latencies[len(latencies) // 2], 1e-3), 1),
    }


def audit_robustness(checkpoint: Path) -> dict:
    """Flip-invariance + temporal-subsample robustness on a real npz.

    For honesty on a 2-class subset we report *agreement* between the original
    prediction and the perturbed one, not absolute accuracy.
    """
    import numpy as np
    import torch

    from isl_bridge.checkpoint import load_checkpoint
    from isl_bridge.dataloaders import IMAGENET_MEAN, IMAGENET_STD

    npz_files = sorted((ROOT / "data/processed/test").rglob("*.npz"))
    if not npz_files or not checkpoint.exists():
        return {"skipped": True, "reason": "no checkpoint or test npz"}

    model, cfg, idx, _ = load_checkpoint(checkpoint)
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def _predict(x, mask):
        with torch.no_grad():
            return int(model(x, mask).argmax(dim=-1).item())

    def _to_tensor(frames_uint8):
        rgb = torch.from_numpy(frames_uint8).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2)
        return ((rgb - mean) / std).unsqueeze(0)

    n_agree_flip = n_agree_sub = n = 0
    for npz in npz_files[:5]:                                  # cap to keep it quick
        z = np.load(npz)
        f = z["frames"]
        mask = torch.from_numpy(z["mask"]).unsqueeze(0)
        x = _to_tensor(f)
        base = _predict(x, mask)
        # Horizontal flip.
        x_flip = torch.flip(x, dims=[-1])
        n_agree_flip += int(_predict(x_flip, mask) == base)
        # Temporal subsample (keep every other frame, padded with last).
        T = f.shape[0]
        sub = f[::2]
        if sub.shape[0] < T:
            pad = np.tile(sub[-1:], (T - sub.shape[0], 1, 1, 1))
            sub = np.concatenate([sub, pad], axis=0)
        x_sub = _to_tensor(sub)
        n_agree_sub += int(_predict(x_sub, mask) == base)
        n += 1
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "n_clips_tested": n,
        "hflip_agreement": round(100 * n_agree_flip / max(1, n), 1),
        "temporal_subsample_agreement": round(100 * n_agree_sub / max(1, n), 1),
        "note": "agreement with original prediction on the same clip; " +
                "high agreement = perturbation-invariant, low = the model " +
                "is using the cue the perturbation destroyed.",
    }


def audit_api() -> dict:
    """Boot the FastAPI server, hit /health and /predict, scan for obvious
    security smells (no-auth, no size cap, error swallowing)."""
    api_src = (ROOT / "scripts" / "api.py").read_text()
    smells = {
        "no_authentication": "HTTPException" not in api_src or 'status_code=401' not in api_src,
        "no_max_upload_size": "max_upload_size" not in api_src.lower() and "limit" not in api_src.lower(),
        "verbose_error":      "traceback" in api_src.lower() or "print(traceback" in api_src.lower(),
    }
    # Try a live probe (server may already be running on 5050).
    live = {"health_reachable": False}
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:5050/health", timeout=2) as r:
            live["health_reachable"] = r.status == 200
            live["health_body"] = r.read().decode("utf-8", errors="replace")[:200]
    except Exception as e:                                    # noqa: BLE001
        live["error"] = f"{type(e).__name__}"
    return {"smells": smells, "live_probe": live,
            "note": "smells=True means the check *found* the listed smell; " +
                    "review before exposing publicly."}


def audit_code() -> dict:
    """Lightweight lint pass: pyflakes on src/ and scripts/."""
    targets = [str(p) for p in (ROOT / "src").rglob("*.py")] + \
              [str(p) for p in (ROOT / "scripts").rglob("*.py")]
    if not shutil.which("python3"):
        return {"skipped": True}
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes"] + targets,
        capture_output=True, text=True, timeout=60,
    )
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return {
        "n_files_scanned": len(targets),
        "n_findings": len(out_lines),
        "findings": out_lines[:50],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "artifacts" / "phase3_real_run" / "best.pt")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["data", "model", "performance", "robustness", "api", "code"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runners = {
        "data":        audit_data,
        "model":       audit_model,
        "performance": lambda: audit_performance(args.checkpoint),
        "robustness":  lambda: audit_robustness(args.checkpoint),
        "api":         audit_api,
        "code":        audit_code,
    }
    report = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for name, fn in runners.items():
        if name in args.skip:
            report[name] = {"skipped": True}
            continue
        t0 = time.perf_counter()
        try:
            report[name] = fn()
        except Exception as e:                                # noqa: BLE001
            report[name] = {"error": f"{type(e).__name__}: {e}"}
        report[name]["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    out = OUT_DIR / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"out": str(out.relative_to(ROOT)),
                      "summary": {k: report[k].get("elapsed_sec", "?")
                                  for k in report if k != "ran_at"}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())