# Audit — `scripts/audit.py`

A single-command audit covering dataset integrity, model bookkeeping,
latency, robustness, API safety, and a lightweight code-lint pass.

## How to run

```bash
PYTHONPATH=src python3 scripts/audit.py
# → artifacts/audit/audit_report.json
```

You can skip any section:

```bash
PYTHONPATH=src python3 scripts/audit.py --skip code
```

## What it checks

| Section       | What                                                                                                |
|---------------|-----------------------------------------------------------------------------------------------------|
| `data`        | INCLUDE inventory, zero-byte / missing / orphan-label counts, % missing, total bytes on disk.       |
| `model`       | Every `best.pt` under `artifacts/` — size on disk, param count, FP32 size, backbone, class count.    |
| `performance` | Single-clip inference median + p95 latency on a real `n_passes` forward passes.                      |
| `robustness`  | Horizontal-flip agreement + temporal-subsample agreement vs. the unperturbed prediction.            |
| `api`         | Scans `scripts/api.py` for known smells (no-auth, no upload-size cap, verbose-error handlers) + live `/health` reachability. |
| `code`        | pyflakes on `src/` and `scripts/`.                                                                  |

## Last run (this machine)

The numbers in `artifacts/audit/audit_report.json` come from the most
recent run on this machine:

```
data        → 22 / 958 clips on disk (97.7% missing), 15 orphan-label rows.
model       → 2 checkpoints (Phase 3 + Phase 4), 8.7 MB / 10.8 MB on disk.
performance → median 1199 ms / clip, p95 1256 ms, ≈0.8 fps on CPU.
robustness  → 75% hflip agreement, 100% temporal-subsample agreement (4 test clips).
api         → 3 smells surfaced (no auth, no upload-size cap); no live server at probe time.
code        → 13 pyflakes findings (unused imports/vars), not blocking.
```

## Reading the report

* The findings are *honest reports*, not silent green checks. A smell = "in"
  means the audit *found* the listed smell; review before exposing
  publicly.
* `robustness` uses *agreement* (not accuracy) on purpose — on a 4-clip
  test set accuracy is noise.
* `data.missing_pct` is high because we only have the first archive of
  INCLUDE-50 on disk. The audit doesn't tell you to download more; it
  tells you the truth about what's there.

## When to re-run

* After data download / extract — `data` section will move.
* After a training run — `model` section grows; `performance` and
  `robustness` change.
* Before a PR — `code` section should be clean.

## What it doesn't check (intentional)

* Visual / perceptual quality of model output — that's a manual review.
* Security beyond the listed smells — a proper penetration test belongs to
  the deployment phase, not a Python smoke.