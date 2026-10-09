"""Project-wide paths, derived centrally so nothing else hardcodes them.

All env overrides use ISL_BRIDGE_DATA_ROOT. Defaults to ./data so that the
project works out of the box for a developer who clones and runs from CWD.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of `src/`.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get(
        "ISL_BRIDGE_DATA_ROOT",
        str(REPO_ROOT / "data"),
    )
).resolve()

RAW_DIR    = DATA_ROOT / "raw" / "include"
CACHE_DIR  = DATA_ROOT / "cache"
PROC_DIR   = DATA_ROOT / "processed"
META_DIR   = DATA_ROOT / "metadata"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"

for _d in (RAW_DIR, CACHE_DIR, PROC_DIR, META_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
