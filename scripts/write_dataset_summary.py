"""Write the Phase-2 dataset summary to disk.

Lives under artifacts/ so the README and report can cite it. Not committed —
the dataset card lives in docs/, but raw numbers live here, regenerated.

Outputs artifacts/dataset_summary.json with:
    - dataset stats (total records, present, missing, byte totals)
    - per-split present counts
    - per-class present counts (matched to INCLUDE-50 label_map)
    - the actual list of unmatched orphan rows (for audit)
    - which archives are on disk (subset plan) so the report can state it
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

# scripts/ sits one level below src/. The repo root is parents[1].
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))

from isl_bridge import config, dataset, dataset_meta  # noqa: E402


def main() -> None:
    inv = dataset.build_inventory(config.RAW_DIR)

    s = dataset.summary(inv)

    cov = dataset.label_coverage(inv)

    # Per-split present counts
    by_split = Counter()
    by_cat = Counter()
    for r in inv:
        if r.exists and r.size_bytes > 0 and dataset.label_for(r.word, dataset_meta.load_label_map("include50")) is not None:
            by_split[r.expected_split or "?"] += 1
            by_cat[r.category] += 1

    # List unmatched orphan rows in their entirety (small set).
    label_map = dataset_meta.load_label_map("include50")
    orphans = [r.relpath for r in inv if dataset.label_for(r.word, label_map) is None]

    archives_on_disk = sorted(p.name for p in config.RAW_DIR.glob("*.zip"))

    out = {
        "splits_pinned_to": "AI4Bharat/INCLUDE master (data/metadata/)",
        "n_records_total": s["n_records"],
        "n_records_present": s["n_present"],
        "n_records_missing": s["n_missing"],
        "bytes_present": s["bytes_total"],
        "per_split_present": dict(by_split),
        "per_category_present": dict(by_cat),
        "n_distinct_classes_present": s["n_distinct_words"],
        "orphan_rows_count": len(orphans),
        "orphan_rows_sample": orphans[:25],
        "archives_on_disk": archives_on_disk,
        "working_subset": "Seasons (1 archive: Seasons_1of1.zip)",
    }
    out_path = ROOT / "artifacts" / "dataset_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "orphan_rows_sample"}, indent=2))


if __name__ == "__main__":
    main()
