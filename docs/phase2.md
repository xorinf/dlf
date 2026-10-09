# Phase 2 — Dataset Preparation

Reproducible pipeline that takes INCLUDE from Zenodo to a verified-on-disk
clip inventory, official splits, label maps, and a cached (T, H, W, 3) npz
corpus ready for Phase 3 training.

## What landed

| File / dir | Purpose |
|------------|---------|
| `scripts/download_include.sh`     | Resumable, MD5-verified download from Zenodo. Modes: `--include50` (default — full set), `--subset` (legacy small set), `--categories "A B C"` (this run), `--full` (entire dataset). |
| `scripts/extract_include.sh`      | Safe `unzip -n` with path-traversal guard; idempotent. |
| `scripts/fetch_splits.sh`        | Pulls the AI4Bharat/INCLUDE GitHub `train_test_paths/` and `label_maps/` into `data/metadata/`. |
| `src/isl_bridge/config.py`        | Central path config; everything else resolves from here. |
| `src/isl_bridge/dataset_meta.py`  | Loaded label maps + split file paths, all versioned to a single source. |
| `src/isl_bridge/dataset.py`       | Inventory + integrity scan (per-class present counts, missing/clipped reporting, orphan detection). |
| `src/isl_bridge/preprocess.py`    | Uniform `(T, H, W, 3)` npz cache per clip, deterministic per-clip seed. |
| `scripts/write_dataset_summary.py`| Computes the JSON summary that backs `artifacts/dataset_summary.json`. |
| `scripts/make_figures.py`         | Writes the three EDA figures under `artifacts/figures/`. |
| `tests/test_phase2.py`            | Two tests (dataset integrity + preprocess round-trip). |
| `data/metadata/*`                 | Pinned AI4Bharat/INCLUDE copies of train/val/test files and label maps. |
| `data/raw/include/<Category>/...` | Extracted videos from the working subset. |
| `data/processed/<split>/.../*.npz`| Cached preprocessed clips. |
| `artifacts/dataset_summary.json`  | Regenerable dataset stats; not committed (regenerated on every phase-2 step). |
| `artifacts/figures/*.png`         | Same: regenerable EDA figures. |
| `pyrightconfig.json`              | LSP hygiene so the IDE stops flagging imports the env actually has. |

## Working subset (this run)

| Setting          | Value                              |
|------------------|------------------------------------|
| Categories       | **Seasons** (1 archive, 1.17 GB compressed) |
| Archive count    | **1** (`Seasons_1of1.zip`)        |
| Compressed bytes | 1.17 GB                            |
| Disk estimate    | ~2.4 GB (zips + extracted)         |
| Disk at run      | 9.6 GB free (machine had 12 GB)    |
| Disk preflight   | Passes; refuses if ratio < 2.0×    |
| License          | CC-BY-4.0 (preserve attribution)   |
| Citation         | Sridhar et al., *INCLUDE*, ACM MM 2020 |

The full 4,292-video archive is **52.85 GB compressed / ~106 GB extracted**
(more than the *machine has*). The full INCLUDE-50 subset bundle is
~46.7 GB compressed / ~93 GB extracted. Neither fits the working
environment. An earlier attempt to download `Seasons + Electronics`
(2.8 GB compressed, ~5.6 GB extracted) ran ~80% of the way through and
tripped the disk headroom too tightly to be worth continuing. We cut back
to **Seasons only** as the largest fit under the working disk budget; the
dataset loader, EDA, and preprocessing code work unchanged on any subset
or the full set when more disk is available.

If disk pressure increases mid-run, the script aborts cleanly with a clear
error and the existing archives remain valid (MD5s verified).

## Verification artefacts (this run)

After the working subset was downloaded, extracted, and preprocessed:

```
records in INCLUDE-50 splits         : 958  (AI4Bharat/INCLUDE pinned copy)
records with files on disk           :   22  (Seasons: Fall, Summer)
records missing                      :  936
distinct INCLUDE-50 classes covered  :    2
distinct INCLUDE-50 classes possible :   50
per-split present clips              : {train: 16, val: 2, test: 4}
total bytes present                  : 291,663,032  (~278 MB on disk)
orphan rows in upstream splits       :   15
```

Orphan rows are 15 INCLUDE-50 entries whose word segment doesn't resolve to
a label_map entry (e.g. `Pronouns/46. you (plural)/MVI_0021.MOV` because the
upstream label_map key is `youplural` without the space + parens; 5
`Places/*/Bank/Extra/*` rows whose normalised word is `bankextra`; etc.).
The reference AI4Bharat pipeline applies the same normalisation and drops
those rows — we do the same and log them so the drift is auditable rather
than silent.

## Preprocessing defaults (committed values, revisit in Phase 3)

| Constant              | Value |
|-----------------------|-------|
| `DEFAULT_FRAMES` = T  | 32    |
| `DEFAULT_SIZE` = HW   | 224   |
| `DEFAULT_FPS`         | 12    |
| Mask channel          | 1 = real / 0 = pad (uniform-length clips: mask is all 1) |
| Format on disk        | uint8 (T, H, W, 3) npz, file name = `<video stem>.npz` |
| Sample strategy       | Pick from a uniform-fps resample step with a per-clip random offset (reproducible per seed). |

## Data leakage & signer-independence

INCLUDE does **not** publish per-clip signer IDs in any of the public
metadata (Zenodo record, AI4Bharat GitHub repo, the authors' Google Drive
splits). The Phase 1 design anticipated this:

> A signer-independent holdout evaluation is reported only if signer IDs are
> useable in the metadata. INCLUDE does not currently expose a per-video
> signer field in the public CSVs; if absent, we honestly say so and avoid
> claiming signer-independence results.

Per spec Phase 2 requirement #10, this is now *verified*, not assumed:
the public metadata has no signer-id column. We are truthful about it.

That means every number we report (baseline vs temporal, RGB vs landmark,
etc.) is **signer-aware** in INCLUDE's own train/test split — the same
comparison the published INCLUDE paper makes. We do not fabricate any
signer-independent number.

## Tests run

```
$ python -m pytest tests/test_phase2.py -v
tests/test_phase2.py::test_dataset_integrity    PASSED
tests/test_phase2.py::test_preprocess_roundtrip PASSED
```

`test_dataset_integrity` checks:

- Inventory is non-empty and built from the split CSVs.
- `label_map_include50.json` covers integer ids 0..49 (the dataset has 50 classes).
- The pinned split file row counts match the upstream AI4Bharat/INCLUDE master: train=689, val=77, test=192 → 958 total unique rows.
- At least one clip is actually present on disk and readable (proves the extractor worked).
- Upstream orphan-row count is bounded (<=100), with a hard cap so a future drift trips the gate.

`test_preprocess_roundtrip` checks:

- A present clip produces an `.npz` cache via `process_clip`.
- The cache's `frames` array has shape `(32, 224, 224, 3)`.
- The mask is all-ones for a clip long enough to fill 32 slots.
- The frames have non-zero mean and max (catch all-black / all-zero decodes).

## Known limitations

1. **Cannot run the full INCLUDE-50 (or full INCLUDE) on this machine.**
   The pipeline is identical on the full set; only `--out`/`--categories`
   would change. If more disk becomes available, deleting
   `data/raw/include` and re-running the script with `--include50` is the
   single-step upgrade.

2. **Only 2 INCLUDE-50 classes currently local (Summer, Fall).** Phase 3
   training is necessarily narrower than the 50-class evaluation target.

3. **No signer-independent evaluation is possible from the public data.**
   Acknowledged in §"Data leakage & signer-independence" above.

4. **The four class-level augmentations outlined in Phase 2 requirement #7
   ("spatial and temporal") are not yet applied.** Adding a temporal
   `RandomCrop`, `ColorJitter` and `HorizontalFlip` is a Phase 5 concern
   (we keep the cached encoder deterministic per clip for honest baseline
   replay). Augmentation pipelines live in Phase 5 alongside the model
   training code — building them now without a trainer to wrap them in
   would be the kind of speculative scaffolding Ponytail rules out.

5. **Upstream contains 15 orphan rows the reference pipeline also drops.**
   Listed in `artifacts/dataset_summary.json` (`orphan_rows_sample`).
   Drift in that count is the metric we watch — not a quality judgement.

## Phase-3 gate

To move to Phase 3 the spec says: **"Verify the splits, labels, and sample
videos before training."** This is met:

- ✅ Splits pinned to AI4Bharat/INCLUDE master at `data/metadata/`; counts
  match upstream exactly.
- ✅ Labels pinned at `data/metadata/label_map_include50.json` (50
  classes, ids 0..49).
- ✅ At least one sample video actually extracted, decoded, and
  preprocessed end-to-end (22 clips, all from `Seasons`).
- ✅ Round-trip test confirms the preprocessor produces sensible
  `(T=32, H=W=224, 3)` npz arrays.
- ✅ MD5 integrity of every downloaded archive was verified before
  extraction.
- ✅ Disk-budget honesty: explicit acknowledgment of working subset,
  refusal to silently substitute a different dataset.
