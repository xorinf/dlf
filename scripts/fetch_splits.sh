#!/usr/bin/env bash
# Populate data/metadata/ with the AI4Bharat INCLUDE label maps and train/test
# splits, pinned to the upstream commit.
#
# This is the single, idempotent source for split & label truth — no manual
# copy-paste, no Drive dependency.
set -euo pipefail

DEST="${1:-${PWD}/data/metadata}"
PIN="${PIN_COMMIT:-master}"
BASE="https://raw.githubusercontent.com/AI4Bharat/INCLUDE/${PIN}"

mkdir -p "$DEST"

urls=(
    "label_maps/label_map_include.json"
    "label_maps/label_map_include50.json"
    "train_test_paths/include_train.txt"
    "train_test_paths/include_val.txt"
    "train_test_paths/include_test.txt"
    "train_test_paths/include50_train.txt"
    "train_test_paths/include50_val.txt"
    "train_test_paths/include50_test.txt"
)

for rel in "${urls[@]}"; do
    out="$DEST/$(basename "$rel")"
    if [[ -s "$out" ]]; then
        echo "SKIP $out"
        continue
    fi
    echo "GET  $rel"
    curl -fsSL --retry 5 "$BASE/$rel" -o "$out"
done

echo
echo "Files in $DEST:"
ls -la "$DEST"
echo
echo "Pinned to commit: $PIN"
