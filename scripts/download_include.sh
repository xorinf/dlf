#!/usr/bin/env bash
# Download the INCLUDE dataset (Sridhar et al., ACM MM 2020) from Zenodo.
#
# Default mode: INCLUDE-50 — small subset (~958 videos) for fast iteration.
# --full mode  : the complete 4,292-video, 263-class dataset (~57 GB).
#
# The official train/test CSVs are NOT in this Zenodo record; we fetch them
# separately from the Google Drive link the authors list on the dataset page.
# Pass --no-splits to skip that step if you only want the videos.
#
# Notes:
#  - These URLs intentionally use the canonical Zenodo /records/<id>/files/... path
#    (verified against https://zenodo.org/records/4010759 on download).
#  - Internet drops will leave the .zip file behind; the loop is idempotent — a
#    partial .zip is reused on resume by wget -c.
set -euo pipefail

ZENODO_RECORD="4010759"
BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"
TARGET_DIR="${INCLUDE_DIR:-$PWD/data/raw/include}"
MODE="include50"
DOWNLOAD_SPLITS=1
SPLITS_DRIVE_URL="https://drive.google.com/file/d/1tGjOR9xbk279ZUYnmk3BE-ktcWBdS24w/view?usp=sharing"
SPLITS_OUT="Train_Test_Split.zip"  # filename the resulting archive is saved as

usage() {
    cat <<EOF
Usage: $0 [--full] [--out DIR] [--no-splits]
  --full         Download every category archive (large disk usage, ~57 GB).
                 Default: download only a curated INCLUDE-50 category bundle.
  --out DIR      Destination directory (default: \$INCLUDE_DIR or ./data/raw/include).
  --no-splits    Skip fetching the official train/test CSVs from Google Drive.
  -h, --help     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) MODE="full"; shift ;;
        --out)  TARGET_DIR="${2:-}"; shift 2 ;;
        --no-splits) DOWNLOAD_SPLITS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

needed=()

# INCLUDE-50 category bundle — keep this list honest, do not expand silently.
# (Sourced from the dataset README on Zenodo.)
case "$MODE" in
    include50)
        needed+=(
            "Greetings_1of2.zip"  "Greetings_2of2.zip"
            "Pronouns_1of2.zip"  "Pronouns_2of2.zip"
            "Colours_1of2.zip"   "Colours_2of2.zip"
            "Animals_1of2.zip"   "Animals_2of2.zip"
            "Days_and_Time_1of3.zip" "Days_and_Time_2of3.zip" "Days_and_Time_3of3.zip"
            "Adjectives_1of8.zip"
            "Seasons_1of1.zip"
        )
        ;;
    full)
        # Mirror every category archive (one wget per part). Order is irrelevant;
        # content is namespaced by category folder after extraction.
        declare -a cats=(
            "Adjectives:8"
            "Animals:2"
            "Clothes:2"
            "Colours:2"
            "Days_and_Time:3"
            "Electronics:2"
            "Greetings:2"
            "Home:4"
            "Jobs:2"
            "Means_of_Transportation:2"
            "People:5"
            "Places:4"
            "Pronouns:2"
            "Seasons:1"
            "Society:3"
        )
        for c in "${cats[@]}"; do
            name="${c%:*}"; n="${c#*:}"
            for ((i=1; i<=n; i++)); do
                needed+=("${name}_${i}of${n}.zip")
            done
        done
        ;;
esac

echo ">> Mode: $MODE"
echo ">> Target: $TARGET_DIR"
echo ">> ${#needed[@]} archive(s) planned."
echo

for f in "${needed[@]}"; do
    if [[ -s "$f" && -f "$f" ]]; then
        echo "SKIP $f (already present)"
        continue
    fi
    echo "GET  $f"
    wget -c --tries=5 --retry-connrefused -q --show-progress "$BASE/$f?download=1" -O "$f"
done

if [[ "$DOWNLOAD_SPLITS" == "1" ]]; then
    if [[ -s "$SPLITS_OUT" ]]; then
        echo "SKIP $SPLITS_OUT (already present)"
    else
        # Try gdown first (handles Drive confirm tokens properly). If unavailable
        # we print the URL so the user can fetch it once manually.
        echo "GET  Train/Test CSVs from Google Drive"
        if command -v gdown >/dev/null 2>&1; then
            gdown --fuzzy "$SPLITS_DRIVE_URL" -O "$SPLITS_OUT"
        elif python3 -c "import gdown" 2>/dev/null; then
            python3 -m gdown --fuzzy "$SPLITS_DRIVE_URL" -O "$SPLITS_OUT"
        else
            echo "NOTE: gdown not installed. Fetch this manually and place as $SPLITS_OUT:"
            echo "      $SPLITS_DRIVE_URL"
            echo "      pip install gdown ; gdown --fuzzy '${SPLITS_DRIVE_URL}' -O '$SPLITS_OUT'"
        fi
    fi
fi

echo
echo ">> Done. To extract, run:"
echo "   unzip -n '$TARGET_DIR'/*.zip -d '$TARGET_DIR'"
