#!/usr/bin/env bash
# Download the INCLUDE dataset (Sridhar et al., ACM MM 2020) from Zenodo.
#
# Modes:
#   default      : INCLUDE-50 — full bundle (~46.7 GB compressed, ~93 GB disk).
#                  Refuses to start when target disk is short.
#   --subset     : smallest-archive subset for limited-disk envs (~5 GB compressed).
#   --full       : mirror the entire dataset (~52.8 GB compressed, see docs/phase2.md).
#   --categories "A B C"
#                : explicit category list (subset of "Adjectives Animals Clothes
#                  Colours Days_and_Time Electronics Greetings Home Jobs
#                  Means_of_Transportation People Places Pronouns Seasons Society").
#
# In all modes the per-archive MD5 is checked after download. The download is
# resumable (curl -C -) and survives transient HTTP errors (curl --retry).
#
# Splits & label maps ship with this repo under data/metadata/ and are NOT
# fetched here — see scripts/fetch_splits.sh. (Zenodo 4010759 does not contain
# a Train_Test_Split.zip; using the AI4Bharat/INCLUDE GitHub copies is the
# canonical, version-controlled path.)
set -euo pipefail

ZENODO_RECORD="4010759"
BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"
RAW_DIR="${INCLUDE_DIR:-$PWD/data/raw/include}"
RAW_DIR="$(mkdir -p "$RAW_DIR" && cd "$RAW_DIR" && pwd -P)"   # resolve to absolute early
MODE="include50"
EXPLICIT_CATEGORIES=""

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    echo
    cat <<EOF

Usage: $0 [--subset | --full | --categories "A B C"] [--out DIR]

Defaults: --include50 (the full INCLUDE-50 category bundle).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)  MODE="subset"; shift ;;
        --include50) MODE="include50"; shift ;;
        --full)    MODE="full"; shift ;;
        --categories) MODE="categories"; EXPLICIT_CATEGORIES="${2:-}"; shift 2 ;;
        --out)     RAW_DIR="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"
# Authoritative archive list (name, category, MD5, bytes) is fetched live from
# Zenodo's JSON API rather than hardcoded — keeps the script honest if Zenodo
# rotates or adds parts.
ARCHIVE_INDEX="$(mktemp -t include_index.XXXXXX.json)"
trap 'rm -f "$ARCHIVE_INDEX"' EXIT
curl -fsSL "https://zenodo.org/api/records/${ZENODO_RECORD}" -o "$ARCHIVE_INDEX"

python3 - "$ARCHIVE_INDEX" "$MODE" "$EXPLICIT_CATEGORIES" > archives.list <<'PY'
import json, sys, os, re
idx_path, mode, explicit = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(idx_path))
# Build {category: [archive_entries]}
by_cat = {}
for f in data["files"]:
    name = f["key"]
    if not name.endswith(".zip"): continue
    # Filename pattern: "<Cat>_<i>of<n>.zip"  where Cat has underscores (e.g. Days_and_Time_1of3.zip)
    m = re.match(r"^(.+)_(\d+)of(\d+)\.zip$", name)
    if not m: continue
    cat, i, n = m.group(1), int(m.group(2)), int(m.group(3))
    md5 = f.get("checksum","").split(":")[-1]
    by_cat.setdefault(cat, []).append((name, int(f["size"]), md5))

# Authoritative order for full sets. "subset" picks the smallest categories.
FULL_ORDER = [
    "Adjectives","Animals","Clothes","Colours","Days_and_Time",
    "Electronics","Greetings","Home","Jobs",
    "Means_of_Transportation","People","Places","Pronouns",
    "Seasons","Society",
]
# Subset = smallest archives first until disk-plausible.
SUBSET_CATS = ["Seasons", "Electronics", "Pronouns"]  # ~5 GB compressed, ~10 GB disk

def emit(names_and_meta):
    for name, size, md5 in sorted(names_and_meta):
        print(f"{name}\t{size}\t{md5}")

if mode == "full":
    for c in FULL_ORDER:
        emit(by_cat.get(c, []))
elif mode == "include50":
    for c in FULL_ORDER:
        emit(by_cat.get(c, []))
elif mode == "subset":
    for c in SUBSET_CATS:
        emit(by_cat.get(c, []))
elif mode == "categories":
    pick = set(explicit.split())
    for c in FULL_ORDER:
        if c in pick:
            emit(by_cat.get(c, []))
else:
    sys.stderr.write(f"unknown mode: {mode}\n"); sys.exit(2)
PY

needed_files=()
needed_bytes=0
while IFS=$'\t' read -r fname fsize fmd5; do
    [[ -z "$fname" ]] && continue
    needed_files+=("$fname")
    needed_bytes=$((needed_bytes + fsize))
done < archives.list

human_size() {  # bytes -> "X.X GB" / "X.X MB" — BSD awk safe on macOS.
    local bytes="$1"
    if (( bytes >= 1073741824 )); then
        awk -v b="$bytes" 'BEGIN{printf "%.2f GB", b/1073741824}'
    else
        awk -v b="$bytes" 'BEGIN{printf "%.0f MB", b/1048576}'
    fi
}

echo "Mode: $MODE"
echo "Raw dir: $RAW_DIR"
echo "Archives planned: ${#needed_files[@]} (compressed bytes: $(human_size "$needed_bytes"))"
echo

# Disk preflight (zips + extracted ≈ 2× compressed).
NEED=$((needed_bytes * 2))
AVAIL_KB=$(df -k "$RAW_DIR" | tail -1 | awk '{print $4}')
AVAIL=$((AVAIL_KB * 1024))
if (( AVAIL < NEED )); then
    echo "ERROR: not enough free space." >&2
    echo "  need=$(human_size "$NEED")" >&2
    echo "  have=$(human_size "$AVAIL")" >&2
    echo "  free up disk, or re-run with --categories 'Seasons' for the smallest possible subset." >&2
    exit 3
fi

fetch_one() {
    local fname="$1" fsize="$2" fmd5="$3"
    local final="$RAW_DIR/$fname"
    local part="$RAW_DIR/$fname.part"

    if [[ -s "$final" ]]; then
        local got; got=$(md5 -q "$final" 2>/dev/null || md5sum "$final" | awk '{print $1}')
        if [[ "$got" == "$fmd5" ]]; then
            echo "OK    $fname (md5 match)"
            return 0
        else
            echo "RETRY $fname (md5 mismatch: got $got != $fmd5)"
            rm -f "$final"
        fi
    fi

    echo "GET   $fname  ($(human_size "$fsize"))"
    # Resumable, quiet-but-with-progress, broad retry envelope.
    curl -C - --retry 20 --retry-delay 4 --retry-all-errors --connect-timeout 30 \
        --max-time 7200 -sSL --show-error \
        "$BASE/$fname?download=1" -o "$part"
    mv "$part" "$final"

    local got; got=$(md5 -q "$final" 2>/dev/null || md5sum "$final" | awk '{print $1}')
    if [[ "$got" != "$fmd5" ]]; then
        echo "FAIL  $fname md5 mismatch (got $got expected $fmd5)" >&2
        return 1
    fi
    echo "OK    $fname"
}

# Run fetches serially — concurrent downloads aren't worth the bandwidth thrash
# against Zenodo's rate limiting on a single connection.
fail=0
while IFS=$'\t' read -r fname fsize fmd5; do
    [[ -z "$fname" ]] && continue
    fetch_one "$fname" "$fsize" "$fmd5" || fail=$((fail + 1))
done < archives.list

if (( fail > 0 )); then
    echo "" >&2
    echo "$fail archive(s) failed. Re-run to resume; partial files kept as .part." >&2
    exit 4
fi

echo
echo "All ${#needed_files[@]} archive(s) downloaded and integrity-verified."
echo "Next: bash scripts/extract_include.sh        # unzip into $RAW_DIR/<Category>/<word>/"
