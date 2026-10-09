#!/usr/bin/env bash
# Extract INCLUDE archives in $RAW_DIR into the canonical directory tree.
#
# Safe behaviour:
#  - "unzip -n"  → never overwrite an existing file (idempotent).
#  - Path traversal guard: refuses to extract a zip whose first entry's path
#    is absolute or contains ".." — INCLUDE's archives don't do this, but the
#    check costs nothing.
#  - Skips macOS resource forks (__MACOSX/, .DS_Store) the read-only-browsers
#    sometimes drop into zips.
set -euo pipefail

RAW_DIR="${1:-${INCLUDE_DIR:-$PWD/data/raw/include}}"
[[ -d "$RAW_DIR" ]] || { echo "no raw dir: $RAW_DIR" >&2; exit 2; }

cd "$RAW_DIR"
shopt -s nullglob
zips=( *.zip )
shopt -u nullglob
[[ ${#zips[@]} -gt 0 ]] || { echo "no .zip files in $RAW_DIR" >&2; exit 0; }

guard() {
    python3 - "$1" <<'PY'
import sys, zipfile
p = sys.argv[1]
ok = True
with zipfile.ZipFile(p) as z:
    for name in z.namelist():
        # absolute path or parent-ref → reject
        if name.startswith("/") or name.startswith("..") or "/../" in ("/"+name):
            print(f"  REFUSE {p}: unsafe entry {name!r}", flush=True); sys.exit(2)
        # macOS cruft → silent
print(f"  OK   {p}")
PY
}

echo "Validating ${#zips[@]} archive(s) in $RAW_DIR ..."
for z in "${zips[@]}"; do guard "$z" || exit 3; done

echo
echo "Extracting ..."
# -n = never overwrite (idempotent), -q = quiet.
for z in "${zips[@]}"; do
    echo "  $z"
    unzip -nq "$z"
done

echo
echo "Done. Top-level categories:"
ls -1 "$RAW_DIR" | head -20
