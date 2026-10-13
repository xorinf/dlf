#!/usr/bin/env bash
# run.sh — boot the whole ISL-Bridge demo: FastAPI + static web server.
#
# What runs:
#   * FastAPI on http://127.0.0.1:5050  (scripts/api.py)  -- the model server
#   * Static server on http://127.0.0.1:8000 (web/)         -- the demo UI
#   * Logs streamed to .hermes/run.log and the terminal
#
# Webcam input goes through the browser: the UI in web/index.html calls
# navigator.mediaDevices.getUserMedia(), records a short clip, and uploads it
# to /predict. No extra process needed for webcam. If you'd rather use the
# terminal-only OpenCV loop, run scripts/webcam_demo.py directly.
#
# Overrides via environment:
#   PYTHON=...   absolute path to a Python that has numpy, torch, fastapi,
#                uvicorn, cv2 installed. Auto-detected if unset.
#   CHECKPOINT=artifacts/phase4_real_run/best.pt
#   HOST=127.0.0.1   API_HOST=127.0.0.1   WEB_HOST=127.0.0.1
#   PORT=5050        WEB_PORT=8000
#
# Stop with Ctrl-C — both children get SIGTERM.

set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-artifacts/phase4_real_run/best.pt}"
API_HOST="${API_HOST:-${HOST:-127.0.0.1}}"
WEB_HOST="${WEB_HOST:-${HOST:-127.0.0.1}}"
API_PORT="${PORT:-5050}"
WEB_PORT="${WEB_PORT:-8000}"

LOG_DIR="$ROOT/.hermes"
LOG_FILE="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

# ---- helpers --------------------------------------------------------------
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }

# `tee` style logging — every line goes to both stdout and the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

PIDS=()
cleanup() {
  trap '' INT TERM EXIT
  note ""
  note "shutting down (pid ${PIDS[*]:-})..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  ok "stopped."
}
trap cleanup INT TERM EXIT

# ---- preflight -------------------------------------------------------------
note "ISL-Bridge boot"
echo "  cwd         : $ROOT"
echo "  checkpoint  : $CHECKPOINT"
echo "  api         : http://${API_HOST}:${API_PORT}"
echo "  web         : http://${WEB_HOST}:${WEB_PORT}"
echo "  log         : $LOG_FILE"
echo ""

[[ -f "$CHECKPOINT" ]] || die "checkpoint not found: $CHECKPOINT (set CHECKPOINT=...)"
[[ -d web ]]            || die "web/ missing -- run from the repo root"
[[ -d src/isl_bridge ]] || die "src/isl_bridge missing -- run from the repo root"

# Python resolver: env PYTHON wins, else try a few known-good paths.
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in \
      /opt/anaconda3/bin/python3 \
      "$HOME/anaconda3/bin/python3" \
      /usr/local/bin/python3 \
      /opt/homebrew/bin/python3 \
      python3
  do
    if [[ -x "$cand" ]] || command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c "import numpy, torch, fastapi, uvicorn, cv2" >/dev/null 2>&1; then
        PY="$cand"; break
      fi
    fi
  done
fi
[[ -n "$PY" ]] || die "no Python found with numpy, torch, fastapi, uvicorn, cv2. Set PYTHON=/path/to/python"
ok "python       : $PY ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"

# Print versions so failures later are easy to read.
"$PY" - <<'PY' || die "Python dep check failed"
import numpy, torch, fastapi, uvicorn, cv2
print(f"  numpy        : {numpy.__version__}")
print(f"  torch        : {torch.__version__}")
print(f"  fastapi      : {fastapi.__version__}")
print(f"  uvicorn      : {uvicorn.__version__}")
print(f"  opencv       : {cv2.__version__}")
PY

# ---- port sanity -----------------------------------------------------------
port_in_use() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | tail -n +2 | grep -q .
}
if port_in_use "$API_PORT"; then
  die "port $API_PORT is already in use. Kill the holder or set PORT=... differently."
fi
if port_in_use "$WEB_PORT"; then
  warn "port $WEB_PORT is already in use. The static server may fail; set WEB_PORT=... to override."
fi

# ---- launch API ------------------------------------------------------------
note "starting API..."
(
  cd "$ROOT"
  PYTHONPATH=src "$PY" scripts/api.py \
    --checkpoint "$CHECKPOINT" \
    --host "$API_HOST" \
    --port "$API_PORT"
) &
API_PID=$!
PIDS+=("$API_PID")
ok "api pid      : $API_PID"

# ---- wait for API to come up -----------------------------------------------
note "waiting for API health..."
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    die "API process died before coming up. Tail of $LOG_FILE:\n$(tail -40 "$LOG_FILE")"
  fi
  if curl -sf "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 0.5
done
[[ "$ready" -eq 1 ]] || die "API did not respond on ${API_HOST}:${API_PORT} within 30s. Tail of $LOG_FILE:\n$(tail -40 "$LOG_FILE")"
ok "api health   : ok"

# ---- launch static server --------------------------------------------------
note "starting web server..."
(
  cd "$ROOT/web"
  "$PY" -m http.server "$WEB_PORT" --bind "$WEB_HOST"
) &
WEB_PID=$!
PIDS+=("$WEB_PID")
ok "web pid      : $WEB_PID"

sleep 0.5
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  die "web server died on startup. Tail of $LOG_FILE:\n$(tail -40 "$LOG_FILE")"
fi
if curl -sf "http://${WEB_HOST}:${WEB_PORT}/" >/dev/null 2>&1; then
  ok "web root     : ok"
else
  warn "web root check failed -- port may still be binding. Open it manually."
fi

# ---- summary ---------------------------------------------------------------
cat <<EOF

$(ok "ready.")

  Webcam + upload UI :  http://${WEB_HOST}:${WEB_PORT}/
  API health          :  http://${API_HOST}:${API_PORT}/health
  API docs (Swagger)  :  http://${API_HOST}:${API_PORT}/docs

  The browser will ask for camera access on first use. Click "Allow".
  Logs stream above and to $LOG_FILE.

EOF

wait "$API_PID" || warn "API exited non-zero."
cleanup