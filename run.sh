#!/usr/bin/env bash
# run.sh — boot the whole ISL-Bridge demo with one command.
#
# Starts two processes and tails their combined logs to the terminal
# (and to .hermes/run.log):
#
#   * FastAPI on http://127.0.0.1:5050  (scripts/api.py)  -- the model server
#   * Static web on http://127.0.0.1:8000 (web/)            -- the demo UI
#
# Webcam input goes through the browser: the UI in web/index.html uses
# navigator.mediaDevices.getUserMedia() and posts the rolling frame window
# to /predict_frames every 2.5s. MediaPipe Pose + HandLandmarker run
# client-side to draw a person/hand box and place the live label on it.
#
# On launch, this script:
#   1. Resolves a working Python (one with numpy, torch, fastapi, uvicorn, cv2)
#   2. Sanity-checks the checkpoint + class labels on disk
#   3. Boots the API, waits for /health, then the web server, waits for /
#   4. Prints the URLs and tries to open the demo in your default browser
#
# Stop with Ctrl-C — both children get SIGTERM, then SIGKILL if needed.
#
# Overrides via environment:
#   PYTHON=...        absolute path to a Python that has numpy, torch, fastapi,
#                     uvicorn, cv2. Auto-detected if unset.
#   CHECKPOINT=artifacts/phase4_real_run/best.pt
#   HOST=127.0.0.1    API_HOST=...    WEB_HOST=...
#   PORT=5050         WEB_PORT=...
#   NO_OPEN=1         don't try to open the browser (CI / headless)
#   QUIET=1           don't tail child stdout (just status)
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-artifacts/phase4_real_run/best.pt}"
API_HOST="${API_HOST:-${HOST:-127.0.0.1}}"
WEB_HOST="${WEB_HOST:-${HOST:-127.0.0.1}}"
API_PORT="${PORT:-5050}"
WEB_PORT="${WEB_PORT:-8000}"
NO_OPEN="${NO_OPEN:-0}"

LOG_DIR="$ROOT/.hermes"
LOG_FILE="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

# ---- helpers --------------------------------------------------------------
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }

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

# ---- preflight ------------------------------------------------------------
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

# Smoke-test the checkpoint + class map before booting the long-lived API.
# Fail loud here, not after the API is half up.
"$PY" - <<PY || die "checkpoint load failed"
import sys
sys.path.insert(0, "src")
import torch
from pathlib import Path
from isl_bridge.checkpoint import load_checkpoint
m, cfg, idx_to_class, _ = load_checkpoint(Path("$CHECKPOINT"))
print(f"  model        : {m.__class__.__name__}")
print(f"  num_classes  : {len(idx_to_class)}")
print(f"  classes      : {sorted(idx_to_class.values())}")
PY
ok "checkpoint   : loaded"

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
) > "$LOG_DIR/api.out" 2>&1 &
API_PID=$!
PIDS+=("$API_PID")
ok "api pid      : $API_PID"

# Tail child output from the start so any startup error is visible even if
# the wait-loop later fails.
tail -n +1 -F "$LOG_DIR/api.out" > >(tee -a "$LOG_FILE") 2>&1 &
TAIL_PID=$!
PIDS+=("$TAIL_PID")

# ---- wait for API to come up -----------------------------------------------
note "waiting for API health..."
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    die "API process died before coming up. Tail of api.out:\n$(tail -40 "$LOG_DIR/api.out")"
  fi
  if curl -sf "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 0.5
done
[[ "$ready" -eq 1 ]] || die "API did not respond on ${API_HOST}:${API_PORT} within 30s. Tail of api.out:\n$(tail -40 "$LOG_DIR/api.out")"
ok "api health   : ok"

# ---- launch static server --------------------------------------------------
note "starting web server..."
(
  cd "$ROOT/web"
  "$PY" -m http.server "$WEB_PORT" --bind "$WEB_HOST"
) > "$LOG_DIR/web.out" 2>&1 &
WEB_PID=$!
PIDS+=("$WEB_PID")
ok "web pid      : $WEB_PID"

# Fold the web server's output into the same tail stream so the user sees
# both sets of logs interleaved.
tail -n +1 -F "$LOG_DIR/web.out" > >(tee -a "$LOG_FILE") 2>&1 &
WEB_TAIL_PID=$!
PIDS+=("$WEB_TAIL_PID")

sleep 0.5
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  die "web server died on startup. Tail of web.out:\n$(tail -40 "$LOG_DIR/web.out")"
fi
if curl -sf "http://${WEB_HOST}:${WEB_PORT}/" >/dev/null 2>&1; then
  ok "web root     : ok"
else
  warn "web root check failed -- port may still be binding. Open it manually."
fi

# ---- open browser ---------------------------------------------------------
WEB_URL="http://${WEB_HOST}:${WEB_PORT}/"
if [[ "$NO_OPEN" != "1" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$WEB_URL" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$WEB_URL" >/dev/null 2>&1 || true
  fi
fi

# ---- summary ---------------------------------------------------------------
cat <<EOF

$(ok "ready.")

  Demo UI         :  $WEB_URL
  API health      :  http://${API_HOST}:${API_PORT}/health
  API docs        :  http://${API_HOST}:${API_PORT}/docs

  Browser will ask for camera access on first use. Click "Allow".
  Logs: tail -f $LOG_FILE  (or api.out / web.out under $LOG_DIR/)
  Stop: Ctrl-C.

EOF

wait "$API_PID"
api_rc=$?
[[ "$api_rc" -ne 0 ]] && warn "API exited with code $api_rc."
exit $api_rc
# cleanup() runs via the EXIT trap set up at the top.
