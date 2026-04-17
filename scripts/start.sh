#!/bin/bash
# Start the dflash-serve background server. Run once; persists until killed or reboot.
# Usage: scripts/start.sh

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${DFLASH_PORT:-8000}"
MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-35B-A3B}"
LOG="/tmp/dflash-serve.log"
PID_FILE="/tmp/dflash-serve.pid"

# Already running?
if curl -sf -m 2 "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  echo "dflash-serve already running on :$PORT"
  exit 0
fi

echo "Starting dflash-serve ($MODEL) on :$PORT..."
DFLASH_MAX_CTX=262144 nohup uv run dflash-serve --model "$MODEL" --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$PID_FILE"

# Wait for it to be ready
for i in $(seq 1 36); do
  sleep 5
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "Ready (PID $(cat "$PID_FILE")). Stop with: kill \$(cat $PID_FILE)"
    exit 0
  fi
done
echo "Failed to start. See $LOG" >&2
exit 1
