#!/bin/bash
# Run the coding-task benchmark against the local dflash-serve.
#
# Prerequisites: ./install.sh has been run and dflash-serve is up on
# port 8000 (or set DFLASH_PORT). Model defaults to whatever TARGET_MODEL
# was at install time (Qwen/Qwen3.5-35B-A3B by default).
#
# Usage:
#   scripts/bench.sh                   # all tests (L1-L8)
#   scripts/bench.sh L1,L4,L8          # specific tests
#   scripts/bench.sh L7                # just the agentic test
#   MODEL=Qwen/Qwen3.5-9B scripts/bench.sh
#   NO_THINK=false scripts/bench.sh    # leave thinking mode on (for comparison)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DFLASH_PORT="${DFLASH_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
NO_THINK="${NO_THINK:-true}"
BASE_URL="http://localhost:${DFLASH_PORT}/v1"
TESTS="${1:-all}"

LOG="[bench]"
say()  { printf "%s %s\n" "$LOG" "$*"; }
die() {
  printf "\n%s ERROR: %s\n" "$LOG" "$1" >&2
  if [[ -n "${2-}" ]]; then
    printf "%s   Fix: %s\n" "$LOG" "$2" >&2
  fi
  exit 1
}

cd "$PROJECT_DIR"

# 1. Server reachable?
if ! curl -sf -m 3 "${BASE_URL}/models" > /dev/null 2>&1; then
  die "dflash-serve not responding on ${BASE_URL}/models." \
      "Start it: cd $PROJECT_DIR && DFLASH_MAX_CTX=262144 nohup uv run dflash-serve --model $MODEL --port $DFLASH_PORT > /tmp/dflash-serve.log 2>&1 & echo \$! > /tmp/dflash-serve.pid"
fi

# 2. Server is serving the requested model?
SERVED=$(curl -s -m 3 "${BASE_URL}/models" | uv run python -c \
  "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
if [[ "$SERVED" != "$MODEL" ]]; then
  die "Server is serving '$SERVED', not '$MODEL'." \
      "Restart dflash-serve with --model $MODEL, or pass MODEL=$SERVED to this script."
fi
say "Server: ${BASE_URL}  model: ${MODEL}  thinking: $([[ $NO_THINK == true ]] && echo OFF || echo ON)"

# 3. Run.
exec uv run python "$PROJECT_DIR/scripts/coding_tests.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --no-think "$NO_THINK" \
  --tests "$TESTS"
