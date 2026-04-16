#!/bin/bash
# =============================================================================
# DFlash + OpenCode + Qwen3.5 Setup for Apple Silicon (M-series)
# Installs venv, models, OpenAI-compatible server, and wires opencode.
# See README.md for what this does, prerequisites, and troubleshooting.
# =============================================================================

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration (overridable via env vars)
# -------------------------------------------------------------------------
# Project root = directory of this script. uv manages a .venv here.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFLASH_PORT="${DFLASH_PORT:-8000}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}"
OPENCODE_CONFIG_DIR="$HOME/.config/opencode"
OPENCODE_CONFIG="$OPENCODE_CONFIG_DIR/opencode.json"

# Minimum requirements
MIN_MACOS_MAJOR=14         # Sonoma. Earlier macOS lacks Metal 3 features mlx uses.
MIN_FREE_DISK_GB=30        # ~20 GB models + buffer
MIN_RAM_GB=16              # Qwen3.5-9B bf16 + draft + KV cache need ~22 GB
MIN_MLX_VERSION="0.31.0"   # ml-explore/mlx#3159 (Metal event leak fix)
# Python version is enforced by pyproject.toml (requires-python). uv will
# install a matching Python automatically if the system one is too old.

LOG_PREFIX="[install]"
SERVER_LOG="/tmp/dflash-serve.log"
SERVER_PID_FILE="/tmp/dflash-serve.pid"

# -------------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------------
say()  { printf "%s %s\n" "$LOG_PREFIX" "$*"; }
warn() { printf "%s WARN: %s\n" "$LOG_PREFIX" "$*" >&2; }
die() {
  # die "<short reason>" "<what to do about it>"
  printf "\n%s ERROR: %s\n" "$LOG_PREFIX" "$1" >&2
  if [[ -n "${2-}" ]]; then
    printf "%s   Fix: %s\n" "$LOG_PREFIX" "$2" >&2
  fi
  exit 1
}

# Trap unexpected failures (set -e firing without our die()) so the user
# at least knows where it died.
on_error() {
  local exit_code=$?
  local line=${1:-?}
  printf "\n%s ERROR: command failed (exit %d) at line %s.\n" \
    "$LOG_PREFIX" "$exit_code" "$line" >&2
  printf "%s   This usually means a step above printed an error explaining why.\n" \
    "$LOG_PREFIX" >&2
  printf "%s   Re-run after fixing it; the script is idempotent.\n" \
    "$LOG_PREFIX" >&2
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

# -------------------------------------------------------------------------
# Preflight: verify *all* prerequisites before doing anything destructive.
# -------------------------------------------------------------------------
preflight() {
  say "Preflight checks..."
  local errors=()

  # OS family
  if [[ "$(uname -s)" != "Darwin" ]]; then
    errors+=("Not macOS (uname -s = $(uname -s)). This script is macOS-only.|Run on a Mac, or adapt the brew/hf paths for Linux.")
  fi

  # Apple Silicon
  if [[ "$(uname -m)" != "arm64" ]]; then
    errors+=("Not Apple Silicon (uname -m = $(uname -m)). MLX requires arm64.|Use an M-series Mac. Intel Macs are not supported by MLX/Metal.")
  fi

  # macOS version
  local macos_major
  macos_major=$(sw_vers -productVersion 2>/dev/null | cut -d. -f1 || echo 0)
  if [[ "$macos_major" -lt "$MIN_MACOS_MAJOR" ]]; then
    errors+=("macOS $(sw_vers -productVersion) is too old (need >= $MIN_MACOS_MAJOR).|Upgrade macOS via System Settings > General > Software Update.")
  fi

  # RAM
  local ram_bytes ram_gb
  ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  ram_gb=$(( ram_bytes / 1024 / 1024 / 1024 ))
  if [[ "$ram_gb" -lt "$MIN_RAM_GB" ]]; then
    warn "Only ${ram_gb} GB RAM detected (recommended: >= ${MIN_RAM_GB} GB)."
    warn "  Qwen3.5-9B bf16 + draft model peaks around 22 GB unified memory."
    warn "  Setup will continue; expect heavy swap and slow inference."
  fi

  # Disk space (in $HOME, where venv + HF cache live)
  local free_kb free_gb
  free_kb=$(df -k "$HOME" | awk 'NR==2 {print $4}')
  free_gb=$(( free_kb / 1024 / 1024 ))
  if [[ "$free_gb" -lt "$MIN_FREE_DISK_GB" ]]; then
    errors+=("Only ${free_gb} GB free in \$HOME (need >= $MIN_FREE_DISK_GB GB).|Free up disk: rm -rf ~/.cache/huggingface (if you don't need cached models), or empty Trash, or move large files off the boot volume.")
  fi

  # Network reachability — needed for HF, brew, pip
  if ! curl -sfI -m 5 https://huggingface.co/ >/dev/null 2>&1; then
    errors+=("Cannot reach https://huggingface.co (timed out or DNS failure).|Check your internet connection. Corporate proxies may need HTTPS_PROXY set, or huggingface.co allowlisted.")
  fi

  # Write permission to $HOME and config dir parent
  if ! touch "$HOME/.dflash-install-write-test" 2>/dev/null; then
    errors+=("Cannot write to \$HOME ($HOME).|Check filesystem permissions; try: ls -ld \$HOME")
  else
    rm -f "$HOME/.dflash-install-write-test"
  fi

  # Xcode CLT (needed by Homebrew and some pip wheels)
  if ! xcode-select -p >/dev/null 2>&1; then
    errors+=("Xcode Command Line Tools not installed.|Install with: xcode-select --install (then re-run this script).")
  fi

  # uv handles Python install if needed (uv python install). We just need
  # uv itself; brew can install it. If neither is present, ask the user.
  if ! command -v uv >/dev/null 2>&1 && ! command -v brew >/dev/null 2>&1; then
    errors+=("Neither 'uv' nor Homebrew is installed.|Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh   — or install Homebrew (https://brew.sh) and re-run this script (we'll brew-install uv automatically).")
  fi

  # Print all errors at once so user can fix them in one pass.
  if [[ ${#errors[@]} -gt 0 ]]; then
    printf "\n%s Preflight failed with %d issue(s):\n\n" "$LOG_PREFIX" "${#errors[@]}" >&2
    local i=1
    for entry in "${errors[@]}"; do
      local reason="${entry%%|*}"
      local fix="${entry#*|}"
      printf "  %d. %s\n     Fix: %s\n\n" "$i" "$reason" "$fix" >&2
      i=$((i + 1))
    done
    exit 1
  fi

  say "Preflight OK: macOS $(sw_vers -productVersion), $(uname -m), ${ram_gb} GB RAM, ${free_gb} GB free."
}

# -------------------------------------------------------------------------
# Step implementations
# -------------------------------------------------------------------------

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return
  fi
  say "Installing Homebrew (will prompt for sudo password)..."
  if ! /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; then
    die "Homebrew install failed." \
        "Run the installer manually from https://brew.sh and retry this script."
  fi
  # On Apple Silicon, brew lives at /opt/homebrew; PATH may not pick it up
  # in this shell session.
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
  command -v brew >/dev/null 2>&1 \
    || die "brew installed but not on PATH." \
           "Add to your shell rc: eval \"\$(/opt/homebrew/bin/brew shellenv)\""
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    say "uv: $(uv --version)"
    return
  fi
  say "Installing uv via Homebrew..."
  brew install uv \
    || die "Failed to install uv via brew." \
           "Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh   then re-run this script."
  command -v uv >/dev/null 2>&1 \
    || die "uv installed but not on PATH." \
           "Restart your shell or add /opt/homebrew/bin to PATH, then re-run."
  say "uv: $(uv --version)"
}

ensure_opencode() {
  if command -v opencode >/dev/null 2>&1; then
    say "opencode: $(opencode --version)"
    return
  fi
  say "Installing opencode via Homebrew..."
  brew install opencode \
    || die "Failed to install opencode." \
           "Run 'brew install opencode' manually to see the error, then re-run this script."
  say "opencode: $(opencode --version)"
}

ensure_uv_env_and_deps() {
  cd "$PROJECT_DIR" \
    || die "Cannot cd to $PROJECT_DIR." "Check the script lives in a real directory."

  if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    die "pyproject.toml missing in $PROJECT_DIR." \
        "Re-clone the repo; pyproject.toml ships with it."
  fi

  # uv sync creates/updates .venv from pyproject.toml. It's idempotent and
  # MUCH faster than pip on re-runs (resolves+verifies in seconds).
  # uv handles installing a matching Python interpreter automatically if
  # the system one is too old.
  say "uv sync (creates .venv if needed; installs/verifies deps)..."
  uv sync 2>&1 | sed "s/^/$LOG_PREFIX   /" \
    || die "uv sync failed." \
           "Run 'uv sync' from $PROJECT_DIR to see the full error. Common: network proxy blocking pypi.org or huggingface.co; corrupt .venv (delete .venv/ and retry); incompatible Python version (uv should auto-fix)."

  # Sanity: import everything end-to-end and check MLX version meets floor.
  uv run python - <<PY \
    || die "Post-install sanity check failed (imports or MLX version too old)." \
           "Try: rm -rf $PROJECT_DIR/.venv && uv sync"
import sys
import mlx.core as mx
import dflash_mlx  # noqa: F401
import mlx_lm  # noqa: F401
import huggingface_hub  # noqa: F401
def parse(v): return tuple(int(x) for x in v.split('.')[:3])
required = parse("${MIN_MLX_VERSION}")
got = parse(mx.__version__)
if got < required:
    sys.stderr.write(f"MLX {mx.__version__} < required {'.'.join(map(str, required))}\n")
    sys.exit(1)
print(f"  mlx={mx.__version__} dflash-mlx + mlx-lm + huggingface_hub OK")
PY
}

patch_serve_py() {
  # Locate serve.py inside the uv-managed .venv via the import system,
  # avoiding any python3.X subdirectory guesswork.
  local serve_py
  serve_py=$(uv run python -c \
    'import dflash_mlx.serve, sys; sys.stdout.write(dflash_mlx.serve.__file__)' 2>/dev/null) \
    || die "Could not locate dflash_mlx/serve.py via uv run." \
           "Re-run 'uv sync' from $PROJECT_DIR; if that fails, delete .venv and re-run this script."

  if [[ ! -f "$serve_py" ]]; then
    die "Cannot find $serve_py." \
        "dflash-mlx layout may have changed. Run 'uv run python -c \"import dflash_mlx.serve as s; print(s.__file__)\"' to investigate."
  fi

  if grep -q "_normalize_messages" "$serve_py"; then
    say "serve.py already patched (idempotent)."
    return
  fi

  say "Patching $serve_py to coalesce system messages (Qwen3.5 chat-template fix)..."
  cp "$serve_py" "$serve_py.bak.$(date +%s)"

  python3 - "$serve_py" <<'PY' \
    || die "serve.py patch failed." \
           "Restore from backup: cp $serve_py.bak.* $serve_py   then file an issue with your dflash-mlx version."
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()
patch = '''def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce all system messages into one at position 0.

    Qwen3.5's chat template requires the system message to be at the
    beginning. Some clients (e.g. opencode) interleave system messages with
    user/assistant turns, which the template rejects with
    'System message must be at the beginning.'.
    """
    system_parts: list[str] = []
    other: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role", "")) == "system":
            text = _message_text(message).strip()
            if text:
                system_parts.append(text)
        else:
            other.append(message)
    if not system_parts:
        return other
    combined = {"role": "system", "content": "\\n\\n".join(system_parts)}
    return [combined, *other]


'''
needle = "def _build_prompt_request("
if needle not in src:
    sys.stderr.write("anchor 'def _build_prompt_request(' not found in serve.py\n")
    sys.exit(2)
idx = src.index(needle)
src = src[:idx] + patch + src[idx:]
old = 'messages = list(payload.get("messages") or [])'
new = 'messages = _normalize_messages(list(payload.get("messages") or []))'
if old not in src:
    sys.stderr.write("anchor for messages assignment not found in serve.py\n")
    sys.exit(3)
src = src.replace(old, new, 1)
path.write_text(src)
print("patched")
PY
}

download_models() {
  say "Downloading $TARGET_MODEL (~18 GB) and $DRAFT_MODEL (~2 GB)..."
  say "  (Resume-safe; Ctrl-C and re-run if interrupted.)"
  uv run hf download "$TARGET_MODEL" \
    || die "Download of $TARGET_MODEL failed." \
           "Common causes: gated model needs HF login (run 'uv run hf auth login'), no disk space, or network drop. Re-run this script — downloads resume."
  uv run hf download "$DRAFT_MODEL" \
    || die "Download of $DRAFT_MODEL failed." \
           "Same fixes as above. Re-run this script — downloads resume."
}

dflash_smoke_test() {
  say "Smoke test: dflash CLI generation..."
  if ! uv run dflash \
              --model "$TARGET_MODEL" \
              --prompt "Reply with exactly the text: dflash-ready" \
              --max-tokens 24 >/tmp/dflash-smoke.log 2>&1; then
    cat /tmp/dflash-smoke.log >&2
    if grep -q "Failed to create Metal shared event" /tmp/dflash-smoke.log; then
      die "Metal shared event allocation failed (known MLX bug)." \
          "1. Verify MLX >= ${MIN_MLX_VERSION} (this script enforces it). 2. Reboot — the macOS Metal event pool only resets cleanly that way. 3. Re-run this script."
    fi
    die "dflash CLI generation failed." \
        "See /tmp/dflash-smoke.log above. Common: model not fully downloaded (re-run script — downloads resume), or out-of-memory (close other apps)."
  fi
  say "dflash CLI works."
}

start_server() {
  if lsof -i ":$DFLASH_PORT" >/dev/null 2>&1; then
    local existing
    existing=$(lsof -ti ":$DFLASH_PORT" 2>/dev/null | head -1)
    # Verify it's actually our dflash-serve answering OpenAI requests, not
    # some unrelated process squatting on the port.
    if curl -sf -m 2 "http://localhost:$DFLASH_PORT/v1/models" >/dev/null 2>&1; then
      say "dflash-serve already running on :$DFLASH_PORT (PID $existing). Reusing."
      # Refresh PID file so later steps reference the actual running PID.
      echo "$existing" > "$SERVER_PID_FILE"
      return
    fi
    die "Port $DFLASH_PORT is bound by PID $existing but it doesn't speak the OpenAI API." \
        "Either kill it (kill $existing) and re-run, or set DFLASH_PORT=<other> and re-run."
  fi
  say "Starting dflash-serve on port $DFLASH_PORT (log: $SERVER_LOG)..."
  # main() chdir'd to $PROJECT_DIR. uv run executes within that project's
  # .venv. We capture the bash background PID; uv typically execs the
  # underlying Python process so this PID is what we kill later.
  nohup uv run dflash-serve \
      --model "$TARGET_MODEL" --port "$DFLASH_PORT" \
      > "$SERVER_LOG" 2>&1 &
  echo $! > "$SERVER_PID_FILE"
  local pid
  pid=$(cat "$SERVER_PID_FILE")

  # Poll /v1/models until ready or timeout.
  local max_attempts=24
  local attempt
  for attempt in $(seq 1 $max_attempts); do
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
      printf "%s ERROR: dflash-serve died during startup.\n" "$LOG_PREFIX" >&2
      printf "%s --- last 20 lines of %s ---\n" "$LOG_PREFIX" "$SERVER_LOG" >&2
      tail -20 "$SERVER_LOG" >&2
      printf "%s -------------------------------\n" "$LOG_PREFIX" >&2
      if grep -q "Failed to create Metal shared event" "$SERVER_LOG"; then
        die "Metal shared event allocation failed at server load." \
            "Reboot — the macOS Metal event pool needs to reset. Then re-run this script."
      fi
      die "dflash-serve crashed (see log above)." \
          "Common: model not on disk (re-run script), port conflict, or insufficient memory."
    fi
    if curl -sf -m 2 "http://localhost:$DFLASH_PORT/v1/models" >/dev/null 2>&1; then
      say "dflash-serve ready (PID $pid)."
      return
    fi
  done
  die "dflash-serve did not become ready within $((max_attempts * 5))s." \
      "Inspect $SERVER_LOG. Slow disk loading the 18 GB model can exceed this; try again or increase the timeout in the script."
}

write_opencode_config() {
  mkdir -p "$OPENCODE_CONFIG_DIR" \
    || die "Cannot create $OPENCODE_CONFIG_DIR." \
           "Check permissions on \$HOME/.config."

  # Compute the desired config and only write (and back up) if it differs
  # from what's already on disk. Re-runs of the script with no changes
  # leave the file untouched and don't accumulate timestamped backups.
  local tmp
  tmp=$(mktemp -t opencode-config-XXXXXX.json) \
    || die "mktemp failed." "Check /tmp is writable."
  trap "rm -f '$tmp'" RETURN

  python3 - "$tmp" "$DFLASH_PORT" "$TARGET_MODEL" "${OPENCODE_CONFIG}" <<'PY' \
    || die "Failed to compose opencode.json." \
           "If existing $OPENCODE_CONFIG has corrupt JSON, delete it and re-run."
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
port, model, existing = sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
cfg = {}
if existing.exists():
    try:
        cfg = json.loads(existing.read_text())
    except Exception as exc:
        sys.stderr.write(f"warning: existing opencode.json is invalid JSON ({exc}); replacing.\n")
        cfg = {}
cfg.setdefault("$schema", "https://opencode.ai/config.json")
cfg.setdefault("provider", {})
cfg["provider"]["dflash"] = {
    "npm": "@ai-sdk/openai-compatible",
    "name": f"DFlash (local {model})",
    "options": {
        "baseURL": f"http://localhost:{port}/v1",
        "apiKey": "dflash",
    },
    "models": {
        model: {
            "name": f"{model} + DFlash",
            "limit": {"context": 32000, "output": 256},
        }
    },
}
cfg.setdefault("agent", {})
cfg["agent"]["qwen-chat"] = {
    "description": f"Plain chat with {model} via DFlash (no tools)",
    "mode": "primary",
    "model": f"dflash/{model}",
    "tools": {
        "write": False, "edit": False, "bash": False, "read": False,
        "list": False, "glob": False, "grep": False, "patch": False,
        "task": False, "todoread": False, "todowrite": False,
        "webfetch": False,
    },
    "prompt": "You are a concise coding assistant. Answer in plain text or fenced code blocks. Do not use tools.",
}
out.write_text(json.dumps(cfg, indent=2) + "\n")
PY

  if [[ -f "$OPENCODE_CONFIG" ]] && cmp -s "$tmp" "$OPENCODE_CONFIG"; then
    say "opencode.json already up to date."
    return
  fi

  if [[ -f "$OPENCODE_CONFIG" ]]; then
    local backup="$OPENCODE_CONFIG.bak.$(date +%s)"
    cp "$OPENCODE_CONFIG" "$backup" \
      || die "Could not back up existing opencode.json to $backup." \
             "Check permissions on $OPENCODE_CONFIG_DIR."
    say "Backed up existing opencode.json to $(basename "$backup")."
  fi

  cp "$tmp" "$OPENCODE_CONFIG" \
    || die "Failed to write $OPENCODE_CONFIG." \
           "Check that $OPENCODE_CONFIG is writable."
  say "Wrote $OPENCODE_CONFIG."
}

opencode_smoke_test() {
  say "Smoke test: opencode -> dflash -> Qwen3.5..."
  local sess_dir="$HOME/.local/share/opencode/storage/part"
  mkdir -p "$sess_dir"

  # Stamp a marker NOW so we only consider session files created after this
  # point. Without this, a re-run picks up a stale response from a prior
  # run and reports "success" even if this run produced nothing.
  local marker
  marker=$(mktemp -t dflash-smoke-marker-XXXXXX) \
    || die "mktemp failed." "Check /tmp is writable."
  trap "rm -f '$marker'" RETURN

  # opencode run only renders to a TTY, so its stdout is useless without one.
  # We let it write into its own session storage and read the result back.
  ( timeout 120 opencode run --agent qwen-chat \
      "Output ONLY a fenced python block defining def add(a,b): return a+b" \
      >/dev/null 2>&1 ) || true

  local new_part
  new_part=$(find "$sess_dir" -type f -newer "$marker" 2>/dev/null \
    | xargs grep -l '"type": "text"' 2>/dev/null \
    | xargs grep -l '```' 2>/dev/null \
    | xargs ls -t 2>/dev/null | head -1 || true)

  if [[ -z "$new_part" ]]; then
    warn "Smoke test produced no NEW opencode response in 120s."
    warn "  Setup is likely still functional — try interactively: opencode run --agent qwen-chat \"hi\""
    warn "  Server log: $SERVER_LOG"
    return
  fi
  say "opencode response:"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['text'])" "$new_part" \
    || warn "Could not pretty-print response part $new_part."
}

print_done() {
  local pid="?"
  [[ -f "$SERVER_PID_FILE" ]] && pid=$(cat "$SERVER_PID_FILE")
  cat <<EOF

================================================================
Setup complete.

Server:  dflash-serve on http://localhost:$DFLASH_PORT/v1
         (PID $pid; log: $SERVER_LOG)

Use it:
  opencode run --agent qwen-chat "your prompt here"
  opencode --agent qwen-chat        # interactive TUI

Stop the server:
  kill \$(cat $SERVER_PID_FILE)

Restart the server later (no env activation — uv handles it):
  cd $PROJECT_DIR
  nohup uv run dflash-serve --model $TARGET_MODEL --port $DFLASH_PORT \\
    > $SERVER_LOG 2>&1 &
  echo \$! > $SERVER_PID_FILE

Run anything in the project's env without activating:
  cd $PROJECT_DIR && uv run <command>      # e.g. uv run python, uv run dflash

If you ever see "RuntimeError: [Event::Event] Failed to create Metal
shared event":
  1. Confirm MLX is >= ${MIN_MLX_VERSION} (this script enforces it; earlier versions
     have a known leak fixed in ml-explore/mlx#3159).
  2. Reboot — the macOS Metal event pool only resets cleanly that way.
  3. If it recurs on ${MIN_MLX_VERSION}+, file upstream against ml-explore/mlx with
     a repro; reference mlx-lm#887.
================================================================
EOF
}

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
main() {
  say "DFlash + opencode + $TARGET_MODEL setup starting."
  say "Project dir: $PROJECT_DIR"
  cd "$PROJECT_DIR" || die "Cannot enter $PROJECT_DIR." "Check the script lives in a real directory."
  preflight
  ensure_homebrew
  ensure_uv
  ensure_opencode
  ensure_uv_env_and_deps
  patch_serve_py
  download_models
  dflash_smoke_test
  start_server
  write_opencode_config
  opencode_smoke_test
  print_done
}

main "$@"
