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
VENV_DIR="${VENV_DIR:-$HOME/dflash-env}"
DFLASH_PORT="${DFLASH_PORT:-8000}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}"
OPENCODE_CONFIG_DIR="$HOME/.config/opencode"
OPENCODE_CONFIG="$OPENCODE_CONFIG_DIR/opencode.json"

# Minimum requirements
MIN_MACOS_MAJOR=14         # Sonoma. Earlier macOS lacks Metal 3 features mlx uses.
MIN_FREE_DISK_GB=30        # ~20 GB models + buffer
MIN_RAM_GB=16              # Qwen3.5-9B bf16 + draft + KV cache need ~22 GB
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10
MIN_MLX_VERSION="0.31.0"   # ml-explore/mlx#3159 (Metal event leak fix)

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

  # Python: check it exists *and* is recent enough. If brew is available
  # we'll auto-upgrade; otherwise this needs to fail.
  local have_python_ok=0
  if command -v python3 >/dev/null 2>&1; then
    local py_major py_minor
    py_major=$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
    py_minor=$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
    if [[ "$py_major" -gt "$MIN_PYTHON_MAJOR" ]] || \
       [[ "$py_major" -eq "$MIN_PYTHON_MAJOR" && "$py_minor" -ge "$MIN_PYTHON_MINOR" ]]; then
      have_python_ok=1
    fi
  fi
  if [[ "$have_python_ok" -eq 0 ]] && ! command -v brew >/dev/null 2>&1; then
    errors+=("Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} not found and Homebrew is missing (can't auto-install).|Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"  — or install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ another way.")
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

ensure_python() {
  local need_install=0
  if ! command -v python3 >/dev/null 2>&1; then
    need_install=1
  else
    local py_major py_minor
    py_major=$(python3 -c 'import sys; print(sys.version_info.major)')
    py_minor=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [[ "$py_major" -lt "$MIN_PYTHON_MAJOR" ]] || \
       [[ "$py_major" -eq "$MIN_PYTHON_MAJOR" && "$py_minor" -lt "$MIN_PYTHON_MINOR" ]]; then
      need_install=1
    fi
  fi
  if [[ "$need_install" -eq 1 ]]; then
    say "Installing Python via Homebrew..."
    brew install python@3.12 \
      || die "Failed to install Python." \
             "Try: brew doctor   (then re-run this script)."
  fi
  say "Python: $(python3 --version)"
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

ensure_venv_and_deps() {
  if [[ ! -d "$VENV_DIR" ]]; then
    say "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" \
      || die "venv creation failed at $VENV_DIR." \
             "Check disk space and write permissions on \$HOME, then retry."
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"

  say "Upgrading pip..."
  pip install --upgrade pip --quiet \
    || die "pip self-upgrade failed." \
           "Likely a network issue. Check connectivity to pypi.org and retry."

  say "Installing MLX, dflash-mlx, mlx-lm, huggingface_hub..."
  # mlx>=0.31.0 carries ml-explore/mlx#3159 (Metal event leak fix). dflash-mlx
  # only requires mlx>=0.25.0 so we pin explicitly.
  pip install --quiet "mlx>=${MIN_MLX_VERSION}" dflash-mlx "huggingface_hub[cli]" mlx-lm \
    || die "pip install of MLX/dflash failed." \
           "Run 'pip install \"mlx>=${MIN_MLX_VERSION}\" dflash-mlx mlx-lm' inside the venv to see the full error. Common causes: outdated pip, network proxy, or no arm64 wheel for your Python version."

  local mlx_ver
  mlx_ver=$(python3 -c 'import mlx.core as m; print(m.__version__)' 2>/dev/null) \
    || die "MLX failed to import after install." \
           "Re-run this script; if it persists, delete $VENV_DIR and start over."
  say "MLX: $mlx_ver"

  python3 - "$mlx_ver" "$MIN_MLX_VERSION" <<'PY' || die "MLX version check failed." \
    "MLX $(python3 -c 'import mlx.core as m; print(m.__version__)') is below the required ${MIN_MLX_VERSION}. Try: pip install -U \"mlx>=${MIN_MLX_VERSION}\""
import sys
def parse(v): return tuple(int(x) for x in v.split('.')[:3])
sys.exit(0 if parse(sys.argv[1]) >= parse(sys.argv[2]) else 1)
PY

  python3 -c "import dflash_mlx" 2>/dev/null \
    || die "dflash-mlx failed to import after install." \
           "Try: pip install --force-reinstall dflash-mlx"
  say "dflash-mlx: installed"
}

patch_serve_py() {
  local serve_py
  serve_py="$VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/dflash_mlx/serve.py"

  if [[ ! -f "$serve_py" ]]; then
    die "Cannot find $serve_py." \
        "dflash-mlx layout may have changed. Run 'pip show -f dflash-mlx' to find serve.py and report the path."
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
  hf download "$TARGET_MODEL" \
    || die "Download of $TARGET_MODEL failed." \
           "Common causes: gated model needs HF login (run 'hf auth login'), no disk space, or network drop. Re-run this script — downloads resume."
  hf download "$DRAFT_MODEL" \
    || die "Download of $DRAFT_MODEL failed." \
           "Same fixes as above. Re-run this script — downloads resume."
}

dflash_smoke_test() {
  say "Smoke test: dflash CLI generation..."
  if ! dflash --model "$TARGET_MODEL" \
              --prompt "Reply with exactly the text: dflash-ready" \
              --max-tokens 24 >/tmp/dflash-smoke.log 2>&1; then
    cat /tmp/dflash-smoke.log >&2
    if grep -q "Failed to create Metal shared event" /tmp/dflash-smoke.log; then
      die "Metal shared event allocation failed (known MLX bug)." \
          "1. Verify MLX >= ${MIN_MLX_VERSION} (this script enforces it). 2. Reboot — the macOS Metal event pool only resets cleanly that way. 3. Re-run this script."
    fi
    die "dflash CLI generation failed." \
        "See /tmp/dflash-smoke.log above. Common: model not fully downloaded (re-run step 7), or out-of-memory (close other apps)."
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
  nohup dflash-serve --model "$TARGET_MODEL" --port "$DFLASH_PORT" \
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

Re-activate the venv later:
  source $VENV_DIR/bin/activate

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
  preflight
  ensure_homebrew
  ensure_python
  ensure_opencode
  ensure_venv_and_deps
  patch_serve_py
  download_models
  dflash_smoke_test
  start_server
  write_opencode_config
  opencode_smoke_test
  print_done
}

main "$@"
