# opencode-mlx-qwen-dflash

A one-shot installer that wires together a fast local code-generation stack
on Apple Silicon: **opencode** (terminal coding agent) → **DFlash**
(speculative decoding for MLX) → **Qwen3.5-9B** running on your Mac's
GPU via **MLX**.

End-to-end on an M-series Mac you get ~24 tok/s with 85% draft acceptance,
no API key, no per-token cost, fully offline after the initial model
download.

## What it does

`install.sh` is a single command that:

1. Verifies your Mac meets the prerequisites (preflight bails early with a
   list of fixes if not).
2. Installs Homebrew (if missing), `uv` (the fast Python package manager),
   and `opencode` — all from the official Homebrew formulas.
3. Runs `uv sync` to create a project-local `.venv` from `pyproject.toml`
   and install `mlx>=0.31.0` (which carries the Metal-event-leak fix —
   ml-explore/mlx#3159), `dflash-mlx`, `mlx-lm`, and `huggingface_hub`.
   uv installs a matching Python automatically if your system one is too old.
4. Patches `dflash_mlx/serve.py` to coalesce interleaved system messages.
   Qwen3.5's chat template otherwise rejects them with
   *"System message must be at the beginning."* The patch is **idempotent**
   — re-running the installer is safe.
5. Downloads the target model (`Qwen/Qwen3.5-9B`, ~18 GB) and the DFlash
   draft model (`z-lab/Qwen3.5-9B-DFlash`, ~2 GB) using `hf download`.
   Resume-safe: re-run if interrupted.
6. Smoke-tests `dflash` to confirm MLX + Qwen + DFlash all load and
   generate.
7. Starts `dflash-serve` in the background — an OpenAI-compatible HTTP
   server on `http://localhost:8000/v1`. Polls `/v1/models` until the
   model is loaded; bails with the server log if it crashes.
8. Writes `~/.config/opencode/opencode.json` with:
   - a `dflash` provider pointing at `localhost:8000`
   - a `qwen-chat` agent with **all tools disabled** (the default
     `build`/`plan` agents inject 20+ tool definitions which inflate
     prompts to 5-10K tokens — slow on a 9B model, and Qwen3.5-9B
     isn't a reliable OpenAI-format tool-caller anyway)
   - any pre-existing config is backed up to `opencode.json.bak.<timestamp>`
9. Runs an end-to-end smoke test through opencode and prints the
   generated code as proof the chain works.

## Prerequisites

The script enforces these in preflight and bails with a clear fix if any
are missing.

| Requirement | Minimum | Notes |
|---|---|---|
| OS | macOS 14 (Sonoma) | MLX needs Metal 3 |
| CPU | Apple Silicon (M1+) | MLX is arm64-only |
| RAM | 16 GB unified (warn) | 22 GB used at runtime; less than 16 GB will swap |
| Free disk | 30 GB in `$HOME` | Models ~20 GB + venv + buffer |
| Python | 3.10+ | Auto-installed by `uv` if your system one is too old |
| Xcode CLT | installed | `xcode-select --install` if not |
| Network | reachable `huggingface.co`, `pypi.org`, `astral.sh` | For model + package downloads |

Homebrew, `uv`, and `opencode` are auto-installed by the script if missing.

## Install

```bash
git clone git@github.com:boxabirds/opencode-mlx-qwen-dflash.git
cd opencode-mlx-qwen-dflash
./install.sh
```

Configurable via env vars:

| Variable | Default | What |
|---|---|---|
| `DFLASH_PORT` | `8000` | Local server port |
| `TARGET_MODEL` | `Qwen/Qwen3.5-9B` | Target (verifier) model |
| `DRAFT_MODEL` | `z-lab/Qwen3.5-9B-DFlash` | Draft model for speculative decoding |

The Python env lives at `.venv/` inside the cloned repo (managed by `uv`).
There's nothing to "activate" — every command goes through `uv run`.

The installer is **idempotent** — re-run it any time. It detects existing
state (venv, downloaded models, patched serve.py, port already bound) and
skips or reuses.

## Usage

After the installer completes, the server is already running. Use:

```bash
# One-shot prompt, response goes to opencode's session storage
opencode run --agent qwen-chat "your prompt here"

# Interactive TUI
opencode --agent qwen-chat
```

To stop the background server:

```bash
kill $(cat /tmp/dflash-serve.pid)
```

To restart it later (after a reboot, say) — no env activation needed:

```bash
cd /path/to/opencode-mlx-qwen-dflash
nohup uv run dflash-serve --model Qwen/Qwen3.5-9B --port 8000 \
  > /tmp/dflash-serve.log 2>&1 &
echo $! > /tmp/dflash-serve.pid
```

Or just re-run `./install.sh` — it detects the existing setup, skips
everything that's already done, and brings the server back up.

Run anything else in the project's Python env without activating:

```bash
cd /path/to/opencode-mlx-qwen-dflash
uv run python                     # REPL
uv run dflash --model Qwen/Qwen3.5-9B --prompt "hi"
```

## Troubleshooting

### `RuntimeError: [Event::Event] Failed to create Metal shared event`

A known macOS Metal-shared-event leak triggered by long MLX streaming
sessions. Mostly fixed in MLX 0.31.0
([ml-explore/mlx#3159](https://github.com/ml-explore/mlx/pull/3159) which
closed [ml-explore/mlx-lm#887](https://github.com/ml-explore/mlx-lm/issues/887)),
which this script enforces.

If you still hit it on heavy use:

1. Confirm MLX is up to date: `pip show mlx` (need ≥ 0.31.0).
2. **Reboot.** The macOS Metal event pool only resets cleanly that way.
   Sleep+wake doesn't help.
3. If it recurs on 0.31.0+, file upstream against `ml-explore/mlx` with
   a minimal repro; reference `mlx-lm#887`.

### Smoke test produces no opencode response

`opencode run` only renders to a TTY; the script reads the response back
from session storage. If the smoke test times out, the server is likely
fine — try interactively:

```bash
opencode --agent qwen-chat
```

### Server didn't come up in 120 seconds

Look at `/tmp/dflash-serve.log`. Common causes:

- Slow disk loading the 18 GB target model from cold cache → just wait
  longer or re-run.
- Port 8000 already bound → `lsof -i :8000` to find the holder.
- OOM during model load → close other apps; check Activity Monitor.

### HF download fails with auth error

`Qwen/Qwen3.5-9B` and the DFlash draft are public, but if you hit auth:

```bash
hf auth login   # or set HF_TOKEN
```

## Files this script touches

- `<repo>/.venv/` — uv-managed Python env (delete to start fresh; `uv sync`
  rebuilds it)
- `<repo>/uv.lock` — lockfile uv writes for reproducibility
- `<repo>/.venv/lib/python*/site-packages/dflash_mlx/serve.py` — patched
  in place; backup written alongside as `serve.py.bak.<timestamp>`
- `~/.cache/huggingface/hub/` — downloaded model weights
- `~/.config/opencode/opencode.json` — provider + agent config (global,
  visible from any directory; existing file backed up to
  `opencode.json.bak.<timestamp>`)
- `/tmp/dflash-serve.log` — server stdout/stderr
- `/tmp/dflash-serve.pid` — server PID

## Background

See [`docs/20260416-installation-notes.md`](docs/20260416-installation-notes.md)
for the full investigation log: what was wrong with my first attempt at
this, the Metal-event-leak deep dive, and a post-review correction
acknowledging where my analysis overreached.

## Caveats

- **macOS only.** Linux/Windows users die at `brew`.
- **Qwen3.5-9B is not a reliable OpenAI-format tool caller.** This is why
  the `qwen-chat` agent has all tools disabled. If you want
  agentic/tool-using opencode behavior, point it at a frontier API or
  use a model trained for tool use.
- **The Metal-event leak is mostly fixed but not provably eliminated.**
  Heavy users may still hit it. Reboot recovers.

## License

The installer in this repo is provided as-is. Component licenses:

- [MLX](https://github.com/ml-explore/mlx) — MIT
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — MIT
- [opencode](https://opencode.ai) — see upstream
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — see model card
- [Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) — MIT
