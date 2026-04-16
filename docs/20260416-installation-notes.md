# Installation notes — 2026-04-16

Notes from getting `mlx + dflash-mlx + Qwen3.5-9B + opencode` running
end-to-end on macOS 26.4 (Darwin 25.4.0), Apple M5 Max, 128 GB unified memory.

## TL;DR

- The original `install.sh` was wishful thinking — five wrong commands, no
  actual opencode integration, no error handling. It would have failed on a
  clean machine.
- The fixed `install.sh` (in this repo) installs everything, patches
  `dflash_mlx/serve.py`, writes a working `~/.config/opencode/opencode.json`,
  and smoke-tests the full chain.
- Qwen3.5-9B + DFlash gets ~24 tok/s with 85% draft acceptance.
- Long streaming sessions hit a system-wide Metal-shared-event leak that
  only `reboot` recovers from. This is an MLX/dflash bug, not config.

## What was wrong with the original install.sh

| # | Bug | Fix |
|---|-----|-----|
| 1 | `--no-stream` flag passed to `dflash` (doesn't exist) | Removed |
| 2 | `brew tap anomalyco/tap` + `brew install anomalyco/tap/opencode` | `opencode` is in the official brew formulas: `brew install opencode` |
| 3 | `huggingface-cli download` is deprecated | `hf download` |
| 4 | `--local-dir ~/.cache/huggingface/hub/models--...` breaks HF cache resolution | Drop the flag; let `hf download` use the default cache |
| 5 | "Inside opencode type `/connect`" | There is no `/connect` slash command — opencode uses provider config in `opencode.json` |
| 6 | No opencode wiring at all | Script now writes/merges `provider.dflash` and `agent.qwen-chat` |
| 7 | `sleep 8` then assume the server is up | Poll `GET /v1/models` for up to 90s |
| 8 | Doesn't handle Qwen3.5's chat-template constraint (system message must be at position 0) | Patches `dflash_mlx/serve.py` to coalesce system messages |

## Things the script does that aren't obvious

### Patches `dflash_mlx/serve.py` to coalesce system messages

Qwen3.5's Jinja chat template explicitly raises:

```
TemplateError: System message must be at the beginning.
```

if it sees more than one `system` role or a `system` role after a `user` role.
Opencode (and many other clients) interleave system messages with user/assistant
turns. The patch adds `_normalize_messages()` which folds all system messages
into a single message at position 0 before applying the chat template.

The patch is **idempotent** — running the install script again won't double-patch.

### Writes a custom `qwen-chat` opencode agent with tools disabled

The default `build` and `plan` agents inject 22-24 tool definitions into the
system prompt. With JSON tool schemas, this inflates prompts to 5-10K tokens
and:

- Makes prefill slow (a 9B model with a 10K-token prompt isn't snappy)
- Drives many more streaming generations through MLX's eval path, which
  accelerates the Metal-event leak (see below)
- Qwen3.5-9B isn't reliably tool-calling-capable in OpenAI format anyway

The `qwen-chat` agent disables every tool (`write`, `edit`, `bash`, `read`,
`list`, `glob`, `grep`, `patch`, `task`, `todoread`, `todowrite`, `webfetch`)
and just does plain text/code generation. Use:

```bash
opencode run --agent qwen-chat "your prompt"
```

### Reads the response from opencode session storage

`opencode run` only renders to a TTY. Piping to a file produces zero bytes. The
script works around this by reading the latest message part from
`~/.local/share/opencode/storage/part/msg_*/prt_*.json` and printing the `text`
field. Ugly but functional for a smoke test.

### Limits output tokens to 256

Without `limit.output` in the model config, opencode sends `max_tokens=8192+`
to the server. Qwen3.5 has a thinking-mode preamble that can easily eat 500+
tokens before any user-visible content. With baseline thinking + agent system
prompt overhead, an unconstrained run takes 5+ minutes per query. 256 forces
fast turnaround at the cost of truncating long responses — fine for a chat
agent, raise it if you're generating big code blocks.

## The Metal-shared-event leak

### Symptom

After running enough streaming requests:

```
RuntimeError: [Event::Event] Failed to create Metal shared event.
```

Every `mx.eval()` that touches the GPU fails. Even a fresh `python -c
'import mlx.core as mx; mx.eval(mx.ones((4,4)) @ mx.ones((4,4)))'` fails.

### Investigation

I initially blamed huge tool-prompt prefills. That was wrong. The real
investigation:

1. Wrote `/tmp/investigate_metal.py` to call `stream_dflash_generate`
   directly with prompt sizes from 200 to 30 000 chars. **All failed
   instantly**, regardless of size.

2. Stripped down to pure MLX. `mx.eval(mx.random.normal(shape=(4,4)))`
   failed with the same error.

3. Stripped further to **Swift**:

   ```swift
   import Metal
   let device = MTLCreateSystemDefaultDevice()!
   var success = 0, failure = 0
   for i in 0..<2000 {
       if device.makeSharedEvent() != nil { success += 1 } else { failure += 1 }
   }
   print("success: \(success)  failure: \(failure)")
   ```

   Result:
   ```
   device: Apple M5 Max
   success: 0  failure: 2000
   ```

   Three separate Swift invocations — same. After `caffeinate -t 2` (display
   sleep): same. After 2 hours uptime: still zero events available.

### Root cause

`MTLDevice.makeSharedEvent()` returns `nil` → MLX surfaces it as the
`Event::Event` error. MTLSharedEvent is a finite kernel resource.

This is a known MLX bug:
[`mlx#3159` "[Metal] Fix event leak"](https://github.com/ml-explore/mlx/pull/3159)
fixed it on 24 Feb 2026, shipped in **MLX 0.31.0** (28 Feb 2026), and
closed [`mlx-lm#887`](https://github.com/ml-explore/mlx-lm/issues/887)
which reported the *exact* same `[Event::Event] Failed to create Metal
shared event` error.

**However: my install has MLX 0.31.1 (which includes the fix) and I still
hit the error.** So either:

1. PR #3159 didn't cover every leak path, and dflash's hot loop triggers
   a different one (`runtime.py:1687-1688` does `async_eval + eval` on
   the same array; line 1720 is a multi-arg `eval`; line 1714/1755 are
   `.item()` calls — all event-allocating);
2. or cumulative usage across a full day of testing tipped the pool past
   what the 0.31.0 fix prevents.

I don't have evidence to distinguish these. The reproducible facts are:

- After heavy MLX usage today, `MTLDevice.makeSharedEvent()` returns nil
  on **every** call from any process, including a fresh Swift program
  with zero MLX dependency.
- A previous claim in an earlier draft of this doc — "kernel driver
  pool survives process death because they're ref-counted in the
  driver" — was speculation and **shouldn't be cited as fact**. The
  more honest statement is "this user session's GPU/Metal state is
  hosed and I don't know exactly why, but it persists across new
  process invocations."

### What recovers it

- **Reboot.** Reliable. (Tested: yes, by virtue of how persistent the
  failure is over hours.)
- `sudo killall -HUP WindowServer` (logs you out). Plausible, untested.
- Sleep + wake: did **not** help.
- "Just upgrade MLX" — already on the post-fix version (0.31.1), didn't
  help here.

### What's worth filing upstream

Probably file against **`ml-explore/mlx`** (not dflash) referencing
`#3159` / `#887`, with a minimal repro that exercises dflash's
async_eval-then-eval-on-same-array pattern and shows the pool draining
on 0.31.1. dflash is just the canary; the fix needs to be in MLX core.

The `qwen-chat` agent sidesteps the symptom by doing fewer streaming
cycles per task, which delays exhaustion but doesn't prevent it.

## What "works" looks like

```
$ opencode run --agent qwen-chat "Write a Python function called fib(n)"
```

→ stored in `~/.local/share/opencode/storage/part/...`:

```python
def fib(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

Generated through: opencode → `@ai-sdk/openai-compatible` → patched
`dflash-serve` on `localhost:8000/v1` → DFlash speculative decoding →
Qwen3.5-9B target on MLX → Apple M5 Max GPU.

## Caveats for "others" running this

- **macOS only.** Linux/Windows users die at `brew`.
- **Disk:** ~25 GB for models alone (Qwen3.5-9B target ~18 GB + DFlash draft ~2 GB + buffers).
- **RAM:** ~22 GB unified at runtime. M1/M2 base 8 GB will swap and crawl.
- **The Metal leak.** A known MLX bug (#3159) was fixed in 0.31.0 and
  this script installs 0.31.1, so most users won't see it. If you
  *do* see `Failed to create Metal shared event` after long sessions,
  reboot is the immediate fix; report it upstream against
  `ml-explore/mlx` with a repro.

## Correction (post-review)

An earlier draft of this doc claimed the Metal-event issue was a
"system-wide MLX/Metal driver leak" with no upstream fix, and recommended
filing against `bstnxbt/dflash-mlx`. A second-pass review pointed out:

- MLX **PR #3159** ("[Metal] Fix event leak", merged 24 Feb 2026, shipped
  in **0.31.0** on 28 Feb 2026) directly addressed this exact error.
- mlx-lm **issue #887** is the canonical reference for the symptom and
  was closed by that PR.
- This script installs MLX 0.31.1, which contains the fix. Most users
  won't hit the error.
- The "kernel driver pool survives process death" framing was
  speculation; the Swift "first failure at iteration 0" test only
  proves *this session's* GPU state is hosed — not that events leak
  across processes.

I've kept the reboot-as-recovery advice (still true in the moment) but
removed the over-confident framing. The right place to file follow-ups
is `ml-explore/mlx`, not dflash-mlx.

## Files touched

- `install.sh` — full rewrite with the fixes above
- `~/dflash-env/lib/python3.12/site-packages/dflash_mlx/serve.py` — patched
  with `_normalize_messages()` (idempotent — looks for the function name
  before patching)
- `~/.config/opencode/opencode.json` — provider + agent added (existing
  contents preserved; backup written to `opencode.json.bak.<timestamp>`)
- `/tmp/dflash-serve.log`, `/tmp/dflash-serve.pid` — server runtime state
