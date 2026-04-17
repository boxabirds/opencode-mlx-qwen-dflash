"""Coding-task benchmark suite for a local OpenAI-compatible server
(dflash-serve serving Qwen3.5).

Tests are graded by complexity:

  L1 trivial single function
  L2 stdlib pitfall (unicode grapheme handling)
  L3 algorithm with edge cases (binary search)
  L4 data structure design (LRU cache)
  L5 debugging — find and fix a subtle bug
  L6 systems design — concurrency + clean API (rate limiter)
  L7 agentic — opencode with qwen-coder reads/edits a sandbox repo
  L8 long context — needle-in-a-haystack across ~64K tokens of fake repo

Each test reports wall time, completion tokens, and tok/s. A cheap
substring/regex check flags clearly broken outputs but does NOT replace
human review of the actual code.

Usage:
    uv run python scripts/coding_tests.py --model Qwen/Qwen3.5-35B-A3B
    uv run python scripts/coding_tests.py --tests L1,L4,L8
    uv run python scripts/coding_tests.py --no-think=false   # leave thinking on
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# -----------------------------------------------------------------------------
# Test definitions
# -----------------------------------------------------------------------------

SYSTEM_CONCISE = (
    "You are a concise coding assistant. Output only the requested code. "
    "Use a single fenced code block. No prose before or after unless the "
    "task explicitly asks for it."
)


@dataclass
class Test:
    name: str
    difficulty: int
    notes: str
    # Either: a chat-completion test (system + user)
    system: Optional[str] = None
    user: Optional[str] = None
    max_tokens: int = 600
    # Sanity check on the model's text response.
    expect_substring: Optional[str] = None
    expect_regex: Optional[str] = None
    # OR: a fully custom run function (for L7 agentic / L8 long-context).
    custom_run: Optional[Callable[["RunContext"], dict]] = None


@dataclass
class RunContext:
    base_url: str
    model: str
    no_think: bool
    project_dir: Path


# -----------------------------------------------------------------------------
# HTTP helper
# -----------------------------------------------------------------------------

def call_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    no_think: bool = True,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed if elapsed > 0 else 0,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls"),
    }


# -----------------------------------------------------------------------------
# L7: agentic — opencode with qwen-coder against a sandbox repo
# -----------------------------------------------------------------------------

L7_BUGGY_FILE = '''def kth_largest(nums, k):
    """Return the kth largest element of nums."""
    nums.sort()
    return nums[k]   # bug: should be nums[-k]


if __name__ == "__main__":
    print(kth_largest([3, 1, 4, 1, 5, 9, 2, 6], 2))  # should print 6
'''


def run_l7_agentic(ctx: RunContext) -> dict[str, Any]:
    """Spin up a sandbox dir with a buggy Python file, run opencode --agent
    qwen-coder asking it to find and fix the bug, then check whether the
    resulting file actually fixes it (runs and prints 6)."""
    if not shutil.which("opencode"):
        return {"error": "opencode CLI not on PATH; install it via brew first."}

    sandbox = Path(tempfile.mkdtemp(prefix="bench-l7-"))
    target = sandbox / "kth.py"
    target.write_text(L7_BUGGY_FILE)
    sess_part_dir = Path.home() / ".local/share/opencode/storage/part"
    sess_part_dir.mkdir(parents=True, exist_ok=True)
    marker = Path(tempfile.mkstemp(prefix="bench-l7-marker-")[1])

    prompt = (
        f"Read kth.py in this directory. There is a bug in kth_largest. "
        f"Use the read tool to inspect it, then use the edit/write tool to "
        f"fix it. Verify the fix would make the script print 6."
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["opencode", "run", "--agent", "qwen-coder", prompt],
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        marker.unlink(missing_ok=True)
        shutil.rmtree(sandbox, ignore_errors=True)
        return {"error": "opencode timed out after 300s", "elapsed_s": elapsed}
    elapsed = time.perf_counter() - t0

    # Did the model actually edit the file to fix the bug?
    fixed_content = target.read_text() if target.exists() else ""
    file_now_correct = "nums[-k]" in fixed_content and "nums[k]" not in fixed_content.replace(
        "nums[-k]", ""
    )

    # Try to run the script as ground truth.
    runs_correctly = False
    try:
        out = subprocess.run(
            ["python3", str(target)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        runs_correctly = out.stdout.strip() == "6"
    except Exception:
        pass

    # Walk session storage to find the assistant's last text part for context.
    assistant_text = ""
    try:
        latest = max(
            (p for p in sess_part_dir.rglob("prt_*.json") if p.stat().st_mtime > marker.stat().st_mtime),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if latest:
            j = json.loads(latest.read_text())
            assistant_text = j.get("text", "") if j.get("type") == "text" else ""
    except Exception:
        pass

    marker.unlink(missing_ok=True)
    shutil.rmtree(sandbox, ignore_errors=True)
    return {
        "elapsed_s": elapsed,
        "completion_tokens": 0,  # opencode doesn't expose token usage
        "tokens_per_s": 0,
        "finish_reason": f"opencode_exit={proc.returncode}",
        "content": (
            f"file edited correctly: {file_now_correct}\n"
            f"runs and prints 6: {runs_correctly}\n"
            f"--- assistant tail ---\n{assistant_text[-800:]}"
        ),
        "_l7_passed": file_now_correct and runs_correctly,
    }


# -----------------------------------------------------------------------------
# L8: long context — needle in a haystack
# -----------------------------------------------------------------------------

def _build_long_haystack(approx_tokens: int) -> tuple[str, str, str]:
    """Build a synthetic codebase-like prompt of ~approx_tokens.

    Returns (haystack, needle_question, expected_substring_in_answer).

    The needle is a specific function definition with a unique signature
    buried at a random position in the haystack. The model must locate it
    and report the function's return value.
    """
    # ~4 chars per token is the rough Qwen tokenizer ratio.
    target_chars = approx_tokens * 4
    # Generate a haystack of fake "module" stubs.
    modules: list[str] = []
    chars = 0
    i = 0
    while chars < target_chars:
        mod = (
            f"# module/widget_{i:05d}.py\n"
            f"def widget_{i:05d}_init(config: dict) -> dict:\n"
            f"    \"\"\"Initialize widget {i}.\"\"\"\n"
            f"    return {{'id': {i}, 'ready': True}}\n"
            f"\n"
            f"def widget_{i:05d}_render(state: dict) -> str:\n"
            f"    return f\"<widget-{i:05d} state={{state}}/>\"\n"
            f"\n"
        )
        modules.append(mod)
        chars += len(mod)
        i += 1

    # Insert a single unique needle at ~75% through the haystack.
    needle_idx = (3 * len(modules)) // 4
    needle = (
        "# module/secret_treasure.py\n"
        "def find_secret_treasure_value() -> int:\n"
        "    \"\"\"This is the only function in the codebase whose return\n"
        "    value is the answer to the question.\"\"\"\n"
        "    return 73821\n"
        "\n"
    )
    modules.insert(needle_idx, needle)
    haystack = "".join(modules)

    question = (
        "Above is a synthetic codebase. Find the function "
        "find_secret_treasure_value (it appears EXACTLY once) and report the "
        "integer it returns. Reply with ONLY the integer on a line by itself, "
        "nothing else."
    )
    return haystack, question, "73821"


def run_l8_long_context(ctx: RunContext, approx_tokens: int = 64_000) -> dict[str, Any]:
    haystack, question, expected = _build_long_haystack(approx_tokens)
    user = haystack + "\n\n" + question
    result = call_chat(
        base_url=ctx.base_url,
        model=ctx.model,
        messages=[
            {"role": "system", "content": "You read code carefully and answer precisely."},
            {"role": "user", "content": user},
        ],
        max_tokens=64,
        no_think=ctx.no_think,
    )
    if "error" not in result:
        result["expected_substring"] = expected
        result["_l8_passed"] = expected in (result.get("content") or "")
        result["_l8_haystack_chars"] = len(haystack)
        result["_l8_target_tokens"] = approx_tokens
    return result


# -----------------------------------------------------------------------------
# Test catalogue
# -----------------------------------------------------------------------------

TESTS: list[Test] = [
    Test(
        name="L1-fizzbuzz",
        difficulty=1,
        system=SYSTEM_CONCISE,
        user="Write a Python function fizzbuzz(n) that prints FizzBuzz from 1 to n.",
        max_tokens=400,
        expect_regex=r"def\s+fizzbuzz",
        notes="Trivial baseline: should be instant and correct.",
    ),
    Test(
        name="L2-string-reverse-unicode",
        difficulty=2,
        system=SYSTEM_CONCISE,
        user=(
            "Write a Python function reverse_string(s: str) -> str that "
            "reverses a string and correctly handles multi-codepoint emoji "
            "like family/skin-tone sequences. You may use any pip package. "
            "Show the function and one doctest demonstrating the emoji case."
        ),
        max_tokens=800,
        expect_regex=r"def\s+reverse_string",
        notes="Tests grapheme awareness — most LLMs fail this with naive [::-1].",
    ),
    Test(
        name="L3-binary-search",
        difficulty=3,
        system=SYSTEM_CONCISE,
        user=(
            "Implement Python: bsearch(arr: list[int], target: int) -> int. "
            "Return the leftmost index of target in a sorted array, or -1 if "
            "absent. Must be O(log n) and handle empty arrays. No imports."
        ),
        max_tokens=700,
        expect_regex=r"def\s+bsearch",
        notes="Easy to get wrong: leftmost duplicate handling.",
    ),
    Test(
        name="L4-lru-cache",
        difficulty=4,
        system=SYSTEM_CONCISE,
        user=(
            "Implement a Python class LRUCache(capacity: int) with get(key) "
            "and put(key, value), both O(1). Use a doubly-linked list + dict. "
            "No imports. Include a brief docstring."
        ),
        max_tokens=900,
        expect_regex=r"class\s+LRUCache",
        notes="Classic technical interview problem — tests data structure design.",
    ),
    Test(
        name="L5-debug-subtle",
        difficulty=5,
        system=(
            "You are a senior engineer. Explain bugs precisely and provide "
            "the corrected code in a fenced block."
        ),
        user=(
            "This Python function is supposed to return the kth largest "
            "element of a list. It has a subtle bug. Find it and fix it.\n\n"
            "```python\n"
            "def kth_largest(nums, k):\n"
            "    nums.sort()\n"
            "    return nums[-k]\n"
            "```\n\n"
            "First explain what the bug is in 2-3 sentences, then provide a "
            "corrected version that handles all edge cases."
        ),
        max_tokens=900,
        expect_regex=r"def\s+kth_largest",
        notes="Bug: in-place mutation. Tests reasoning + careful review.",
    ),
    Test(
        name="L6-design-rate-limiter",
        difficulty=6,
        system=(
            "You are a senior engineer. Design APIs that are simple to use "
            "and hard to misuse. Output: brief design discussion, then code."
        ),
        user=(
            "Design and implement a Python rate limiter class that supports "
            "token bucket semantics, is thread-safe, and lets the caller "
            "either block until a token is available or get a 'denied' "
            "response immediately. Discuss the API choices in 3-4 sentences "
            "before showing the code. Pay extra attention to lock handling — "
            "do not release a lock manually inside a `with` block."
        ),
        max_tokens=1400,
        expect_regex=r"class\s+\w*RateLimiter|class\s+\w*Bucket|class\s+\w*Limiter",
        notes="Tests synthesis: concurrency + design + clean API.",
    ),
    Test(
        name="L7-agentic-bugfix",
        difficulty=7,
        notes=(
            "Spawn opencode with qwen-coder against a sandbox containing a "
            "buggy file; verify the model uses tools to find and fix the bug."
        ),
        custom_run=run_l7_agentic,
    ),
    Test(
        name="L8-long-context-niah",
        difficulty=8,
        notes=(
            "Needle in a ~64K-token haystack of fake module stubs. Must "
            "locate a unique function and report its return value."
        ),
        custom_run=lambda ctx: run_l8_long_context(ctx, approx_tokens=64_000),
    ),
]


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def render_result(test: Test, result: dict[str, Any]) -> bool:
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"  L{test.difficulty}  {test.name}")
    print(f"  {test.notes}")
    print(bar)
    if "error" in result:
        print(f"  ERROR: {result['error']}  ({result['elapsed_s']:.1f}s)")
        return False
    content = result.get("content", "")
    print(
        f"  {result['elapsed_s']:6.1f}s  "
        f"{result.get('completion_tokens', 0):5d} tok  "
        f"{result.get('tokens_per_s', 0):6.1f} tok/s  "
        f"finish={result.get('finish_reason')}"
    )
    passed = True
    # Custom-run tests provide their own pass/fail.
    if "_l7_passed" in result:
        passed = bool(result["_l7_passed"])
        print(f"  {'✓' if passed else '✗'} L7 sandbox check (file edited & runs correctly)")
    elif "_l8_passed" in result:
        passed = bool(result["_l8_passed"])
        print(f"  {'✓' if passed else '✗'} L8 needle found (haystack {result.get('_l8_haystack_chars',0):,} chars)")
    elif test.expect_regex and not re.search(test.expect_regex, content):
        passed = False
        print(f"  ✗ expected regex /{test.expect_regex}/ not found")
    elif test.expect_substring and test.expect_substring not in content:
        passed = False
        print(f"  ✗ expected substring {test.expect_substring!r} not found")
    else:
        print("  ✓ basic check passed")
    print()
    print("  --- output ---")
    for line in (content or "(no content)").splitlines()[:80]:
        print(f"  {line}")
    if len(content.splitlines()) > 80:
        print(f"  ... [{len(content.splitlines()) - 80} more lines truncated]")
    return passed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--tests",
        default="all",
        help="comma-separated names (e.g. 'L1,L4,L8') or 'all'",
    )
    ap.add_argument(
        "--no-think",
        default="true",
        choices=["true", "false"],
        help="pass enable_thinking=False to the chat template (default: true)",
    )
    args = ap.parse_args()

    no_think = args.no_think == "true"

    if args.tests == "all":
        tests = TESTS
    else:
        wanted = set(args.tests.split(","))
        tests = [
            t for t in TESTS
            if t.name in wanted
            or t.name.split("-", 1)[0] in wanted   # allow "L4" or "L4-lru-cache"
        ]

    print(f"Model: {args.model}")
    print(f"Server: {args.base_url}")
    print(f"Thinking: {'OFF (enable_thinking=false)' if no_think else 'ON (default)'}")
    print(f"Running {len(tests)} test(s).")

    project_dir = Path(__file__).resolve().parent.parent
    ctx = RunContext(
        base_url=args.base_url,
        model=args.model,
        no_think=no_think,
        project_dir=project_dir,
    )

    summary = []
    for t in tests:
        if t.custom_run is not None:
            r = t.custom_run(ctx)
        else:
            r = call_chat(
                base_url=args.base_url,
                model=args.model,
                messages=[
                    {"role": "system", "content": t.system or ""},
                    {"role": "user", "content": t.user or ""},
                ],
                max_tokens=t.max_tokens,
                no_think=no_think,
            )
        passed = render_result(t, r)
        summary.append((t, r, passed))

    print("\n\n========== summary ==========")
    print(f"{'name':<28}  {'sec':>7}  {'tok':>5}  {'tps':>6}  status")
    overall_pass = 0
    for t, r, passed in summary:
        if "error" in r:
            status = "ERROR"
            tok = "-"
            tps = "-"
            sec = f"{r['elapsed_s']:7.1f}"
        else:
            status = "ok " if passed else "FAIL"
            tok = f"{r.get('completion_tokens', 0):5d}"
            tps = f"{r.get('tokens_per_s', 0):6.1f}"
            sec = f"{r['elapsed_s']:7.1f}"
            if passed:
                overall_pass += 1
        print(f"{t.name:<28}  {sec}  {tok}  {tps}  {status}")
    print(f"\n{overall_pass}/{len(summary)} passed")
    sys.exit(0 if overall_pass == len(summary) else 1)


if __name__ == "__main__":
    main()
