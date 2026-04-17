"""qwen3_coder ↔ OpenAI tool-call adapter.

Qwen3.5 emits tool calls in its proprietary qwen3_coder XML format:

    <tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    <parameter=other_param>123</parameter>
    </function>
    </tool_call>

OpenAI clients (including opencode) expect:

    {
      "choices": [{
        "message": {
          "role": "assistant",
          "content": null,
          "tool_calls": [
            {
              "id": "call_<uuid>",
              "type": "function",
              "function": {
                "name": "function_name",
                "arguments": "{\"param_name\": \"value\", \"other_param\": 123}"
              }
            }
          ]
        },
        "finish_reason": "tool_calls"
      }]
    }

This module converts between the two for both non-streaming (one-shot) and
streaming (SSE delta) responses.

Reference: vLLM's qwen3coder_tool_parser.py.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator, Optional

# -----------------------------------------------------------------------------
# Markers
# -----------------------------------------------------------------------------
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN_PREFIX = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN_PREFIX = "<parameter="
PARAMETER_CLOSE = "</parameter>"

# -----------------------------------------------------------------------------
# Non-streaming parser: full text in, OpenAI dict out
# -----------------------------------------------------------------------------

# Match a complete <tool_call>...</tool_call> block (DOTALL so newlines match).
_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
# Inside a tool_call: <function=NAME>BODY</function>
_FUNCTION_RE = re.compile(
    re.escape(FUNCTION_OPEN_PREFIX) + r"(.*?)>(.*?)" + re.escape(FUNCTION_CLOSE),
    re.DOTALL,
)
# Inside a function body: <parameter=NAME>VALUE</parameter>
_PARAMETER_RE = re.compile(
    re.escape(PARAMETER_OPEN_PREFIX) + r"(.*?)>(.*?)" + re.escape(PARAMETER_CLOSE),
    re.DOTALL,
)


def _coerce_param_value(raw: str, schema_type: Optional[str]) -> Any:
    """Convert qwen-emitted string parameter into the JSON-schema-typed value.

    Qwen emits everything as text inside <parameter=...>...</parameter>. The
    OpenAI tool spec expects properly typed JSON. If the tool schema declares
    the parameter type we coerce; otherwise we try a best-effort guess (numeric
    if it parses, JSON-decoded if it parses as a JSON literal, else string).
    """
    text = raw.strip()
    if schema_type:
        st = schema_type.lower()
        if st in {"string", "str", "text"}:
            return text
        if st in {"integer", "int"}:
            try:
                return int(text)
            except ValueError:
                return text
        if st in {"number", "float", "double"}:
            try:
                num = float(text)
                return int(num) if num.is_integer() else num
            except ValueError:
                return text
        if st in {"boolean", "bool"}:
            return text.lower() in {"true", "1", "yes"}
        if st in {"object", "dict", "array", "list"}:
            try:
                return json.loads(text)
            except (ValueError, json.JSONDecodeError):
                return text
        if st == "null":
            return None
    # No schema hint: try JSON literal, then number, else raw string.
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text


def _build_param_type_lookup(
    tools: Optional[list[dict[str, Any]]],
) -> dict[str, dict[str, str]]:
    """Map function name → {param name → JSON-schema type}."""
    lookup: dict[str, dict[str, str]] = {}
    if not tools:
        return lookup
    for tool in tools:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        params = (fn.get("parameters") or {}).get("properties") or {}
        lookup[name] = {
            pname: (pdef or {}).get("type", "")
            for pname, pdef in params.items()
            if isinstance(pname, str)
        }
    return lookup


def has_tool_call(text: str) -> bool:
    """Cheap pre-check before paying for full parse."""
    return TOOL_CALL_OPEN in text


def parse_tool_calls(
    text: str,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Split a complete model output into (preamble_text, tool_calls).

    Returns:
        preamble_text: any text the model produced *before* the first
            <tool_call>. Often the model's chain-of-thought / "I'll call X
            because..." narration. None if no tool call was emitted.
        tool_calls: list of OpenAI-format tool call dicts. Empty if no tool
            call was emitted.
    """
    if not has_tool_call(text):
        return None, []

    type_lookup = _build_param_type_lookup(tools)
    preamble_idx = text.find(TOOL_CALL_OPEN)
    preamble = text[:preamble_idx].rstrip() if preamble_idx > 0 else ""

    tool_calls: list[dict[str, Any]] = []
    for tc_match in _TOOL_CALL_BLOCK_RE.finditer(text):
        inner = tc_match.group(1)
        fn_match = _FUNCTION_RE.search(inner)
        if not fn_match:
            continue
        fn_name = fn_match.group(1).strip()
        fn_body = fn_match.group(2)
        param_types = type_lookup.get(fn_name, {})

        args: dict[str, Any] = {}
        for p_match in _PARAMETER_RE.finditer(fn_body):
            p_name = p_match.group(1).strip()
            p_value = p_match.group(2)
            args[p_name] = _coerce_param_value(p_value, param_types.get(p_name))

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )

    return (preamble or None), tool_calls


# -----------------------------------------------------------------------------
# Streaming parser: feed text chunks, get OpenAI delta dicts
# -----------------------------------------------------------------------------

# Streaming strategy:
#   We accumulate model output. We track a "mode" — either content (forwarding
#   plain text to client) or tool_call (buffering until we have a complete
#   <tool_call>...</tool_call> block, then emit the full tool_calls delta).
#
#   This is a simpler / chunkier strategy than vLLM's per-parameter streaming
#   parser. We emit one tool_calls delta per complete <tool_call> block. That's
#   functionally fine for opencode — it just waits for the full call before
#   executing it anyway. Per-parameter streaming would be a UX win (showing
#   args populating live) but adds significant state-machine complexity.


class StreamingToolCallExtractor:
    """Stateful streaming text → (content_chunks, tool_call_chunks).

    Usage:
        extractor = StreamingToolCallExtractor(tools=request_tools)
        for token_text in model_stream():
            for evt in extractor.feed(token_text):
                yield evt   # one of: {"content": str} or {"tool_call": dict}
        for evt in extractor.finish():
            yield evt
    """

    def __init__(self, *, tools: Optional[list[dict[str, Any]]] = None):
        self._buffer = ""
        self._in_tool_call = False
        self._type_lookup = _build_param_type_lookup(tools)
        self._tools = tools
        self._tool_index = 0

    def feed(self, chunk: str) -> Iterator[dict[str, Any]]:
        """Feed a token / chunk of model output. Yield events to forward."""
        self._buffer += chunk
        # Drain as much as we can from the buffer.
        while True:
            event = self._step()
            if event is None:
                return
            yield event

    def _step(self) -> Optional[dict[str, Any]]:
        if not self._in_tool_call:
            # Look for the start of a tool call. Until we see one (or are
            # certain we won't see one in the next few chars), we forward
            # text to the client as content.
            idx = self._buffer.find(TOOL_CALL_OPEN)
            if idx == -1:
                # No tool call start in buffer. We can safely forward
                # anything that *can't* possibly be the start of a marker.
                # The longest marker prefix we'd otherwise truncate is
                # len(TOOL_CALL_OPEN) - 1, so hold back that many chars.
                safe_len = max(0, len(self._buffer) - (len(TOOL_CALL_OPEN) - 1))
                if safe_len <= 0:
                    return None
                # Don't emit if the held-back tail could be the start of marker.
                emit = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                if not emit:
                    return None
                return {"content": emit}
            # idx >= 0: we have <tool_call>. Emit any preamble text first.
            if idx > 0:
                preamble = self._buffer[:idx]
                self._buffer = self._buffer[idx:]
                return {"content": preamble}
            # Buffer starts with <tool_call>. Switch to tool-call mode.
            self._in_tool_call = True
            return self._step()
        # In tool-call mode: wait for </tool_call>.
        end_idx = self._buffer.find(TOOL_CALL_CLOSE)
        if end_idx == -1:
            return None
        block_end = end_idx + len(TOOL_CALL_CLOSE)
        block_text = self._buffer[:block_end]
        self._buffer = self._buffer[block_end:]
        self._in_tool_call = False
        # Parse the single block we just consumed.
        _, tool_calls = parse_tool_calls(block_text, tools=self._tools)
        if not tool_calls:
            # Malformed — treat the raw text as content so the user sees it.
            return {"content": block_text}
        # Re-index to keep a global tool index.
        for tc in tool_calls:
            tc_event = {"tool_call": tc, "index": self._tool_index}
            self._tool_index += 1
            return tc_event
        return None

    def finish(self) -> Iterator[dict[str, Any]]:
        """Flush anything remaining in the buffer at end-of-stream."""
        if self._in_tool_call:
            # Stream ended mid tool-call. Best-effort parse what we have.
            _, tool_calls = parse_tool_calls(
                self._buffer + TOOL_CALL_CLOSE, tools=self._tools
            )
            self._buffer = ""
            self._in_tool_call = False
            for tc in tool_calls:
                yield {"tool_call": tc, "index": self._tool_index}
                self._tool_index += 1
            return
        if self._buffer:
            yield {"content": self._buffer}
            self._buffer = ""

    @property
    def emitted_any_tool_call(self) -> bool:
        return self._tool_index > 0
