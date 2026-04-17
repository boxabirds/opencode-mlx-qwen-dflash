"""Idempotent patch script: add qwen3_coder tool-call support to dflash-serve.

Usage:
    python patch_serve_py.py <path-to-serve.py> <project-dir>

This:
1. Adds the system-message coalescing fix (Qwen3.5 chat template requirement).
2. Adds an import of qwen_tool_adapter from <project-dir>.
3. Wraps the non-streaming response so qwen3_coder XML tool calls in the
   model output become OpenAI-format tool_calls JSON.
4. Wraps the streaming event loop so tool calls are extracted incrementally
   and emitted as OpenAI streaming tool_call deltas.

Idempotent: looks for marker comments before applying each change and skips
if already present.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCH_MARKER = "# DFLASH_QWEN_TOOL_PATCH_v1"


def patched(src: str) -> bool:
    return PATCH_MARKER in src


def _normalize_messages_block() -> str:
    return '''def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _qta_imports_block(project_dir: str) -> str:
    return f'''
{PATCH_MARKER}
# Import qwen_tool_adapter from the install repo to convert Qwen3.5's
# qwen3_coder XML tool-call format into OpenAI tool_calls JSON.
import sys as _qta_sys
_QTA_PROJECT_DIR = {project_dir!r}
if _QTA_PROJECT_DIR not in _qta_sys.path:
    _qta_sys.path.insert(0, _QTA_PROJECT_DIR)
try:
    from qwen_tool_adapter import (
        parse_tool_calls as _qta_parse,
        StreamingToolCallExtractor as _QtaStreamExtractor,
        has_tool_call as _qta_has_tool_call,
    )
    _QTA_AVAILABLE = True
except Exception as _qta_exc:
    import sys as _qta_sys2
    _qta_sys2.stderr.write(
        f"warning: qwen_tool_adapter not importable from {{_QTA_PROJECT_DIR!r}} "
        f"({{_qta_exc}}). Tool calls will be returned as raw text.\\n"
    )
    _QTA_AVAILABLE = False


def _qta_make_response_with_tools(
    *,
    response_id,
    created,
    model_ref,
    text,
    summary,
    tools,
):
    """Build a chat.completion response, lifting any qwen3_coder tool calls
    in `text` into OpenAI-format tool_calls."""
    preamble = None
    tool_calls = []
    if _QTA_AVAILABLE and tools and _qta_has_tool_call(text):
        preamble, tool_calls = _qta_parse(text, tools=tools)
    if tool_calls:
        message = {{
            "role": "assistant",
            "content": preamble,
            "tool_calls": tool_calls,
        }}
        finish_reason = "tool_calls"
    else:
        message = {{"role": "assistant", "content": text}}
        finish_reason = "stop"
    return {{
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_ref,
        "choices": [
            {{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }}
        ],
        "usage": _usage_from_summary(summary),
    }}
'''


# The replacement for the streaming token-emit loop. The original is the
# `for event in event_iter:` block that writes one SSE chunk per token.
_STREAM_LOOP_OLD = '''                for event in event_iter:
                    if event.get("event") == "token":
                        text = decode_token(state.tokenizer, int(event["token_id"]))
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [
                                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                ],
                            },
                        )
                    elif event.get("event") == "summary":
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            },
                        )'''

_STREAM_LOOP_NEW = '''                # qwen3_coder tool-call adapter: feed each token through the
                # streaming extractor; emit content deltas for plain text and
                # tool_calls deltas for parsed tool calls.
                _qta_extractor = (
                    _QtaStreamExtractor(tools=tools)
                    if (_QTA_AVAILABLE and tools)
                    else None
                )
                _qta_finish_reason = "stop"

                def _qta_emit(evt):
                    nonlocal _qta_finish_reason
                    if "content" in evt and evt["content"]:
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [
                                    {"index": 0, "delta": {"content": evt["content"]}, "finish_reason": None}
                                ],
                            },
                        )
                    elif "tool_call" in evt:
                        _qta_finish_reason = "tool_calls"
                        tc = evt["tool_call"]
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": evt["index"],
                                                    "id": tc["id"],
                                                    "type": "function",
                                                    "function": tc["function"],
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            },
                        )

                for event in event_iter:
                    if event.get("event") == "token":
                        token_id = int(event["token_id"])
                        # Skip stop/EOS tokens (e.g. <|im_end|>) that leak
                        # through the event iter — they shouldn't be content.
                        if token_id in stop_token_ids:
                            continue
                        text = decode_token(state.tokenizer, token_id)
                        if _qta_extractor is None:
                            _sse_write(
                                self,
                                {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": state.model_ref,
                                    "choices": [
                                        {"index": 0, "delta": {"content": text}, "finish_reason": None}
                                    ],
                                },
                            )
                        else:
                            for _qta_evt in _qta_extractor.feed(text):
                                _qta_emit(_qta_evt)
                    elif event.get("event") == "summary":
                        if _qta_extractor is not None:
                            for _qta_evt in _qta_extractor.finish():
                                _qta_emit(_qta_evt)
                        _sse_write(
                            self,
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": state.model_ref,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": _qta_finish_reason}],
                            },
                        )'''


# In do_POST, capture tools from payload right after parsing it.
_PAYLOAD_OLD = '''            payload = self._read_json()'''
_PAYLOAD_NEW = '''            payload = self._read_json()
            tools = payload.get("tools") or None  # forwarded to qwen tool adapter'''


# Replace the non-streaming response builder.
_RESPONSE_OLD = '''            self._send_json(
                200,
                _make_chat_response(
                    response_id=response_id,
                    created=created,
                    model_ref=state.model_ref,
                    text=text,
                    summary=summary,
                ),
            )'''
_RESPONSE_NEW = '''            self._send_json(
                200,
                _qta_make_response_with_tools(
                    response_id=response_id,
                    created=created,
                    model_ref=state.model_ref,
                    text=text,
                    summary=summary,
                    tools=tools,
                ),
            )'''


# The system-message anchor for inserting _normalize_messages.
_NORM_ANCHOR = "def _build_prompt_request("
_NORM_OLD = 'messages = list(payload.get("messages") or [])'
_NORM_NEW = "messages = _normalize_messages(list(payload.get('messages') or []))"

# Thread `tools` through to apply_chat_template so Qwen3.5's template can
# inject the qwen3_coder XML format instructions into the prompt.
_TOOLS_ARG_OLD = '''        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return "", list(prompt_tokens)'''
_TOOLS_ARG_NEW = '''        # Pass tools through to the chat template. Qwen3.5's template uses
        # them to inject the qwen3_coder XML format instructions; without
        # this the model improvises and emits unparseable XML.
        tools = payload.get("tools") or None
        # vLLM-style chat_template_kwargs: lets clients pass model-specific
        # template flags. We DEFAULT enable_thinking=False because clients
        # like opencode don't expose chat_template_kwargs and Qwen3.5's
        # default thinking-mode dump eats tokens AND blows past typical
        # client request timeouts. Opt back in: {"enable_thinking": true}.
        chat_template_kwargs = payload.get("chat_template_kwargs") or {}
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            kwargs = {"tokenize": True, "add_generation_prompt": True}
            if tools:
                kwargs["tools"] = tools
            template_kwargs = {"enable_thinking": False}
            if isinstance(chat_template_kwargs, dict):
                template_kwargs.update(chat_template_kwargs)
            kwargs.update(template_kwargs)
            prompt_tokens = tokenizer.apply_chat_template(messages, **kwargs)
            return "", list(prompt_tokens)'''


def apply(src: str, project_dir: str) -> str:
    if patched(src):
        return src

    out = src

    # 1. Insert _normalize_messages function and rewrite the messages line.
    if "_normalize_messages" not in out:
        if _NORM_ANCHOR not in out:
            raise SystemExit(f"anchor not found: {_NORM_ANCHOR!r}")
        idx = out.index(_NORM_ANCHOR)
        out = out[:idx] + _normalize_messages_block() + out[idx:]
        if _NORM_OLD not in out:
            raise SystemExit(f"anchor not found: {_NORM_OLD!r}")
        out = out.replace(_NORM_OLD, _NORM_NEW, 1)

    # 1b. Thread `tools` through apply_chat_template.
    if _TOOLS_ARG_OLD not in out:
        raise SystemExit("anchor for tools-in-chat-template not found")
    out = out.replace(_TOOLS_ARG_OLD, _TOOLS_ARG_NEW, 1)

    # 2. Insert qta imports and helper function near the top (after the
    # existing _make_chat_response so _usage_from_summary is defined).
    if "_make_chat_response" not in out:
        raise SystemExit("anchor _make_chat_response not found")
    insert_after = "def _sse_write("
    if insert_after not in out:
        raise SystemExit(f"anchor not found: {insert_after!r}")
    insert_idx = out.index(insert_after)
    out = out[:insert_idx] + _qta_imports_block(project_dir) + "\n\n" + out[insert_idx:]

    # 3. Capture tools from payload.
    if _PAYLOAD_OLD not in out:
        raise SystemExit("anchor for payload extraction not found")
    out = out.replace(_PAYLOAD_OLD, _PAYLOAD_NEW, 1)

    # 4. Replace the non-streaming response builder.
    if _RESPONSE_OLD not in out:
        raise SystemExit("anchor for non-streaming response not found")
    out = out.replace(_RESPONSE_OLD, _RESPONSE_NEW, 1)

    # 5. Replace the streaming event loop.
    if _STREAM_LOOP_OLD not in out:
        raise SystemExit("anchor for streaming event loop not found")
    out = out.replace(_STREAM_LOOP_OLD, _STREAM_LOOP_NEW, 1)

    # 6. Connection: close for streaming responses. Without this, HTTP/1.1
    # keep-alive keeps the socket open after data: [DONE] and the client
    # (opencode, curl) waits indefinitely for more data.
    _KEEPALIVE = 'self.send_header("Connection", "keep-alive")'
    _CLOSE = (
        '# Connection: close so client knows we\'re done after data: [DONE].\n'
        '                self.send_header("Connection", "close")'
    )
    if _KEEPALIVE in out:
        out = out.replace(_KEEPALIVE, _CLOSE, 1)

    return out


def main():
    if len(sys.argv) != 3:
        print("usage: patch_serve_py.py <path-to-serve.py> <project-dir>",
              file=sys.stderr)
        sys.exit(2)
    serve_py = Path(sys.argv[1])
    project_dir = sys.argv[2]
    src = serve_py.read_text()
    if patched(src):
        print("already patched (idempotent).")
        return
    new_src = apply(src, project_dir)
    # Sanity: must compile.
    compile(new_src, str(serve_py), "exec")
    serve_py.write_text(new_src)
    print(f"patched {serve_py}")


if __name__ == "__main__":
    main()
