"""Unit tests for qwen_tool_adapter.

Run with: uv run pytest test_qwen_tool_adapter.py -v
Or: uv run python -m unittest test_qwen_tool_adapter
"""

import json
import unittest

from qwen_tool_adapter import (
    StreamingToolCallExtractor,
    has_tool_call,
    parse_tool_calls,
)

# A representative tool spec, the kind opencode would send.
READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}


class HasToolCallTests(unittest.TestCase):
    def test_negative(self):
        self.assertFalse(has_tool_call("Hello world."))
        self.assertFalse(has_tool_call(""))

    def test_positive(self):
        self.assertTrue(has_tool_call("ok <tool_call> something"))


class ParseToolCallsTests(unittest.TestCase):
    def test_no_tool_call_returns_none(self):
        preamble, calls = parse_tool_calls("just a text response")
        self.assertIsNone(preamble)
        self.assertEqual(calls, [])

    def test_single_call_with_string_param(self):
        text = (
            "I'll read the hostname file.\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            "<parameter=path>/etc/hostname</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        preamble, calls = parse_tool_calls(text, tools=[READ_FILE_TOOL])
        self.assertEqual(preamble, "I'll read the hostname file.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args, {"path": "/etc/hostname"})
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_typed_param_coercion(self):
        text = (
            "<tool_call>"
            "<function=read_file>"
            "<parameter=path>/foo</parameter>"
            "<parameter=max_lines>42</parameter>"
            "</function>"
            "</tool_call>"
        )
        _, calls = parse_tool_calls(text, tools=[READ_FILE_TOOL])
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["max_lines"], 42)
        self.assertIsInstance(args["max_lines"], int)

    def test_no_schema_falls_back_to_smart_guess(self):
        text = (
            "<tool_call><function=foo>"
            "<parameter=n>7</parameter>"
            "<parameter=flag>true</parameter>"
            "<parameter=label>hello</parameter>"
            "</function></tool_call>"
        )
        _, calls = parse_tool_calls(text)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["n"], 7)
        self.assertIs(args["flag"], True)
        self.assertEqual(args["label"], "hello")

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>/a</parameter></function></tool_call>\n"
            "<tool_call><function=read_file>"
            "<parameter=path>/b</parameter></function></tool_call>"
        )
        _, calls = parse_tool_calls(text, tools=[READ_FILE_TOOL])
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["path"], "/a")
        self.assertEqual(json.loads(calls[1]["function"]["arguments"])["path"], "/b")
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_object_param_decoded_as_dict(self):
        OBJ_TOOL = {
            "type": "function",
            "function": {
                "name": "create",
                "parameters": {
                    "type": "object",
                    "properties": {"meta": {"type": "object"}},
                },
            },
        }
        text = (
            '<tool_call><function=create>'
            '<parameter=meta>{"k": 1}</parameter>'
            "</function></tool_call>"
        )
        _, calls = parse_tool_calls(text, tools=[OBJ_TOOL])
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["meta"], {"k": 1})

    def test_call_with_no_preamble(self):
        text = (
            "<tool_call><function=read_file>"
            "<parameter=path>/x</parameter></function></tool_call>"
        )
        preamble, calls = parse_tool_calls(text)
        self.assertIsNone(preamble)
        self.assertEqual(len(calls), 1)


class StreamingExtractorTests(unittest.TestCase):
    """Exercise the extractor with realistic chunking patterns."""

    def _drain(self, extractor, chunks):
        """Feed chunks one at a time, return list of all events."""
        events = []
        for c in chunks:
            events.extend(extractor.feed(c))
        events.extend(extractor.finish())
        return events

    def test_pure_content_no_tool(self):
        ex = StreamingToolCallExtractor()
        events = self._drain(ex, ["Hello ", "world", "!"])
        joined = "".join(e["content"] for e in events if "content" in e)
        self.assertEqual(joined, "Hello world!")
        self.assertFalse(ex.emitted_any_tool_call)

    def test_complete_tool_call_in_one_chunk(self):
        ex = StreamingToolCallExtractor(tools=[READ_FILE_TOOL])
        chunks = [
            "Reading file now.\n"
            "<tool_call><function=read_file>"
            "<parameter=path>/etc/hostname</parameter>"
            "</function></tool_call>"
        ]
        events = self._drain(ex, chunks)
        contents = [e["content"] for e in events if "content" in e]
        tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
        self.assertIn("Reading file now.", "".join(contents))
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "read_file")

    def test_tool_call_split_across_many_chunks(self):
        # Hostile chunking: split inside markers and inside parameter values.
        ex = StreamingToolCallExtractor(tools=[READ_FILE_TOOL])
        full = (
            "<tool_call>"
            "<function=read_file>"
            "<parameter=path>/etc/hostname</parameter>"
            "</function>"
            "</tool_call>"
        )
        # Split into 3-char chunks.
        chunks = [full[i : i + 3] for i in range(0, len(full), 3)]
        events = self._drain(ex, chunks)
        contents = [e["content"] for e in events if "content" in e]
        tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
        # No content should leak before/within the tool call.
        self.assertEqual("".join(contents), "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"path": "/etc/hostname"},
        )

    def test_content_then_tool_call_with_split_marker(self):
        # The split should NOT leak the start of "<tool_call>" into content.
        ex = StreamingToolCallExtractor(tools=[READ_FILE_TOOL])
        full = (
            "I will call: <tool_call><function=read_file>"
            "<parameter=path>/x</parameter></function></tool_call>"
        )
        chunks = []
        # Split right inside the "<tool_call>" marker.
        marker_idx = full.find("<tool_call>")
        chunks.append(full[: marker_idx + 5])  # up to "<tool"
        chunks.append(full[marker_idx + 5 :])
        events = self._drain(ex, chunks)
        contents = "".join(e["content"] for e in events if "content" in e)
        self.assertEqual(contents, "I will call: ")
        tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
        self.assertEqual(len(tool_calls), 1)

    def test_two_tool_calls_in_stream(self):
        ex = StreamingToolCallExtractor(tools=[READ_FILE_TOOL])
        full = (
            "<tool_call><function=read_file>"
            "<parameter=path>/a</parameter></function></tool_call>"
            "<tool_call><function=read_file>"
            "<parameter=path>/b</parameter></function></tool_call>"
        )
        events = self._drain(ex, [full])
        tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(events[0]["index"], 0)
        self.assertEqual(events[1]["index"], 1)

    def test_truncated_tool_call_at_eof_is_best_effort(self):
        ex = StreamingToolCallExtractor(tools=[READ_FILE_TOOL])
        # Stream cut off before </tool_call>. Best-effort: parse with
        # synthetic close so opencode at least sees the partial call.
        chunks = [
            "<tool_call><function=read_file>"
            "<parameter=path>/etc/hostname</parameter>"
            "</function>"
            # NOTE: no </tool_call>
        ]
        events = self._drain(ex, chunks)
        tool_calls = [e["tool_call"] for e in events if "tool_call" in e]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"path": "/etc/hostname"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
