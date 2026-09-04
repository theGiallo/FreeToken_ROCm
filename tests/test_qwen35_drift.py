#!/usr/bin/env python3
"""CPU-only unit test: qwen35 drift-dialect tool-call parsing.

Replays the real drift strings captured in results/sweep_tools_20260827.csv
through FunctionCallParser (no server, no GPU) and asserts the tolerant
Qwen3CoderDetector extracts the tool call the model actually emitted.
"""
from __future__ import annotations

import io
import os
import sys
import json
import types
import importlib.util
import unittest

_PYROOT = os.path.join(os.path.dirname(__file__), "..", "python")
_SERVER = os.path.join(_PYROOT, "freetoken", "server")


_SNIPPED = [
    "freetoken",
    "freetoken.server",
    "freetoken.server.api_models",
    "freetoken.server.reasoning_parser",
    "freetoken.server.function_call_parser",
]


def _leaf_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if name not in _SNIPPED:
        _SNIPPED.append(name)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_freetoken_without_torch():
    pkg = types.ModuleType("freetoken")
    pkg.__path__ = [os.path.join(_PYROOT, "freetoken")]
    pkg.__package__ = "freetoken"
    sys.modules["freetoken"] = pkg

    server = types.ModuleType("freetoken.server")
    server.__path__ = [_SERVER]
    server.__package__ = "freetoken.server"
    sys.modules["freetoken.server"] = server

    api_models = _leaf_module("freetoken.server.api_models",
                              os.path.join(_SERVER, "api_models.py"))
    _leaf_module("freetoken.server.reasoning_parser",
                 os.path.join(_SERVER, "reasoning_parser.py"))
    fcp = _leaf_module("freetoken.server.function_call_parser",
                       os.path.join(_SERVER, "function_call_parser.py"))
    return api_models, fcp


_AM, _FCP = _load_freetoken_without_torch()

# Un-stub sys.modules so these test modules don't shadow the real freetoken packages for
# other test files collected in the same pytest process. The stub objects we need are
# already bound to module-level names below, so restoring is safe here.
_names = {}
for _n in _SNIPPED:
    _names[_n] = sys.modules.pop(_n, None)


def _restore(names):
    for _n, _m in names.items():
        if _m is not None:
            sys.modules[_n] = _m


_restore(_names)
del _names, _restore

FunctionCallParser = _FCP.FunctionCallParser
Qwen3CoderDetector = _FCP.Qwen3CoderDetector
Tool = _AM.Tool
Function = _AM.Function

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="bash",
            description="Run a bash command and return its output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"}
                },
                "required": ["command"],
            },
        ),
    ),
    Tool(
        type="function",
        function=Function(
            name="write",
            description="Write text content to a file at the given path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "Text to write"},
                },
                "required": ["path", "content"],
            },
        ),
    ),
]


def parse_one(text: str) -> tuple[str, list]:
    parser = FunctionCallParser(TOOLS, "qwen35")
    res = parser.parse_non_stream(text)
    return res.normal_text, [(c.name, c.parameters) for c in res.calls]


class Qwen35DriftTest(unittest.TestCase):
    def assert_calls(self, text, expected, normal_expected=""):
        normal, calls = parse_one(text)
        self.assertEqual(normal, normal_expected, f"normal_text for {text!r}")
        self.assertEqual(calls, expected, f"calls for {text!r}")

    # ---- real samples from the sweep CSV ----
    def test_bare_tag(self):
        self.assert_calls('<bash>\n<command>ls /tmp</command>\n</bash>',
                          [("bash", '{"command": "ls /tmp"}')])

    def test_bare_tag_multiline_value(self):
        self.assert_calls('<bash>\n<command>\nls /tmp\n</command>\n</bash>',
                          [("bash", '{"command": "ls /tmp"}')])

    def test_parameter_tags(self):
        self.assert_calls('<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>',
                          [("bash", '{"command": "ls /tmp"}')])

    def test_parameter_tags_double_call(self):
        text = ('<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n'
                '<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>')
        exp = [("bash", '{"command": "ls /tmp"}'), ("bash", '{"command": "ls /tmp"}')]
        self.assert_calls(text, exp)

    def test_parameter_tags_stray_close(self):
        self.assert_calls('<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</tool_call>',
                          [("bash", '{"command": "ls /tmp"}')])

    def test_tool_call_tag(self):
        self.assert_calls('<tool_call>\n<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</tool_call>',
                          [("bash", '{"command": "ls /tmp"}')])

    def test_tool_call_tag_with_lead_prose(self):
        self.assert_calls(
            "I'll use the `ls` command to list the files in the current directory.\n\n"
            "<tool_call>\n<bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
            [("bash", '{"command": "ls -la"}')],
            normal_expected="I'll use the `ls` command to list the files in the current directory.",
        )

    def test_canonical_still_works(self):
        # Pre-existing behavior: wrapperless canonical text parses the call; the
        # +<function=...>+ markup also stays in normal_text (no regression).
        normal, calls = parse_one(
            '<function=bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>'
        )
        self.assertEqual(calls, [("bash", '{"command": "ls /tmp"}')])
        self.assertIn("<function=bash>", normal)

    def test_canonical_with_wrapper_still_works(self):
        self.assert_calls(
            '<tool_call>\n<function=bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</tool_call>',
            [("bash", '{"command": "ls /tmp"}')])

    def test_write_drift_multi_param(self):
        text = ('<write>\n<parameter=path>\n/tmp/out.txt\n</parameter>\n'
                '<parameter=content>\nhello\n</parameter>\n</function>')
        self.assert_calls(text, [("write", '{"path": "/tmp/out.txt", "content": "hello"}')])

    def test_prose_is_not_mangled(self):
        normal, calls = parse_one("Just answer: the file list is in /tmp. No tags here.")
        self.assertEqual(normal, "Just answer: the file list is in /tmp. No tags here.")
        self.assertEqual(calls, [])

    def test_has_tool_call_drift(self):
        det = Qwen3CoderDetector()
        for s in (
            "<bash>\n<command>ls /tmp</command>\n</bash>",
            "<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>",
        ):
            self.assertTrue(det.has_tool_call(s), s)
        self.assertFalse(det.has_tool_call("Just a plain sentence."))

    def test_normalizer_is_idempotent(self):
        det = Qwen3CoderDetector()
        src = "<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>"
        once = det._normalize_qwen35_drift(src, TOOLS)
        twice = det._normalize_qwen35_drift(once, TOOLS)
        self.assertEqual(once, twice)

    def test_call_tool_wrapper_parses_without_leaking_markup(self):
        # The duck_hunt model emitted <call_tool> (not <tool_call>). It must
        # parse into a call AND not leak the raw XML as normal_text.
        text = ("Let me check the file.\n\n"
                "<call_tool>\n"
                " <function=read>\n"
                " <parameter=limit>\n 25\n </parameter>\n"
                " <parameter=offset>\n 805\n </parameter>\n"
                " <parameter=path>\n /workspace/duck_hunt/duck_hunt.py\n </parameter>\n"
                " </function>\n"
                "</call_tool>")
        normal, calls = parse_one(text)
        self.assertEqual(normal, "Let me check the file.")
        self.assertEqual([c[0] for c in calls], ["read"])
        self.assertNotIn("call_tool", normal)
        self.assertNotIn("<function", normal)

    def test_call_tool_wrapper_bash_parses_and_strips(self):
        text = ("I'll check the package.\n\n"
                "<call_tool>\n"
                " <function=bash>\n"
                " <parameter=command>\n pip show windows-curses\n </parameter>\n"
                " </function>\n"
                "</call_tool>")
        normal, calls = parse_one(text)
        self.assertEqual(normal, "I'll check the package.")
        self.assertEqual(calls, [("bash", '{"command": " pip show windows-curses\\n "}')])
        self.assertNotIn("call_tool", normal)
        self.assertNotIn("<function", normal)

    def test_has_tool_call_recognizes_call_tool(self):
        det = Qwen3CoderDetector()
        self.assertTrue(det.has_tool_call("<call_tool>"))
        self.assertTrue(det.has_tool_call("<call_tool>\n<function=read>\n</call_tool>"))

    def test_call_tool_streaming_parses(self):
        text = ("Let me read it.\n\n<call_tool>\n<function=read>\n"
                "<parameter=path>\n/workspace/duck_hunt/duck_hunt.py\n</parameter>\n"
                "</function>\n</call_tool>")
        det = Qwen3CoderDetector()
        res = det.parse_streaming_increment(text, TOOLS)
        finish = det.finish_streaming()
        self.assertEqual((res.normal_text + finish).strip(), "Let me read it.")
        self.assertTrue(res.calls or det._buffer == "")

    def test_generation_gate_scenario(self):
        # What the sweep measured: drift content with finish_reason=stop.
        for s in (
            "<bash>\n<command>ls /tmp</command>\n</bash>",
            "<bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>",
        ):
            parser = FunctionCallParser(TOOLS, "qwen35")
            self.assertTrue(parser.has_tool_call(s), s)
            result = parser.parse_non_stream(s)
            self.assertTrue(result.calls, s)


if __name__ == "__main__":
    unittest.main(verbosity=2)