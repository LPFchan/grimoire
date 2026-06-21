import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

from grimoire.plugins.tool_arg_sanitize import ToolArgSanitizePlugin, _try_repair_json_object


def _msg(arguments):
    return {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "do_thing", "arguments": arguments}}],
    }


def _args(payload):
    return payload["messages"][0]["tool_calls"][0]["function"]["arguments"]


class ToolArgSanitizeTests(unittest.TestCase):

    def setUp(self):
        self.p = ToolArgSanitizePlugin()

    def run_plugin(self, arguments):
        payload = {"messages": [_msg(arguments)]}
        return self.p.before_request(payload, "qwen3.6-mtp-27B", {"family": "qwen"})

    def test_bare_open_brace_normalized(self):
        out = self.run_plugin("{")
        self.assertEqual(_args(out), "{}")

    def test_empty_string_normalized(self):
        out = self.run_plugin("")
        self.assertEqual(_args(out), "{}")

    def test_valid_json_untouched(self):
        out = self.run_plugin('{"path": "/etc/hosts"}')
        self.assertEqual(json.loads(_args(out)), {"path": "/etc/hosts"})

    def test_recoverable_missing_brace_repaired(self):
        out = self.run_plugin('{"a": 1, "b": 2')
        self.assertEqual(json.loads(_args(out)), {"a": 1, "b": 2})

    def test_unterminated_string_normalized(self):
        # truncated mid-string-literal — not safely recoverable
        out = self.run_plugin('{"path": "/home/user/ba')
        self.assertEqual(_args(out), "{}")
        # whatever we emit must itself be valid JSON
        json.loads(_args(out))

    def test_non_string_arguments_untouched(self):
        payload = {"messages": [_msg({"already": "object"})]}
        out = self.p.before_request(payload, "m", {})
        self.assertEqual(_args(out), {"already": "object"})

    def test_message_without_tool_calls_is_noop(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = self.p.before_request(payload, "m", {})
        self.assertEqual(out, payload)

    def test_disabled_is_passthrough(self):
        self.p.set_enabled(False)
        out = self.run_plugin("{")
        self.assertEqual(_args(out), "{")

    def test_repair_helper_nested(self):
        self.assertEqual(_try_repair_json_object('{"a": [1, 2'), {"a": [1, 2]})
        self.assertIsNone(_try_repair_json_object('{"a": "open'))
        self.assertIsNone(_try_repair_json_object(""))


if __name__ == "__main__":
    unittest.main()
