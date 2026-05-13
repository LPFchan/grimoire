import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grimoire.plugins import DFLASH_AWARENESS_MARKER, DflashPflashAwarenessPlugin


class DflashPflashAwarenessPluginTests(unittest.TestCase):
    def setUp(self):
        self.plugin = DflashPflashAwarenessPlugin()
        self.model_cfg = {
            "backend": "dflash",
            "drafter": "gguf/Qwen3-0.6B-BF16.gguf",
            "prefill-compression": "auto",
            "prefill-threshold": 48000,
        }

    def test_injects_runtime_note_for_recall_aware_dflash_requests(self):
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"type": "function", "function": {"name": "conversation_recall"}}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn(DFLASH_AWARENESS_MARKER, result["messages"][0]["content"])
        self.assertIn("48,000+", result["messages"][0]["content"])
        self.assertIn("conversation_recall", result["messages"][0]["content"])
        self.assertEqual(result["messages"][1], payload["messages"][0])

    def test_appends_to_existing_system_message_and_supports_legacy_functions(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "ping"},
            ],
            "functions": [{"name": "conversation_recall"}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("You are terse.", result["messages"][0]["content"])
        self.assertIn(DFLASH_AWARENESS_MARKER, result["messages"][0]["content"])
        self.assertIn("conversation_recall", result["messages"][0]["content"])

    def test_appends_to_existing_system_text_parts_without_second_system(self):
        payload = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
                {"role": "user", "content": "ping"},
            ],
            "tools": [{"type": "function", "function": {"name": "conversation_recall"}}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        self.assertEqual([message["role"] for message in result["messages"]], ["system", "user"])
        self.assertIsInstance(result["messages"][0]["content"], list)
        rendered = "".join(part.get("text", "") for part in result["messages"][0]["content"])
        self.assertIn("You are terse.", rendered)
        self.assertIn(DFLASH_AWARENESS_MARKER, rendered)
        self.assertIn("conversation_recall", rendered)

    def test_skips_requests_without_conversation_recall(self):
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        self.assertEqual(result, payload)

    def test_skips_when_dflash_pflash_is_not_available(self):
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"type": "function", "function": {"name": "conversation_recall"}}],
        }

        for cfg in [
            {"backend": "llama", "drafter": "gguf/Qwen3-0.6B-BF16.gguf", "prefill-compression": "auto"},
            {"backend": "dflash", "drafter": "gguf/Qwen3-0.6B-BF16.gguf", "prefill-compression": "never"},
            {"backend": "dflash", "prefill-compression": "auto"},
        ]:
            with self.subTest(cfg=cfg):
                result = self.plugin.before_request(copy.deepcopy(payload), "model", cfg)
                self.assertEqual(result, payload)

    def test_does_not_duplicate_existing_runtime_note(self):
        payload = {
            "messages": [
                {"role": "system", "content": f"Existing note\n\n{DFLASH_AWARENESS_MARKER} already here"},
                {"role": "user", "content": "ping"},
            ],
            "tools": [{"type": "function", "function": {"name": "conversation_recall"}}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        self.assertEqual(result["messages"][0]["content"].count(DFLASH_AWARENESS_MARKER), 1)

    def test_does_not_duplicate_existing_runtime_note_in_text_parts(self):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"Existing note\n\n{DFLASH_AWARENESS_MARKER} already here"},
                    ],
                },
                {"role": "user", "content": "ping"},
            ],
            "tools": [{"type": "function", "function": {"name": "conversation_recall"}}],
        }

        result = self.plugin.before_request(copy.deepcopy(payload), "dflash-pflash-qwen-27B", dict(self.model_cfg))

        rendered = "".join(part.get("text", "") for part in result["messages"][0]["content"])
        self.assertEqual(rendered.count(DFLASH_AWARENESS_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
