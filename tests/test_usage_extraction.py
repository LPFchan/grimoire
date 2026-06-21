import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

from grimoire.proxy.sse import _usage_from_object


class UsageExtractionTests(unittest.TestCase):

    def test_reads_prompt_tokens_details(self):
        out = _usage_from_object({"usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
        }})
        self.assertEqual(out, {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 80})

    def test_reads_input_tokens_details(self):
        out = _usage_from_object({"usage": {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 80},
        }})
        self.assertEqual(out, {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 80})

    def test_no_cache_still_works(self):
        out = _usage_from_object({"usage": {"prompt_tokens": 5, "completion_tokens": 7}})
        self.assertEqual(out, {"input_tokens": 5, "output_tokens": 7})
        self.assertNotIn("cached_tokens", out)

    def test_zero_cached_is_absent(self):
        out = _usage_from_object({"usage": {
            "prompt_tokens": 5, "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 0},
        }})
        self.assertEqual(out, {"input_tokens": 5, "output_tokens": 7})
        self.assertNotIn("cached_tokens", out)

    def test_details_without_cached_key(self):
        out = _usage_from_object({"usage": {
            "prompt_tokens": 5, "completion_tokens": 7,
            "prompt_tokens_details": {},
        }})
        self.assertEqual(out, {"input_tokens": 5, "output_tokens": 7})
        self.assertNotIn("cached_tokens", out)

    def test_does_not_parse_anthropic_shape(self):
        out = _usage_from_object({"usage": {
            "prompt_tokens": 100, "completion_tokens": 20,
            "cache_read_input_tokens": 80,
        }})
        self.assertNotIn("cached_tokens", out)

    def test_flat_cached_tokens_still_supported(self):
        out = _usage_from_object({"usage": {
            "prompt_tokens": 100, "completion_tokens": 20,
            "cached_tokens": 80,
        }})
        self.assertEqual(out["cached_tokens"], 80)

    def test_none_returns_none(self):
        self.assertIsNone(_usage_from_object(None))
        self.assertIsNone(_usage_from_object({}))
        self.assertIsNone(_usage_from_object({"usage": {}}))

    def test_zero_tokens_returns_none(self):
        self.assertIsNone(_usage_from_object({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}))


if __name__ == "__main__":
    unittest.main()