"""Conversation KV slots.

Two failure modes, both silent — nothing errors, the cache is just wrong or
missed:

1. The cache is saved and restored on one specific slot. llama-server assigns
   any idle slot unless the request names one, so on a model running more than
   one slot the restore would land somewhere the request never runs.
2. A restarted backend has empty slots, but the marker naming the resident
   conversation outlives the process, so the next request for it skips a restore
   it needed.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grimoire.proxy import llama as llama_proxy


class ConversationSlotPinningTests(unittest.TestCase):
    def test_conversation_requests_are_pinned_to_the_saved_slot(self):
        """The request must name the slot the cache was restored into."""
        source = (ROOT / "src" / "grimoire" / "proxy" / "llama.py").read_text()
        self.assertIn('payload["id_slot"] = CONVERSATION_SLOT', source)
        self.assertIn('f"http://127.0.0.1:{active.port}/slots/{CONVERSATION_SLOT}"', source)

    def test_pin_and_save_target_the_same_slot(self):
        """A constant, so the two can never drift apart."""
        self.assertIsInstance(llama_proxy.CONVERSATION_SLOT, int)
        source = (ROOT / "src" / "grimoire" / "proxy" / "llama.py").read_text()
        self.assertNotIn("/slots/0", source, "slot number must not be hard-coded alongside the constant")

    def test_ordinary_requests_are_not_pinned(self):
        """Requests without a conversation id must stay spread across slots."""
        source = (ROOT / "src" / "grimoire" / "proxy" / "llama.py").read_text()
        guard = source.index("needs_slot_guard = validated_conversation_id is not None")
        pin = source.index('payload["id_slot"] = CONVERSATION_SLOT')
        between = source[guard:pin]
        self.assertIn("if needs_slot_guard:", between,
                      "pinning must sit inside the conversation-only branch")


class ResidentConversationMarkerTests(unittest.TestCase):
    def test_stop_clears_the_resident_conversation(self):
        """Stopping the backend discards its slots, so the marker must go too."""
        import grimoire.model_manager as mm

        active = mm.ActiveModel.__new__(mm.ActiveModel)
        active.process = None
        active._current_conv_id = "user\x00conv-42"
        active.stop()
        self.assertIsNone(active._current_conv_id)

    def test_marker_survives_normal_operation(self):
        """Only stop() clears it; a live model keeps its resident conversation."""
        source = (ROOT / "src" / "grimoire" / "model_manager.py").read_text()
        self.assertEqual(source.count("self._current_conv_id = None"), 1)


if __name__ == "__main__":
    unittest.main()
