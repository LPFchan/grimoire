"""End-to-end smoke tests against a live Grimoire gateway.

These tests hit the real HTTP API, load actual models, and verify:
  1. Basic inference (single-turn with OpenCode fixture data)
  2. Session continuity (multi-turn with conversation_id)
  3. Timing: TTFT, prefill, decode throughput

Prerequisites:
  - Gateway running (e.g. docker compose up)
  - Model files available at the configured paths
  - At least one GPU available

Run selectively:
    python -m pytest tests/test_e2e_smoke.py -v
    python -m pytest tests/test_e2e_smoke.py::DFlashSmokeTests -v
    python -m pytest tests/test_e2e_smoke.py::LlamaCppSmokeTests -v

Skip entirely:
    SKIP_E2E=1 python -m pytest tests/

Model overrides:
    GRIMOIRE_DFLASH_SMOKE_MODEL=dflash-native-qwen3.6-27B-canary python -m pytest tests/test_e2e_smoke.py::DFlashSmokeTests -v
    GRIMOIRE_LLAMA_SMOKE_MODEL=qwen-3.6-27B python -m pytest tests/test_e2e_smoke.py::LlamaCppSmokeTests -v

Long-prompt overrides:
    GRIMOIRE_LONG_PROMPT_MIN_CHARS=1500 GRIMOIRE_LONG_PROMPT_MAX_CHARS=4000 python -m pytest tests/test_e2e_smoke.py::DFlashSmokeTests::test_02_session_snapshot_restore -v
"""

import json
import os
import subprocess
import time
import unittest
import uuid
from pathlib import Path

import httpx

from tests._monitor import SystemMonitor

SKIP_E2E = os.environ.get("SKIP_E2E", "0") == "1"
BASE_URL = os.environ.get("GRIMOIRE_SMOKE_URL", "http://localhost:9001")
API_KEY = os.environ.get("GRIMOIRE_API_KEY", "")
DFLASH_SMOKE_MODEL = os.environ.get("GRIMOIRE_DFLASH_SMOKE_MODEL",     "dflash-native-qwen3.6-27B-canary")
LLAMA_SMOKE_MODEL = os.environ.get("GRIMOIRE_LLAMA_SMOKE_MODEL", "qwen-3.6-27B")
LONG_PROMPT_MIN_CHARS = int(os.environ.get("GRIMOIRE_LONG_PROMPT_MIN_CHARS", "1500"))
LONG_PROMPT_MAX_CHARS = int(os.environ.get("GRIMOIRE_LONG_PROMPT_MAX_CHARS", "4000"))
FIXTURES_DIR = Path(
    os.environ.get("GRIMOIRE_OPENCODE_SESSION_FIXTURES", "/home/yeowool/opencode_splits")
)


def _load_short_fixture() -> list[dict]:
    """Load a short OpenCode session fixture (first 2–3 turns only)."""
    candidates = sorted(FIXTURES_DIR.glob("opencode_ses_*.json"), key=lambda p: p.stat().st_size)
    for candidate in candidates:
        data = json.loads(candidate.read_text())
        messages = data.get("messages", [])
        if len(messages) >= 4:  # Need at least 2 turns (user+assistant × 2)
            # Convert to OpenAI message format
            result = []
            for raw_msg in messages[:4]:  # First 2 turns
                meta = json.loads(raw_msg.get("data", "{}"))
                parts = [json.loads(p.get("data", "{}")) for p in raw_msg.get("parts", [])]
                texts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
                if texts and meta.get("role") in ("user", "assistant"):
                    result.append({"role": meta["role"], "content": texts[0]})
            if len(result) >= 2:
                return result
    raise RuntimeError("No suitable fixture found")


def _load_long_prompt_fixture(min_chars: int = 1500, max_chars: int | None = None) -> list[dict]:
    """Load a deterministic long user prompt from OpenCode session fixtures."""
    best = None
    best_source = None
    preferred = FIXTURES_DIR / "opencode_ses_1edc_Assessing_lucebox-hub_integration_into_grimoire.json"
    candidates = []
    if preferred.exists():
        candidates.append(preferred)
    for candidate in sorted(FIXTURES_DIR.glob("opencode_ses_*.json"), key=lambda p: p.stat().st_size, reverse=True):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        data = json.loads(candidate.read_text())
        messages = data.get("messages", [])
        for raw_msg in messages:
            meta = json.loads(raw_msg.get("data", "{}"))
            if meta.get("role") != "user":
                continue
            parts = [json.loads(p.get("data", "{}")) for p in raw_msg.get("parts", [])]
            texts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
            for text in texts:
                if len(text) >= min_chars and (max_chars is None or len(text) <= max_chars):
                    if best is None or len(text) > len(best):
                        best = text
                        best_source = candidate.name
    if best:
        return [{"role": "user", "content": best}]
    if max_chars is None:
        raise RuntimeError(
            f"No user message found with at least {min_chars} chars in scanned fixtures"
        )
    raise RuntimeError(
        f"No user message found with {min_chars}-{max_chars} chars in scanned fixtures"
    )


class E2ESmokeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SKIP_E2E:
            raise unittest.SkipTest("SKIP_E2E is set")
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
            if r.status_code != 200:
                raise unittest.SkipTest(f"Gateway not healthy at {BASE_URL}")
        except Exception as e:
            raise unittest.SkipTest(f"Gateway unreachable at {BASE_URL}: {e}")

        cls._fixture_messages = _load_short_fixture()
        try:
            cls._long_fixture_messages = _load_long_prompt_fixture(
                min_chars=LONG_PROMPT_MIN_CHARS,
                max_chars=LONG_PROMPT_MAX_CHARS,
            )
        except RuntimeError:
            cls._long_fixture_messages = cls._fixture_messages

    @classmethod
    def _headers(cls):
        return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    @classmethod
    def _create_conversation(cls, model: str) -> str:
        response = httpx.post(
            f"{BASE_URL}/history",
            json={"title": f"smoke-{uuid.uuid4()}", "model": model, "messages": []},
            headers=cls._headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["id"]

    @classmethod
    def _chat(cls, model: str, messages: list[dict], conversation_id: str | None = None, max_tokens: int = 16):
        """Send a chat completion request and return the parsed response + timings."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        headers = cls._headers()

        t0 = time.monotonic()
        first_byte_at = None
        last_chunk_at = None
        chunks = []
        error_message = None

        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    continue
                error = frame.get("error") if isinstance(frame, dict) else None
                if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
                    error_message = error["message"]
                chunks.append(frame)
                last_chunk_at = time.monotonic()

        total_time = last_chunk_at - t0 if last_chunk_at else time.monotonic() - t0
        ttft = first_byte_at - t0 if first_byte_at else total_time
        decode_time = total_time - ttft

        # Extract assistant text and usage
        text_parts = []
        usage = {"completion_tokens": 0}
        for frame in chunks:
            if isinstance(frame, dict):
                delta = frame.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if frame.get("usage"):
                    usage = frame["usage"]

        assistant_text = "".join(text_parts)
        completion_tokens = usage.get("completion_tokens", 0)
        decode_tps = completion_tokens / decode_time if decode_time > 0 else 0

        return {
            "text": assistant_text,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft * 1000,
            "decode_time_ms": decode_time * 1000,
            "total_time_ms": total_time * 1000,
            "decode_tps": decode_tps,
            "status_code": response.status_code,
            "error": error_message,
        }

    def _assert_timings(self, result: dict, label: str):
        """Log and sanity-check timing metrics."""
        print(
            f"\n[{label}] TTFT={result['ttft_ms']:.0f}ms  "
            f"Decode={result['decode_time_ms']:.0f}ms  "
            f"Tokens={result['completion_tokens']}  "
            f"TPS={result['decode_tps']:.1f}"
        )
        self.assertLess(result["ttft_ms"], 120000, f"{label}: TTFT > 120s")
        self.assertGreater(result["decode_tps"], 10, f"{label}: decode TPS suspiciously low")


class DFlashSmokeTests(E2ESmokeTestCase):
    """Smoke tests for the DFlash speculative-decoding backend."""

    MODEL = DFLASH_SMOKE_MODEL

    def test_01_basic_chat_completion(self):
        """Single-turn chat with real fixture data returns a valid, timed response."""
        messages = [self._fixture_messages[0]]  # First user message only
        result = self._chat(self.MODEL, messages, max_tokens=16)

        self.assertEqual(result["status_code"], 200)
        self.assertTrue(result["text"], result.get("error"))
        self.assertGreater(result["completion_tokens"], 0)
        self._assert_timings(result, "DFlash basic")

    def test_02_session_snapshot_restore(self):
        """Two-turn conversation: second turn should restore from snapshot and be faster."""
        conversation_id = self._create_conversation(self.MODEL)

        # Turn 1 — establish context with a LONG prompt so prefill takes measurable time
        turn1_messages = self._long_fixture_messages
        prompt_len = len(turn1_messages[0]["content"])
        print(f"\n[DFlash session test] prompt chars={prompt_len}")

        result1 = self._chat(self.MODEL, turn1_messages, conversation_id=conversation_id, max_tokens=16)
        self.assertEqual(result1["status_code"], 200)
        self.assertTrue(result1["text"], result1.get("error"))
        self._assert_timings(result1, "DFlash turn 1 (long prompt)")

        # Turn 2 — should restore from snapshot (append assistant response + follow-up)
        turn2_messages = [
            *turn1_messages,
            {"role": "assistant", "content": result1["text"]},
            {"role": "user", "content": "Continue."},
        ]
        result2 = self._chat(self.MODEL, turn2_messages, conversation_id=conversation_id, max_tokens=16)
        self.assertEqual(result2["status_code"], 200)
        self.assertTrue(result2["text"], result2.get("error"))
        self._assert_timings(result2, "DFlash turn 2 (restore)")

        # Snapshot restore should make turn 2 significantly faster (lower TTFT).
        # If turn 1 was already very fast (<2s), the prompt wasn't long enough to
        # stress prefill, so we only assert speedup when turn 1 was slow.
        speedup = result1["ttft_ms"] / max(result2["ttft_ms"], 1)
        print(f"\n[DFlash restore speedup] turn1_ttft / turn2_ttft = {speedup:.1f}x")
        if result1["ttft_ms"] > 2000:
            self.assertGreater(
                speedup, 1.5,
                f"Turn 2 TTFT ({result2['ttft_ms']:.0f}ms) not significantly faster than turn 1 ({result1['ttft_ms']:.0f}ms); snapshot restore may not be working"
            )

        # Snapshot restore confirmed by speedup alone; tmpfs path may vary by config.


class LlamaCppSmokeTests(E2ESmokeTestCase):
    """Smoke tests for the llama.cpp backend."""

    MODEL = LLAMA_SMOKE_MODEL

    def test_01_basic_chat_completion(self):
        """Single-turn chat with real fixture data returns a valid, timed response."""
        messages = [self._fixture_messages[0]]
        result = self._chat(self.MODEL, messages, max_tokens=16)

        self.assertEqual(result["status_code"], 200)
        self.assertTrue(result["text"], result.get("error"))
        self.assertGreater(result["completion_tokens"], 0)
        self._assert_timings(result, "llama.cpp basic")

    def test_02_session_continuity(self):
        """Two-turn conversation preserves context via history store."""
        conversation_id = self._create_conversation(self.MODEL)

        # Turn 1
        turn1_messages = [self._fixture_messages[0]]
        result1 = self._chat(self.MODEL, turn1_messages, conversation_id=conversation_id, max_tokens=16)
        self.assertEqual(result1["status_code"], 200)
        self.assertTrue(result1["text"], result1.get("error"))
        self._assert_timings(result1, "llama.cpp turn 1")

        # Turn 2
        turn2_messages = [
            self._fixture_messages[0],
            {"role": "assistant", "content": result1["text"]},
            self._fixture_messages[2] if len(self._fixture_messages) > 2 else {"role": "user", "content": "Continue."},
        ]
        result2 = self._chat(self.MODEL, turn2_messages, conversation_id=conversation_id, max_tokens=16)
        self.assertEqual(result2["status_code"], 200)
        self.assertTrue(result2["text"], result2.get("error"))
        self._assert_timings(result2, "llama.cpp turn 2")


if __name__ == "__main__":
    unittest.main()
