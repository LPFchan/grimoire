"""DFlash stress test — replay an entire long OpenCode conversation end-to-end.

This test sends every turn of the largest available fixture to a live DFlash
model, exercising the snapshot store under sustained load. It is skipped by
default; run explicitly with:

    STRESS_TEST=1 python -m pytest tests/test_stress_dflash.py -v -s

Or standalone:

    python tests/test_stress_dflash.py

Metrics collected per turn:
  - TTFT (ms)
  - Decode time (ms)
  - Tokens generated
  - Decode throughput (tok/s)
  - Snapshot store RAM usage (MB)
  - Snapshot store disk usage (MB)
"""

import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path

import httpx

SKIP_STRESS = os.environ.get("STRESS_TEST", "0") != "1"
BASE_URL = os.environ.get("GRIMOIRE_SMOKE_URL", "http://localhost:9001")
API_KEY = os.environ.get("GRIMOIRE_API_KEY", "")
FIXTURES_DIR = Path(
    os.environ.get("GRIMOIRE_OPENCODE_SESSION_FIXTURES", "/home/yeowool/opencode_splits")
)
MODEL = os.environ.get("STRESS_MODEL", "dflash-pflash-qwen3.6-27B")
MAX_TOKENS = int(os.environ.get("STRESS_MAX_TOKENS", "64"))


def _find_largest_fixture() -> Path:
    files = list(FIXTURES_DIR.glob("opencode_ses_*.json"))
    if not files:
        raise RuntimeError(f"No fixtures in {FIXTURES_DIR}")
    return max(files, key=lambda p: p.stat().st_size)


def _normalize_fixture(path: Path) -> list[dict]:
    """Convert an OpenCode session fixture into OpenAI message format."""
    data = json.loads(path.read_text())
    messages = []
    for raw_msg in data.get("messages", []):
        meta = json.loads(raw_msg.get("data", "{}"))
        role = meta.get("role")
        if role not in ("user", "assistant"):
            continue
        parts = [json.loads(p.get("data", "{}")) for p in raw_msg.get("parts", [])]
        texts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
        if texts:
            messages.append({"role": role, "content": texts[0]})
    return messages


def _send_turn(model: str, messages: list[dict], conversation_id: str | None = None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": MAX_TOKENS,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id

    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    t0 = time.monotonic()
    first_byte_at = None
    last_chunk_at = None
    text_parts = []
    usage = {"completion_tokens": 0}

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=300.0,
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
            delta = frame.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                text_parts.append(delta["content"])
            if frame.get("usage"):
                usage = frame["usage"]
            last_chunk_at = time.monotonic()

    total_time = last_chunk_at - t0 if last_chunk_at else time.monotonic() - t0
    ttft = first_byte_at - t0 if first_byte_at else total_time
    decode_time = total_time - ttft
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "text": "".join(text_parts),
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "decode_time_ms": decode_time * 1000,
        "total_time_ms": total_time * 1000,
        "decode_tps": completion_tokens / decode_time if decode_time > 0 else 0,
    }


def _snapshot_store_usage():
    """Return (ram_mb, disk_mb) for the snapshot store inside the container."""
    try:
        ram = subprocess.run(
            ["docker", "exec", "grimoire", "du", "-sm", "/dev/shm/grimoire-snapshots/"],
            capture_output=True, text=True,
        )
        ram_mb = int(ram.stdout.split()[0]) if ram.returncode == 0 else 0
    except Exception:
        ram_mb = 0

    try:
        disk = subprocess.run(
            ["docker", "exec", "grimoire", "du", "-sm", "/var/lib/grimoire/snapshot_swap/"],
            capture_output=True, text=True,
        )
        disk_mb = int(disk.stdout.split()[0]) if disk.returncode == 0 else 0
    except Exception:
        disk_mb = 0

    return ram_mb, disk_mb


class DFlashStressTest(unittest.TestCase):
    """Stress-test DFlash snapshot system by replaying a full long conversation."""

    @classmethod
    def setUpClass(cls):
        if SKIP_STRESS:
            raise unittest.SkipTest("Set STRESS_TEST=1 to run stress tests")

        # Health check
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
            if r.status_code != 200:
                raise unittest.SkipTest(f"Gateway not healthy at {BASE_URL}")
        except Exception as e:
            raise unittest.SkipTest(f"Gateway unreachable at {BASE_URL}: {e}")

        cls.fixture_path = _find_largest_fixture()
        cls.fixture_messages = _normalize_fixture(cls.fixture_path)
        cls.conversation_id = str(uuid.uuid4())
        cls.results = []

        print(f"\n{'='*60}")
        print(f"DFlash Stress Test")
        print(f"Fixture: {cls.fixture_path.name}")
        print(f"Messages: {len(cls.fixture_messages)}")
        print(f"Model: {MODEL}")
        print(f"Max tokens/turn: {MAX_TOKENS}")
        print(f"{'='*60}\n")

    def test_replay_full_conversation(self):
        """Replay every assistant turn of the largest fixture."""
        messages = []
        turn_count = 0
        cumulative_tokens = 0

        for msg in self.fixture_messages:
            messages.append(msg)

            if msg["role"] != "assistant":
                continue

            turn_count += 1
            ram_before, disk_before = _snapshot_store_usage()
            t0 = time.monotonic()

            try:
                result = _send_turn(MODEL, messages, self.conversation_id)
            except Exception as e:
                self.fail(f"Turn {turn_count} failed after {len(messages)} messages: {e}")

            elapsed = time.monotonic() - t0
            ram_after, disk_after = _snapshot_store_usage()
            cumulative_tokens += result["completion_tokens"]

            self.results.append({
                "turn": turn_count,
                "prompt_messages": len(messages),
                **result,
                "ram_mb": ram_after,
                "disk_mb": disk_after,
                "wall_time_s": elapsed,
            })

            # Progress print every 10 turns
            if turn_count % 10 == 0:
                print(
                    f"  Turn {turn_count:3d} | TTFT {result['ttft_ms']:6.0f}ms | "
                    f"Decode {result['decode_tps']:5.1f} tok/s | "
                    f"RAM {ram_after:4d}MB | Disk {disk_after:4d}MB | "
                    f"Prompt {len(messages):3d} msgs"
                )

            # Sanity: snapshot store should not be empty after turn 1
            if turn_count == 1:
                self.assertGreater(ram_after + disk_after, 0, "No snapshot files created")

            # Sanity: response should not be empty
            self.assertTrue(result["text"], f"Turn {turn_count} produced empty response")

            # Append assistant response to message history for next turn
            messages.append({"role": "assistant", "content": result["text"]})

        # Final report
        self._print_report(turn_count, cumulative_tokens)

    def _print_report(self, turns: int, cumulative_tokens: int):
        if not self.results:
            return

        ttfts = [r["ttft_ms"] for r in self.results]
        decodes = [r["decode_tps"] for r in self.results]
        rams = [r["ram_mb"] for r in self.results]
        disks = [r["disk_mb"] for r in self.results]
        total_wall = sum(r["wall_time_s"] for r in self.results)

        print(f"\n{'='*60}")
        print(f"Stress Test Complete")
        print(f"{'='*60}")
        print(f"Turns replayed:      {turns}")
        print(f"Total wall time:     {total_wall/60:.1f} min")
        print(f"Total tokens out:    {cumulative_tokens}")
        print(f"")
        print(f"TTFT (ms):")
        print(f"  Min: {min(ttfts):.0f}  |  Max: {max(ttfts):.0f}  |  Avg: {sum(ttfts)/len(ttfts):.0f}")
        print(f"Decode throughput (tok/s):")
        print(f"  Min: {min(decodes):.1f}  |  Max: {max(decodes):.1f}  |  Avg: {sum(decodes)/len(decodes):.1f}")
        print(f"Snapshot store (MB):")
        print(f"  RAM:  min={min(rams)}  max={max(rams)}  final={rams[-1]}")
        print(f"  Disk: min={min(disks)}  max={max(disks)}  final={disks[-1]}")
        print(f"{'='*60}\n")

        # Write raw results to file for plotting
        report_path = Path("/tmp/dflash-stress-report.json")
        report_path.write_text(json.dumps({
            "model": MODEL,
            "fixture": self.fixture_path.name,
            "max_tokens": MAX_TOKENS,
            "turns": turns,
            "results": self.results,
        }, indent=2))
        print(f"Raw report written to: {report_path}")


if __name__ == "__main__":
    # Allow running standalone without pytest
    if SKIP_STRESS:
        print("Set STRESS_TEST=1 to run stress tests")
        sys.exit(0)
    unittest.main()
