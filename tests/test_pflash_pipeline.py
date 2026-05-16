#!/usr/bin/env python3
"""Test the full PFlash compression pipeline end-to-end.

Usage:
    MODEL=pflash-qwen3.6-27B python tests/test_pflash_pipeline.py

Environment:
    MODEL       Model name (default: pflash-qwen3.6-27B)
    BASE_URL    Gateway URL (default: http://localhost:9001)
    API_KEY     Auth key
    MAX_TOKENS  Output tokens (default: 30)
    TARGET_CHARS Approx prompt size in chars (default: 120000)
    THRESHOLD   Min tokens to trigger compression (default: auto from cfg)
    TURNS       Number of conversation turns (default: 0 = single)
    TIMEOUT     Request timeout seconds (default: 600)
    GPU_INDEX   Host GPU index for VRAM sampling (default: 0)
"""

import httpx
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pflash_test")

MODEL = os.environ.get("MODEL", "pflash-qwen3.6-27B")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:9001")
API_KEY = os.environ.get("API_KEY", "7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "30"))
TARGET_CHARS = int(os.environ.get("TARGET_CHARS", "200000"))
TURNS = int(os.environ.get("TURNS", "0"))
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))
GPU_INDEX = os.environ.get("GPU_INDEX", "0")
SMOKE_CONTAINER = os.environ.get("GRIMOIRE_SMOKE_CONTAINER", "")
FIXTURES = Path(os.environ.get("FIXTURES", "/home/yeowool/opencode_splits"))
MODELS_JSON = Path(
    os.environ.get(
        "MODELS_JSON",
        str(Path(__file__).resolve().parents[1] / "etc" / "models.json"),
    )
)
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

PFLASH_DEBUG_RE = re.compile(r"pflash debug: fired=(True|False) orig=(\d+) compressed=(\d+)")


def build_single_prompt(chars):
    """Build a single long user message."""
    sessions = sorted(FIXTURES.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
    texts = []
    for fp in sessions:
        if sum(len(t) for t in texts) >= chars:
            break
        data = json.loads(fp.read_text())
        for msg in data.get("messages", []):
            for part in msg.get("parts", []):
                try:
                    pd = json.loads(part.get("data", "{}"))
                    if pd.get("type") == "text" and pd.get("text"):
                        texts.append(pd["text"])
                        if sum(len(t) for t in texts) >= chars:
                            break
                except Exception:
                    pass
            if sum(len(t) for t in texts) >= chars:
                break
    return [{"role": "user", "content": "\n\n".join(texts)[:chars]}]


def build_multi_turn_prompt(chars, turns):
    """Build a multi-turn conversation with `turns` user/assistant pairs.
    Each turn is ~chars/turns chars split across the two messages."""
    if turns <= 0:
        return build_single_prompt(chars)

    sessions = sorted(FIXTURES.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
    all_texts = []
    target_chars_total = chars * 3  # collect 3x for headroom
    for fp in sessions:
        data = json.loads(fp.read_text())
        for msg in data.get("messages", []):
            for part in msg.get("parts", []):
                try:
                    pd = json.loads(part.get("data", "{}"))
                    if pd.get("type") == "text" and pd.get("text"):
                        all_texts.append(pd["text"])
                        if sum(len(t) for t in all_texts) >= target_chars_total:
                            break
                except Exception:
                    pass
            if sum(len(t) for t in all_texts) >= target_chars_total:
                break
        if sum(len(t) for t in all_texts) >= target_chars_total:
            break
        data = json.loads(fp.read_text())
        for msg in data.get("messages", []):
            for part in msg.get("parts", []):
                try:
                    pd = json.loads(part.get("data", "{}"))
                    if pd.get("type") == "text" and pd.get("text"):
                        all_texts.append(pd["text"])
                        if sum(len(t) for t in all_texts) >= chars * 2:
                            break
                except Exception:
                    pass
            if sum(len(t) for t in all_texts) >= chars * 2:
                break

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    all_chars = " ".join(all_texts)
    chunk = max(2000, chars // max(turns, 1))
    pos = 0
    for t in range(turns):
        if pos >= len(all_chars):
            break
        user_text = all_chars[pos:pos + chunk]
        pos += chunk
        messages.append({"role": "user", "content": user_text})
        if pos >= len(all_chars):
            break
        asst_text = all_chars[pos:pos + chunk // 2]
        pos += chunk // 2
        messages.append({"role": "assistant",
                         "content": f"<think>ok</think>\n\n{asst_text or 'I understand.'}"})
    return messages


def vram_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU_INDEX],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip() or 0)


def recent_pflash_debug(since: str):
    if not SMOKE_CONTAINER:
        return None
    out = subprocess.run(
        ["/usr/bin/docker", "logs", "--since", since, SMOKE_CONTAINER],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    latest = None
    for line in out.stdout.splitlines() + out.stderr.splitlines():
        match = PFLASH_DEBUG_RE.search(line)
        if not match:
            continue
        latest = {
            "fired": match.group(1) == "True",
            "orig_tokens": int(match.group(2)),
            "compressed_tokens": int(match.group(3)),
        }
    return latest


def model_prefill_settings():
    try:
        data = json.loads(MODELS_JSON.read_text())
    except Exception:
        return None

    models = data.get("models") if isinstance(data, dict) else None
    cfg = models.get(MODEL) if isinstance(models, dict) else None
    if not isinstance(cfg, dict):
        return None

    threshold = cfg.get("prefill-threshold")
    keep_ratio = cfg.get("prefill-keep-ratio")
    tail_budget = cfg.get("prefill-tail-budget")
    try:
        return {
            "threshold": int(threshold) if threshold is not None else None,
            "keep_ratio": float(keep_ratio) if keep_ratio is not None else 0.05,
            "tail_budget": int(tail_budget) if tail_budget is not None else 16000,
        }
    except (TypeError, ValueError):
        return None


def main():
    chars = TARGET_CHARS
    messages = build_multi_turn_prompt(chars, TURNS) if TURNS > 0 else build_single_prompt(chars)
    prefill_settings = model_prefill_settings()

    total_chars = sum(len(m.get("content", "")) for m in messages)
    log.info(f"Messages: {len(messages)}")
    log.info(f"Total chars: {total_chars:,}")

    vmb = vram_mb()
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    t0 = time.monotonic()
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "stream": False},
        headers=H, timeout=TIMEOUT,
    )
    elapsed = time.monotonic() - t0
    vma = vram_mb()
    d = r.json() if r.text else {}
    u = d.get("usage", {})

    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    reason = d.get("choices", [{}])[0].get("finish_reason", "?")

    log.info(f"HTTP {r.status_code} in {elapsed:.1f}s")
    log.info(f"Prompt tokens: {pt:,}  Completion tokens: {ct:,}  Finish: {reason}")
    log.info(f"VRAM: {vmb:,} -> {vma:,} MB (delta {vma - vmb:+,})")

    if r.status_code != 200:
        try:
            log.error(f"Error: {d.get('detail', '')[:400]}")
        except Exception:
            log.error(f"Error: {r.text[:400] if r.text else 'empty'}")
        return

    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    rc = d.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
    if content:
        log.info(f"Content ({len(content)}): {content[:200]}")
    if rc:
        log.info(f"Reasoning ({len(rc)}): {rc[:200]}")

    debug = recent_pflash_debug(started_at)
    if debug is not None:
        fired = debug["fired"]
        orig = debug["orig_tokens"]
        compressed_tokens = debug["compressed_tokens"]
        ratio = (orig / max(compressed_tokens, 1)) if compressed_tokens > 0 else 0.0
        log.info(
            "PFlash debug: fired=%s raw=%s compressed=%s ratio=%.2fx",
            fired,
            f"{orig:,}",
            f"{compressed_tokens:,}",
            ratio,
        )
        if (
            not fired
            and prefill_settings is not None
            and prefill_settings["threshold"] is not None
            and orig >= prefill_settings["threshold"]
        ):
            log.info(
                "Threshold was crossed without reduction; this usually means head/tail protection covered all prompt blocks."
            )
        return

    log.info("")
    log.info("No parseable PFlash debug line found in recent container logs.")
    if prefill_settings is None:
        log.info("Heuristic estimate unavailable because model prefill settings could not be loaded.")
        return

    keep_ratio = prefill_settings["keep_ratio"]
    tail_budget = prefill_settings["tail_budget"]
    tail = min(tail_budget, pt)
    middle = max(0, pt - tail)
    compressed = tail + int(middle * keep_ratio)
    log.info(
        "Heuristic only: using API prompt_tokens and model config because raw/compressed counts were unavailable."
    )
    log.info(
        "At %.0f%% keep ratio + %s tail budget, %s tokens would",
        keep_ratio * 100,
        f"{tail_budget:,}",
        f"{pt:,}",
    )
    log.info(f"  compress to ~{compressed:,} tokens if eligible for PFlash.")


if __name__ == "__main__":
    main()
