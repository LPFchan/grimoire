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
"""

import httpx
import json
import logging
import os
import subprocess
import sys
import time
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
FIXTURES = Path(os.environ.get("FIXTURES", "/home/yeowool/opencode_splits"))
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


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
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip() or 0)


def main():
    chars = TARGET_CHARS
    messages = build_multi_turn_prompt(chars, TURNS) if TURNS > 0 else build_single_prompt(chars)

    total_chars = sum(len(m.get("content", "")) for m in messages)
    log.info(f"Messages: {len(messages)}")
    log.info(f"Total chars: {total_chars:,}")

    vmb = vram_mb()
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

    # Check if PFlash fired by looking at gateway logs
    log.info("")
    log.info(f"If compressed vs raw differ significantly, PFlash fired.")
    log.info(f"At 5% keep ratio + 16K tail budget, {pt:,} tokens would")
    tail = min(16000, pt)
    middle = max(0, pt - tail)
    compressed = tail + int(middle * 0.05)
    log.info(f"  compress to ~{compressed:,} tokens if PFlash enabled.")


if __name__ == "__main__":
    main()
