#!/usr/bin/env python3
"""Measure VMM park/unpark overhead vs normal PFlash coexistence.

Compares:
  - pflash-park-qwen3.6-27B (park/unpark enabled — VMM for llama-server)
  - pflash-qwen3.6-27B    (no park/unpark — both processes coexist)

Reports TTFT, VRAM delta, and park/unpark timing for each.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("park_measure")

BASE_URL = os.environ.get("BASE_URL", "http://localhost:9001")
API_KEY = os.environ.get("API_KEY", "7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88")
FIXTURES = Path(os.environ.get("FIXTURES", "/home/yeowool/opencode_splits"))
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def vram_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=15,
    )
    return int(r.stdout.strip().split("\n")[0])


def build_big_prompt():
    text = ("The field of artificial intelligence has seen remarkable progress in recent years. "
            "Large language models have demonstrated impressive capabilities. ") * 1500
    return [
        {"role": "system", "content": text[:130000]},
        {"role": "user", "content": "Hello, can you help me understand AI?"},
        {"role": "assistant", "content": "Yes, I'd be happy to help explain artificial intelligence concepts."},
        {"role": "user", "content": "What is deep learning and how does it relate to neural networks?"},
    ]


def stop_model(name):
    import httpx
    try:
        r = httpx.post(f"{BASE_URL}/stop/{name}", headers=H, timeout=30)
        log.info("  Stop %s: %s", name, r.json().get("status", r.text[:50]))
    except Exception as e:
        log.warning("  Stop %s failed: %s", name, e)


def measure_model(model_name, label):
    import httpx
    log.info("")
    log.info("=" * 60)
    log.info("  %s", label)
    log.info("=" * 60)

    messages = build_big_prompt()
    payload = {"model": model_name, "messages": messages, "max_tokens": 10, "stream": False}

    # Stop any running model first
    stop_model("pflash-qwen3.6-27B")
    stop_model("pflash-park-qwen3.6-27B")
    stop_model("dflash-qwen3.6-27B")
    time.sleep(3)

    vr_before = vram_mb()
    log.info("  VRAM before load: %s MB", f"{vr_before:,}")

    # Model load + dry run (triggers model loading)
    log.info("  Loading model...")
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=600)
    load_s = time.monotonic() - t0
    vr_after = vram_mb()
    pt = r.json()["usage"]["prompt_tokens"]
    log.info("  Dry run: %s MB -> %s MB  (%.1fs, %d tokens, HTTP %d)",
             f"{vr_before:,}", f"{vr_after:,}", load_s, pt, r.status_code)

    if r.status_code != 200:
        log.error("  FAILED: %s", r.text[:200])
        return None

    # Cold PFlash request
    log.info("  Cold PFlash request...")
    vr_cold_before = vram_mb()
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=600)
    cold_s = time.monotonic() - t0
    vr_cold_after = vram_mb()
    pt = r.json()["usage"]["prompt_tokens"]
    ct = r.json()["usage"]["completion_tokens"]
    log.info("  Cold: %.3fs  (%d+%d tokens, VRAM %s MB, HTTP %d)",
             cold_s, pt, ct, f"{vr_cold_after:,}", r.status_code)

    if r.status_code != 200:
        log.error("  FAILED: %s", r.text[:200])
        return None

    time.sleep(3)

    # Warm PFlash request (KV cache hit)
    log.info("  Warm PFlash request...")
    vr_warm_before = vram_mb()
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=600)
    warm_s = time.monotonic() - t0
    vr_warm_after = vram_mb()
    cached = r.json().get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    log.info("  Warm: %.3fs  (cached=%d, VRAM %s MB, HTTP %d)",
             warm_s, cached, f"{vr_warm_after:,}", r.status_code)

    if r.status_code != 200:
        log.error("  FAILED: %s", r.text[:200])
        return None

    return {
        "model": model_name,
        "pt": pt,
        "load_s": load_s,
        "cold_s": cold_s,
        "warm_s": warm_s,
        "cached": cached,
        "vr_before": vr_before,
        "vr_after_load": vr_after,
        "vr_cold": vr_cold_after,
        "vr_warm": vr_warm_after,
        "speedup": cold_s / max(warm_s, 0.001),
    }


def main():
    results = []

    r1 = measure_model("pflash-park-qwen3.6-27B", "WITH park/unpark (VMM)")
    if r1:
        results.append(r1)

    r2 = measure_model("pflash-qwen3.6-27B", "WITHOUT park/unpark (coexist)")
    if r2:
        results.append(r2)

    if len(results) < 2:
        log.error("Need both results to compare")
        return

    log.info("")
    log.info("=" * 60)
    log.info("  COMPARISON")
    log.info("=" * 60)
    log.info("  %-30s %15s %15s", "", "With VMM", "Without VMM")
    log.info("  %-30s %15s %15s", "-" * 30, "-" * 15, "-" * 15)
    for key, fmt in [("Prompt tokens", "pt", "d"),
                     ("Load time", "load_s", ".1f"),
                     ("Cold TTFT", "cold_s", ".3f"),
                     ("Warm TTFT", "warm_s", ".3f"),
                     ("Speedup", "speedup", ".2f"),
                     ("Cached tokens", "cached", "d"),
                     ("VRAM after load", "vr_after_load", ",d"),
                     ("VRAM cold", "vr_cold", ",d"),
                     ("VRAM warm", "vr_warm", ",d")]:
        v1 = r1.get(key[1], 0)
        v2 = r2.get(key[1], 0)
        f = key[2]
        if f == "d":
            log.info("  %-30s %15s %15s", key[0], f"{v1:,}", f"{v2:,}")
        elif f == ".3f":
            log.info("  %-30s %15.3f %15.3f", key[0], v1, v2)
        elif f == ".1f":
            log.info("  %-30s %15.1f %15.1f", key[0], v1, v2)
        elif f == ".2f":
            log.info("  %-30s %15.2f %15.2f", key[0], v1, v2)
        elif f == ",d":
            log.info("  %-30s %15s MB %15s MB", key[0], f"{v1:,}", f"{v2:,}")

    log.info("")
    cold_diff = r1["cold_s"] - r2["cold_s"]
    if cold_diff > 0.2:
        log.info("  VMM park/unpark adds %.0fms overhead (%.3fs)", cold_diff * 1000, cold_diff)
    elif cold_diff < -0.2:
        log.info("  VMM park/unpark saves %.0fms (%.3fs)!", abs(cold_diff) * 1000, abs(cold_diff))
    else:
        log.info("  No significant TTFT difference (%.0fms)", cold_diff * 1000)


if __name__ == "__main__":
    main()
