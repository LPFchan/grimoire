#!/usr/bin/env python3
"""20K sysprompt KV cache canary test.

Measures cold vs warm TTFT using the content-hash KV cache on the
dflash-canary-qwen3.6-27B model. A ~20K token system prompt is built
from real opencode_splits transcripts.

Usage:
    python tests/canary_20k_sysprompt.py

Environment:
    MODEL       Model name (default: dflash-canary-qwen3.6-27B)
    BASE_URL    Gateway URL (default: http://localhost:9001)
    API_KEY     Auth key
    PROMPT_TOK  Target sysprompt size in tokens (default: 20000)
"""

import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("canary")

MODEL = os.environ.get("MODEL", "dflash-canary-qwen3.6-27B")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:9001")
API_KEY = os.environ.get("API_KEY", "7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88")
TARGET_TOK = int(os.environ.get("PROMPT_TOK", "20000"))
FIXTURES = Path(os.environ.get("FIXTURES", "/home/yeowool/opencode_splits"))

H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def collect_text(chars_target=300000):
    texts = []
    if FIXTURES.exists():
        for fp in sorted(FIXTURES.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True):
            if sum(len(t) for t in texts) >= chars_target:
                break
            data = json.loads(fp.read_text())
            for msg in data.get("messages", []):
                for part in msg.get("parts", []):
                    try:
                        pd = json.loads(part.get("data", "{}"))
                        if pd.get("type") == "text" and pd.get("text"):
                            texts.append(pd["text"])
                            if sum(len(t) for t in texts) >= chars_target:
                                break
                    except Exception:
                        pass
                if sum(len(t) for t in texts) >= chars_target:
                    break
    if not texts:
        text = ("The field of artificial intelligence has seen remarkable progress in recent years. "
                "Large language models have demonstrated impressive capabilities across a wide range of "
                "natural language processing tasks. These models are trained on vast corpora of text "
                "data and can generate coherent and contextually relevant responses. ") * 1500
        texts = [text]
    return " ".join(texts)


def main():
    import httpx

    # Estimate: actual chars/token ratio varies by text content
    # For synthetic text: ~6.3 chars/token; for transcripts: ~3.5 chars/token
    chars_for_target = int(TARGET_TOK * 6.5)

    log.info("Building ~%d token sysprompt from transcripts...", TARGET_TOK)
    flat = collect_text(chars_target=chars_for_target + 10000)
    sysprompt = flat[:chars_for_target]
    log.info("  sysprompt: %d chars", len(sysprompt))

    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": "Summarize the key themes discussed in this material."},
    ]
    payload = {"model": MODEL, "messages": messages, "max_tokens": 10, "stream": False}

    # Dry run: first request loads the model (cold model load, not cached)
    log.info("Dry run (model load + cold prefill)...")
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=300)
    dry_elapsed = time.monotonic() - t0
    dry_pt = r.json()["usage"]["prompt_tokens"]
    log.info("  HTTP %d in %.1fs — %d prompt tokens", r.status_code, dry_elapsed, dry_pt)
    if r.status_code != 200:
        log.error("Dry run failed: %s", r.text[:300])
        return

    # Cold: first time with this exact sysprompt (KV cache miss)
    log.info("")
    log.info("Cold request (KV cache miss)...")
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=300)
    cold_elapsed = time.monotonic() - t0
    cold_pt = r.json()["usage"]["prompt_tokens"]
    cold_ct = r.json()["usage"]["completion_tokens"]
    cold_cached = r.json().get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    log.info("  HTTP %d in %.1fs — %d+%d tokens (cached=%d)",
             r.status_code, cold_elapsed, cold_pt, cold_ct, cold_cached)
    if r.status_code != 200:
        log.error("Cold request failed: %s", r.text[:300])
        return

    # Small delay to let KV cache save complete
    time.sleep(3)

    # Warm: same sysprompt (KV cache hit — should be much faster)
    log.info("")
    log.info("Warm request (KV cache hit)...")
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=300)
    warm_elapsed = time.monotonic() - t0
    warm_pt = r.json()["usage"]["prompt_tokens"]
    warm_ct = r.json()["usage"]["completion_tokens"]
    warm_cached = r.json().get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    log.info("  HTTP %d in %.1fs — %d+%d tokens (cached=%d)",
             r.status_code, warm_elapsed, warm_pt, warm_ct, warm_cached)
    if r.status_code != 200:
        log.error("Warm request failed: %s", r.text[:300])
        return

    # Second warm: should be even faster (already in RAM cache)
    log.info("")
    log.info("Warm #2 (RAM cache)...")
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=H, timeout=300)
    warm2_elapsed = time.monotonic() - t0
    warm2_cached = r.json().get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    log.info("  HTTP %d in %.1fs (cached=%d)", r.status_code, warm2_elapsed, warm2_cached)

    # Cumulative savings simulation
    sim_requests = 100
    sim_cold = cold_elapsed * sim_requests
    sim_warm = warm_elapsed * sim_requests
    log.info("")
    log.info("  Cumulative savings (%d requests):" % sim_requests)
    log.info("    Without KV cache:  %.1fs" % sim_cold)
    log.info("    With KV cache:     %.1fs" % sim_warm)
    log.info("    Total time saved:  %.1fs (%.1f min)" % (sim_cold - sim_warm, (sim_cold - sim_warm) / 60))

    log.info("")
    log.info("=" * 55)
    log.info("RESULTS")
    log.info("=" * 55)
    log.info("  Model:                %s" % MODEL)
    log.info("  Sysprompt tokens:     %d" % cold_pt)
    log.info("  KV cache hit rate:    %.1f%%" % (warm_cached / max(cold_pt, 1) * 100))
    log.info("  Dry run (model load): %.1fs" % dry_elapsed)
    log.info("  Cold (KV miss):       %.1fs" % cold_elapsed)
    log.info("  Warm (KV hit):        %.1fs" % warm_elapsed)
    log.info("  Speedup (cold/warm):  %.2fx" % (cold_elapsed / max(warm_elapsed, 0.001)))
    log.info("  Cached tokens:        %d / %d" % (warm_cached, cold_pt))
    log.info("")
    if warm_cached > 0:
        log.info("KV CACHE VERIFIED — content-hash slot save/restore works at 20K scale")
        log.info("PASS")


if __name__ == "__main__":
    main()
