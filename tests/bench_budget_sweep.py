#!/usr/bin/env python3
"""DDTree budget sweep benchmark.

Measures acceptance rate and tok/s across DDTree budgets to find the
sweet spot on the target hardware (RTX 3090, Qwen3.6-27B Q4_K_M).

Usage:
  1. Build the image: docker compose build
  2. Start the server with one config (edit etc/models.json, then restart)
  3. Run: python3 tests/bench_budget_sweep.py --port 8080

Alternatively, run each config manually with the curl commands below.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

BASE_URL = "http://127.0.0.1:{port}/v1/chat/completions"

SWEEP_CONFIGS = [
    # (label, n_max, branch_budget, total_tree, draft_topk, note)
    # Current production config (flat mode, under the 25-token cap)
    ("flat-baseline",   16,  0, 16,  1, "current flat production"),

    # Under the old 25-token cap — should hit GPU tape path
    ("tree-under-cap",  16,  6, 22,  4, "Luce sweet spot (budget=22)"),
    ("tree-under-cap2", 16,  8, 24,  4, "boundary test"),

    # Over the old 25-token cap — was hitting CPU fallback before fix
    ("tree-small",      24,  8, 32,  4, "small overcap"),

    # Your current config (was silently CPU-fallback due to cap)
    ("tree-current",    32, 16, 48,  4, "current models.json config"),

    # Aggressive budgets
    ("tree-aggro",      32, 24, 56,  4, "aggressive"),
    ("tree-max",        40, 30, 70,  4, "max reasonable"),
]

PROMPTS = [
    "Write a Python function to compute fibonacci numbers using dynamic programming.",
    "def quicksort(arr):",
    "Explain the difference between speculative decoding and speculative prefill.",
]


def call_server(port, prompt, max_tokens=256, config_override=None):
    headers = {"Content-Type": "application/json"}
    body = {
        "model": "dflash-canary-qwen3.6-27B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
    }
    if config_override:
        body["draft_n_max"] = config_override.get("n_max")
        body["branch_budget"] = config_override.get("branch_budget")
        body["draft_topk"] = config_override.get("draft_topk")

    url = BASE_URL.format(port=port)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0

    usage = result.get("usage", {})
    tokens_out = usage.get("completion_tokens", 0)
    tok_s = tokens_out / elapsed if elapsed > 0 else 0

    # Check server logs for acceptance stats (printed by Bee)
    return {
        "tokens": tokens_out,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(tok_s, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="DDTree budget sweep")
    parser.add_argument("--port", type=int, default=8082,
                        help="llama-server port")
    parser.add_argument("--warmup", type=int, default=2,
                        help="warmup prompts per config")
    parser.add_argument("--runs", type=int, default=3,
                        help="measured runs per config")
    args = parser.parse_args()

    results = []
    for label, n_max, branch_budget, total, topk, note in SWEEP_CONFIGS:
        override = {"n_max": n_max, "branch_budget": branch_budget,
                    "draft_topk": topk}
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"  n_max={n_max}, branch_budget={branch_budget}, "
              f"total_tree={total}, topk={topk}")
        print(f"  Note: {note}")

        # Warmup
        for _ in range(args.warmup):
            for p in PROMPTS:
                try:
                    call_server(args.port, p, max_tokens=64,
                                config_override=override)
                except Exception as e:
                    print(f"  Warmup error: {e}")

        # Benchmark
        stats = []
        for _ in range(args.runs):
            for p in PROMPTS:
                try:
                    r = call_server(args.port, p, max_tokens=256,
                                    config_override=override)
                    stats.append(r)
                    print(f"  Prompt: {r['tok_s']} tok/s, "
                          f"{r['tokens']} tokens in {r['elapsed_s']}s")
                except Exception as e:
                    print(f"  Error: {e}")

        if stats:
            avg = sum(s["tok_s"] for s in stats) / len(stats)
            print(f"  AVG: {round(avg, 2)} tok/s over {len(stats)} runs")
            results.append((label, n_max, branch_budget, total,
                          round(avg, 2), note))
        else:
            results.append((label, n_max, branch_budget, total, 0, note))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Label':<20} {'n_max':<6} {'budget':<8} {'total':<6} "
          f"{'tok/s':<8} {'Note'}")
    print("-" * 70)
    for label, n_max, bb, total, tok_s, note in results:
        marker = " <<< BEST" if any(
            r[-2] and tok_s >= r[-2] for r in results
        ) else ""
        print(f"{label:<20} {n_max:<6} {bb:<8} {total:<6} "
              f"{tok_s:<8} {note}{marker}")


if __name__ == "__main__":
    main()
