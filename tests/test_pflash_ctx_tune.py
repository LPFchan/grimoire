#!/usr/bin/env python3
"""
PID-like context-length tuning for pflash-qwen3.6-27B.

Binary searches for the maximum stable ctx-size on RTX 3090 24GB
by sending long prompts from OpenCode session fixtures and checking
for OOM failures.

Usage:
    python tests/test_pflash_ctx_tune.py           # full binary search
    python tests/test_pflash_ctx_tune.py 80000     # single-shot test at given ctx
    python tests/test_pflash_ctx_tune.py 60000 200000  # custom range

Environment:
    TUNE_MAX_TOKENS=50   output tokens per test (default: 50)
    TUNE_QUICK=1         skip fine-tuning phase (default: full)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:9001"
API_KEY = "7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88"
MODEL = "pflash-qwen3.6-27B"
FIXTURES_DIR = Path("/home/yeowool/opencode_splits")
MODELS_JSON = Path("/home/yeowool/grimoire/etc/models.json")
GRIMOIRE_DIR = Path("/home/yeowool/grimoire")
MAX_TOKENS_OUT = int(os.environ.get("TUNE_MAX_TOKENS", "50"))
QUICK = os.environ.get("TUNE_QUICK", "") == "1"

# q8_0 KV cache: 34816 bytes/token
KV_BYTES_PER_TOKEN = 34816
# Fixed VRAM cost (no draft, no DDTree rollback):
#   target weights (Q4_K_M): ~14.0 GB
#   SSM/conv/feat states:    ~0.35 GB
#   CUDA workspace/overhead: ~0.50 GB
FIXED_VRAM_GB = 14.85
GPU_TOTAL_GB = 23.5


def api(path, method="GET", body=None, timeout=300):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}{path}"
    if method == "POST":
        r = httpx.post(url, json=body, headers=headers, timeout=timeout)
    else:
        r = httpx.get(url, headers=headers, timeout=timeout)
    return r.json() if r.text else {}


def get_health():
    return api("/health")


def gpu_vram_used(gpu_id=0):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True, timeout=5,
    )
    try:
        return int(out.stdout.strip())  # MB
    except Exception:
        return 0


def build_prompt(target_tokens):
    """Build a prompt of approximately target_tokens tokens from session fixtures.
    Uses 2 chars/token heuristic for English text (conservative for Qwen tokenizer)."""
    target_chars = target_tokens * 2
    sessions = sorted(
        FIXTURES_DIR.glob("opencode_ses_*.json"),
        key=lambda f: f.stat().st_size, reverse=True,
    )
    all_texts = []
    for fp in sessions:
        if sum(len(t) for t in all_texts) >= target_chars:
            break
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        for msg in data.get("messages", []):
            for part in msg.get("parts", []):
                try:
                    pd = json.loads(part.get("data", "{}"))
                    if pd.get("type") == "text" and pd.get("text"):
                        all_texts.append(pd["text"])
                        if sum(len(t) for t in all_texts) >= target_chars:
                            break
                except Exception:
                    pass
            if sum(len(t) for t in all_texts) >= target_chars:
                break
    prompt = "\n\n".join(all_texts)
    return prompt[:target_chars] if len(prompt) > target_chars else prompt


def send_chat(messages, max_tokens=MAX_TOKENS_OUT, stream=False, timeout=300):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    t0 = time.monotonic()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}/v1/chat/completions"
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=timeout)
        elapsed = time.monotonic() - t0
        status = r.status_code

        if status != 200:
            return False, f"HTTP {status}: {r.text[:200]}", {}, elapsed
        if not r.text:
            return False, "Empty response body", {}, elapsed
        try:
            data = r.json()
        except Exception as e:
            return False, f"JSON parse error: {e}", {}, elapsed

        if "error" in data:
            return False, data.get("error", str(data)), {}, elapsed
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        if not choices:
            return False, f"No choices in response: {json.dumps(data)[:200]}", usage, elapsed
        content = choices[0].get("message", {}).get("content", "")
        finish = choices[0].get("finish_reason", "?")
        ctx = data.get("context_window", "?")
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        if not content.strip():
            return False, f"Empty content: finish={finish}", usage, elapsed
        if prompt_tokens <= 0:
            return False, f"prompt_tokens=0 (broken response?)", usage, elapsed

        msg = (f"prompt={prompt_tokens} completion={completion_tokens} "
               f"finish={finish} ctx={ctx} {elapsed:.1f}s")
        return True, msg, usage, elapsed
    except httpx.ReadTimeout:
        return False, "HTTP timeout", {}, time.monotonic() - t0
    except httpx.ConnectError:
        return False, "Connection failed (gateway crashed?)", {}, time.monotonic() - t0
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", {}, time.monotonic() - t0


def set_ctx_size_and_restart(ctx_size):
    """Update models.json ctx-size, restart container, reload model. Returns bool."""
    print(f"  [config] Setting ctx-size={ctx_size}...")
    with open(MODELS_JSON) as f:
        data = json.load(f)
    cfg = data["models"]["pflash-qwen3.6-27B"]
    old = cfg.get("ctx-size", 0)
    cfg["ctx-size"] = ctx_size
    cfg["max-effective-context"] = ctx_size
    with open(MODELS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [config] Updated: {old} → {ctx_size}")

    print(f"  [restart] docker compose up -d...")
    subprocess.run(
        ["/usr/bin/docker", "compose", "-f", "docker-compose.yml", "up", "-d"],
        cwd=str(GRIMOIRE_DIR), capture_output=True, timeout=60,
    )
    # Wait for healthy
    for _ in range(60):
        time.sleep(2)
        try:
            h = get_health()
            if h.get("status") == "healthy":
                print(f"  [restart] Gateway healthy")
                break
        except Exception:
            pass
    else:
        print(f"  [restart] FAILED: gateway not healthy")
        return False

    # Switch model
    print(f"  [switch] Starting {MODEL}...")
    api(f"/switch/{MODEL}", method="POST")
    for i in range(120):
        time.sleep(2)
        h = get_health()
        if MODEL in h.get("active_models", []):
            print(f"  [switch] Model active")
            time.sleep(3)  # let prefill settle
            return True
    print(f"  [switch] FAILED: model not active after 240s")
    return False


def stop_model():
    api(f"/stop/{MODEL}", method="POST")
    time.sleep(3)


def smoke_test_ctx(target_ctx, target_tokens=None):
    """Full smoke test: configure ctx, restart, load model, send prompt, check result."""
    if target_tokens is None:
        target_tokens = target_ctx  # send ~full ctx worth of tokens

    print(f"\n{'─'*55}")
    print(f"Testing ctx-size={target_ctx:,} with ~{target_tokens:,} token prompt")

    if not set_ctx_size_and_restart(target_ctx):
        return False, "infra_setup_failed", 0, 0

    # Build and send prompt
    prompt = build_prompt(target_tokens)
    chars = len(prompt)
    print(f"  [prompt] {chars:,} chars (est ~{target_tokens:,} tokens)")

    vram_before = gpu_vram_used()
    success, msg, usage, elapsed = send_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS_OUT,
    )
    vram_after = gpu_vram_used()

    actual_pt = usage.get("prompt_tokens", 0)
    print(f"  {'[PASS]' if success else '[FAIL]'} {msg}")
    print(f"  [VRAM] before={vram_before}MB after={vram_after}MB delta={vram_after-vram_before:+d}MB")
    print(f"  [KV]   {actual_pt:,} tokens × {KV_BYTES_PER_TOKEN} = {actual_pt*KV_BYTES_PER_TOKEN/1024**3:.2f} GB KV")

    # Stop model to free GPU
    stop_model()
    return success, msg, actual_pt, elapsed


def main():
    if len(sys.argv) == 2:
        # Single-shot mode
        test_ctx = int(sys.argv[1])
        success, msg, tokens, elapsed = smoke_test_ctx(test_ctx)
        print(f"\nResult: {'PASS' if success else 'FAIL'} at ctx={test_ctx}")
        sys.exit(0 if success else 1)

    if len(sys.argv) >= 3:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
    else:
        lo, hi = 60000, 240000

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  pflash-qwen3.6-27B Context Length Tuning                   ║
║  Model: {MODEL}                                           ║
║  Backend: dflash (pflash compression only, no DDTree)       ║
║  KV cache: q8_0 ({KV_BYTES_PER_TOKEN:,} bytes/token)                ║
║  Fixed VRAM: {FIXED_VRAM_GB:.1f} GB (weights + SSM states + overhead)         ║
║  GPU VRAM:   {GPU_TOTAL_GB:.1f} GB                                                ║
║  Available:  ~{GPU_TOTAL_GB-FIXED_VRAM_GB:.1f} GB → ~{int((GPU_TOTAL_GB-FIXED_VRAM_GB)*1024**3/KV_BYTES_PER_TOKEN):,} tokens theoretical max        ║
║  Search range: {lo:,} – {hi:,}                                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Phase 1: Verify baseline
    print("─── Phase 1: Baseline (lower bound) ───")
    success, _, _, _ = smoke_test_ctx(lo)
    if not success:
        print(f"ERROR: Baseline at ctx={lo} failed. Infrastructure issue?")
        sys.exit(1)
    max_working = lo

    # Phase 2: Test upper bound
    print(f"\n─── Phase 2: Upper bound test ───")
    success, _, _, _ = smoke_test_ctx(hi)
    if success:
        print(f"\n*** ctx-size ≥ {hi} — upper bound works! No limit found in range. ***")
        print(f"    May need to test beyond {hi}")
        return

    # Phase 3: Binary search
    print(f"\n─── Phase 3: Binary search {lo:,}–{hi:,} ───")
    iteration = 0
    while lo < hi - 2000:
        iteration += 1
        mid = ((lo + hi) // 2000) * 1000  # round to nearest 1K
        if mid <= lo:
            mid = lo + 1000
        print(f"\n[Iter {iteration}] lo={lo:,} hi={hi:,} mid={mid:,}")

        try:
            success, msg, actual_tokens, elapsed = smoke_test_ctx(mid)
        except Exception as e:
            print(f"  [CRASH] Exception: {e}")
            success = False

        if success:
            lo = mid
            max_working = mid
            print(f"  → PASS — range now [{lo:,}, {hi:,}]")
        else:
            hi = mid
            print(f"  → FAIL — range now [{lo:,}, {hi:,}]")

    # Phase 4: Fine-tune (optional)
    if not QUICK:
        print(f"\n─── Phase 4: Fine-tuning near {max_working:,} ───")
        for delta in [5000, 3000, 1000]:
            test_val = max_working + delta
            if test_val >= hi:
                continue
            try:
                success, msg, _, _ = smoke_test_ctx(test_val)
            except Exception as e:
                success = False
            if success:
                max_working = test_val
                print(f"  +{delta:,} → PASS, max={max_working:,}")
            else:
                print(f"  +{delta:,} → FAIL, max={max_working:,}")

    # Final report
    kv_gb = max_working * KV_BYTES_PER_TOKEN / 1024**3
    total_gb = FIXED_VRAM_GB + kv_gb
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FINAL RESULT                                               ║
║  Max stable ctx-size: {max_working:,} tokens                       ║
║  KV cache VRAM:       {kv_gb:.1f} GB (@ q8_0)                         ║
║  Total VRAM used:     {total_gb:.1f} GB / {GPU_TOTAL_GB:.1f} GB                         ║
║  Headroom:            {GPU_TOTAL_GB - total_gb:.1f} GB                             ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
