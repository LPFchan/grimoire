#!/opt/grimoire-venv/bin/python
"""VRAM drift soak for PFlash compression pipeline.

Runs N iterations against the live server with a multi-turn prompt
that triggers PFlash compression (threshold >36K, multiple blocks
with a compressible middle after head+tail protection).
"""

import json, logging, os, subprocess, sys, time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("soak")

FIXTURES = Path("/home/yeowool/opencode_splits")
ITERS = 10
COOLDOWN = 5
THRESHOLD_MB = 512
LARGE_CHUNK = 150000


def vram_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=15,
    )
    return int(r.stdout.strip().split("\n")[0])


def collect(chars_limit=300000):
    texts = []
    for fp in sorted(FIXTURES.glob("*.json"), key=lambda f: f.stat().st_size, reverse=True):
        if sum(len(t) for t in texts) >= chars_limit:
            break
        try:
            data = json.loads(fp.read_text())
            for msg in data.get("messages", []):
                for part in msg.get("parts", []):
                    try:
                        pd = json.loads(part.get("data", "{}"))
                        if pd.get("type") == "text" and pd.get("text"):
                            texts.append(pd["text"])
                            if sum(len(t) for t in texts) >= chars_limit:
                                break
                    except Exception:
                        pass
                if sum(len(t) for t in texts) >= chars_limit:
                    break
        except Exception:
            pass
    return texts


def build_conversation(all_texts):
    flat = " ".join(all_texts)
    return [
        {"role": "system", "content": "You are a helpful AI coding assistant."},
        {"role": "user", "content": flat[:500]},
        {"role": "assistant", "content": "I understand the context and will help you."},
        {"role": "user", "content": flat[500:500 + LARGE_CHUNK]},
        {"role": "assistant", "content": flat[500 + LARGE_CHUNK:500 + 2 * LARGE_CHUNK]},
    ]


def main():
    import httpx
    H = {
        "Authorization": f"Bearer {os.environ.get('GRIMOIRE_API_KEY', '7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88')}",
        "Content-Type": "application/json",
    }

    texts = collect()
    log.info("Collected %s chars", f"{sum(len(t) for t in texts):,}")
    messages = build_conversation(texts)
    log.info("Built %d messages", len(messages))

    log.info("Dry run — checking PFlash fires...")
    r = httpx.post(
        "http://localhost:9001/v1/chat/completions",
        json={"model": "pflash-qwen3.6-27B", "messages": messages, "max_tokens": 1, "stream": False},
        headers=H, timeout=180,
    )
    if r.status_code == 413:
        log.error("PFlash won't compress: %s", r.json().get("detail", "")[:300])
        sys.exit(1)
    if r.status_code != 200:
        log.error("Dry run failed: HTTP %d %s", r.status_code, r.text[:300])
        sys.exit(1)
    pt = r.json()["usage"]["prompt_tokens"]
    log.info("Dry run OK — %d prompt tokens (threshold=36000)", pt)
    if pt < 36000:
        log.warning("Below threshold — increase LARGE_CHUNK")

    baseline = vram_mb()
    log.info("Baseline VRAM: %s MB", f"{baseline:,}")
    log.info("%-4s %10s %10s %9s  %s", "Run", "VRAM(MB)", "Drift", "Delta", "Status")
    log.info("%-4s %10s %10s %9s  %s", "-" * 4, "-" * 10, "-" * 10, "-" * 9, "-" * 30)

    samples = [baseline]
    failures = 0
    max_drift = 0
    peak = baseline

    for i in range(ITERS):
        t0 = time.monotonic()
        r = httpx.post(
            "http://localhost:9001/v1/chat/completions",
            json={"model": "pflash-qwen3.6-27B", "messages": messages, "max_tokens": 30, "stream": False},
            headers=H, timeout=300,
        )
        elapsed = time.monotonic() - t0
        v = vram_mb()
        samples.append(v)
        drift = v - baseline
        max_drift = max(max_drift, drift)
        peak = max(peak, v)
        last_delta = v - samples[-2]

        if r.status_code == 200:
            u = r.json()["usage"]
            status = f"ok ({u['prompt_tokens']}+{u['completion_tokens']}t)"
        else:
            failures += 1
            status = f"HTTP {r.status_code}"
        log.info("#%-3d %10s %10s %+9s  %s (%.1fs)", i + 1, f"{v:,}", f"{drift:+,}", f"{last_delta:+,}", status, elapsed)

        if COOLDOWN and i < ITERS - 1:
            time.sleep(COOLDOWN)

    log.info("")
    log.info("=" * 50)
    total = samples[-1] - baseline
    log.info("Baseline: %s MB  Final: %s MB  Drift: %+d MB  Max: %d MB  Peak: %d MB  Fail: %d",
             f"{baseline:,}", f"{samples[-1]:,}", total, max_drift, peak, failures)
    if total > THRESHOLD_MB:
        log.error("FAIL — drift > %d MB", THRESHOLD_MB)
        sys.exit(1)
    log.info("PASS")


if __name__ == "__main__":
    main()
