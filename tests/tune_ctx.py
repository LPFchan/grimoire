#!/usr/bin/env python3
"""Binary search for max stable ctx-size on pflash-qwen3.6-27B."""
import json, subprocess, sys, time
from pathlib import Path
import httpx

BASE = "http://localhost:9001"
KEY = "7JcW7xX82ypTQPlsYle6XdjlBSWfG3NwbtYSRSXZQ88"
MODEL = "pflash-qwen3.6-27B"
FIXTURES = Path("/home/yeowool/opencode_splits")
CONFIG = Path("/home/yeowool/grimoire/etc/models.json")
GRIMOIRE = Path("/home/yeowool/grimoire")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def build_prompt(chars):
    sessions = sorted(FIXTURES.glob("opencode_ses_*.json"),
                      key=lambda f: f.stat().st_size, reverse=True)
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
    return "\n\n".join(texts)[:chars]


def set_ctx(ctx):
    data = json.loads(CONFIG.read_text())
    data["models"]["pflash-qwen3.6-27B"]["ctx-size"] = ctx
    data["models"]["pflash-qwen3.6-27B"]["max-effective-context"] = ctx
    CONFIG.write_text(json.dumps(data, indent=2))


def restart():
    subprocess.run(
        ["/usr/bin/docker", "compose", "-f", "docker-compose.yml", "restart"],
        cwd=str(GRIMOIRE), capture_output=True, timeout=30)
    time.sleep(3)
    for _ in range(30):
        time.sleep(3)
        try:
            if httpx.get(f"{BASE}/health", timeout=5).json().get("status") == "healthy":
                return True
        except Exception:
            pass
    return False


def load_model():
    httpx.post(f"{BASE}/switch/{MODEL}", headers=H, timeout=30)
    for _ in range(60):
        time.sleep(3)
        try:
            r = httpx.get(f"{BASE}/health", timeout=5)
            if MODEL in r.json().get("active_models", []):
                time.sleep(5)
                return True
        except Exception:
            pass
    return False


def stop_model():
    httpx.post(f"{BASE}/stop/{MODEL}", headers=H, timeout=10)
    time.sleep(3)


def gpu_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def test_ctx(ctx):
    set_ctx(ctx)
    if not restart():
        return None
    if not load_model():
        return None

    chars = int(ctx * 0.9 * 2)
    prompt = build_prompt(chars)
    vram_before = gpu_mb()

    t0 = time.monotonic()
    try:
        r = httpx.post(f"{BASE}/v1/chat/completions",
                       json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": 30, "stream": False},
                       headers=H, timeout=300)
    except Exception as e:
        stop_model()
        return False
    elapsed = time.monotonic() - t0
    vram_after = gpu_mb()

    success = False
    pt = 0
    ct = 0
    if r.status_code == 200 and r.text:
        try:
            d = r.json()
            if "error" not in d and d.get("choices"):
                c = d["choices"][0].get("message", {}).get("content", "")
                if c.strip() and d.get("usage", {}).get("prompt_tokens", 0) > 0:
                    success = True
            pt = d.get("usage", {}).get("prompt_tokens", 0)
            ct = d.get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass

    print(f"  ctx={ctx:>6,} -> {'PASS' if success else 'FAIL'}  "
          f"prompt={pt:,} completion={ct:,}  "
          f"VRAM={vram_before}->{vram_after}MB  {elapsed:.1f}s", flush=True)
    if not success and r.status_code != 200:
        try:
            err = r.json().get("detail", r.text[:200])
        except Exception:
            err = r.text[:200] if r.text else "empty"
        print(f"    HTTP {r.status_code}: {err}", flush=True)

    stop_model()
    time.sleep(5)
    return success


def main():
    lo, hi = int(sys.argv[1]) if len(sys.argv) > 1 else 10000, \
             int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    print(f"Binary search pflash-qwen3.6-27B ctx-size: [{lo:,}, {hi:,}]")
    print(f"=" * 55)

    results = {}

    # Test lo
    r = test_ctx(lo)
    if r is None:
        print(f"SETUP FAIL at {lo}. Aborting.")
        return
    results[lo] = r
    if not r:
        print(f"Lower bound {lo} FAILED. Try smaller value.")
        return

    max_ok = lo

    # Test hi
    r = test_ctx(hi)
    if r is None:
        hi = max_ok
    else:
        results[hi] = r
        if r:
            print(f"Upper bound {hi} PASSES! Max ctx >= {hi}")
            return
        else:
            print(f"Upper bound {hi} FAILS")

    # Binary search
    while lo < hi - 1000:
        mid = ((lo + hi) // 2000) * 1000
        if mid <= lo:
            mid = lo + 1000
        print(f"\n[range {lo:,}..{hi:,}] testing mid={mid:,}")
        r = test_ctx(mid)
        if r is None:
            hi = mid
            continue
        results[mid] = r
        if r:
            lo = mid
            max_ok = mid
        else:
            hi = mid

    # Fine tune
    for delta in [5000, 3000, 2000, 1000]:
        test_val = max_ok + delta
        if test_val >= hi:
            continue
        print(f"\n[fine-tune +{delta:,}] testing {test_val:,}")
        r = test_ctx(test_val)
        if r is None:
            continue
        results[test_val] = r
        if r:
            max_ok = test_val

    kv_gb = max_ok * 34816 / 1024**3
    print(f"\n{'='*55}")
    print(f"MAX ctx-size: {max_ok:,} tokens")
    print(f"KV cache:     {kv_gb:.1f} GB @ q8_0")
    print(f"\nAll results:")
    for ctx in sorted(results):
        ok = results[ctx]
        kvg = ctx * 34816 / 1024**3
        print(f"  ctx={ctx:>6,}  {'PASS' if ok else 'FAIL'}  KV={kvg:.1f}GB")


if __name__ == "__main__":
    main()
