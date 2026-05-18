"""Think-split prototype using KV slot save/restore.

Phase 1: Streaming request → reads tokens, detects when reasoning ends
Phase 2: Save KV slot, restore with speculation params changed → continues at DFlash speed

Requires bench server with --slot-save-path /dev/shm/grimoire-slots
"""

import json
import time
import urllib.request
import httpx
import re

PORT = 8082
MODEL = "Qwen3.6-27B-Q4_K_M.gguf"
PROMPT = "Explain the difference between TCP and UDP in detail, with examples."
MAX_TOKENS = 2048
SLOT_URL = f"http://127.0.0.1:{PORT}/slots/0"


def _req_json(body):
    """Send a request and get parsed JSON response."""
    data = json.dumps(body).encode()
    resp = urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data, timeout=300)
    return json.loads(resp.read().decode(errors="replace"), strict=False)


def _slot_save(hash_key="think_split"):
    """Save the current KV slot state."""
    try:
        body = json.dumps({"filename": hash_key}).encode()
        resp = urllib.request.urlopen(
            f"{SLOT_URL}?action=save", data=body, timeout=10)
        return resp.status == 200
    except Exception as e:
        print(f"  slot save failed: {e}")
        return False


def _slot_restore(hash_key="think_split"):
    """Restore a saved KV slot state."""
    try:
        body = json.dumps({"filename": hash_key}).encode()
        resp = urllib.request.urlopen(
            f"{SLOT_URL}?action=restore", data=body, timeout=10)
        return resp.status == 200
    except Exception as e:
        print(f"  slot restore failed: {e}")
        return False


def _slot_erase(hash_key="think_split"):
    try:
        body = json.dumps({"filename": hash_key}).encode()
        urllib.request.urlopen(f"{SLOT_URL}?action=erase", data=body, timeout=5)
    except Exception:
        pass


def strategy_baseline():
    r = _req_json({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    })
    tok = r["usage"]["completion_tokens"]
    return {"tokens": tok, "response": r}


def strategy_no_reasoning():
    r = _req_json({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    tok = r["usage"]["completion_tokens"]
    return {"tokens": tok, "response": r}


def strategy_hybrid_slot():
    """Phase 1: get thinking with no speculation → save slot.
       Phase 2: restore slot with speculation → continue (output-only)."""

    # Phase 1: AR request (no speculation), stream to detect when thinking ends  
    _slot_erase()
    t0 = time.time()

    # Send streaming request to detect the think→answer transition
    # We'll use httpx for streaming
    import httpx
    body1 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }

    thought_tokens = []
    answer_tokens = []
    in_reasoning = False
    transition_detected = False

    with httpx.Client(timeout=300) as client:
        with client.stream("POST",
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=body1) as resp:
            for line in resp.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [{}])
                    delta = choices[0].get("delta", {})
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        in_reasoning = True
                        thought_tokens.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"]:
                        if in_reasoning:
                            # transition detected: reasoning → content
                            transition_detected = True
                            in_reasoning = False
                            answer_tokens.append(delta["content"])
                        else:
                            answer_tokens.append(delta["content"])

                    finish = choices[0].get("finish_reason", "")
                    if finish == "length" and in_reasoning:
                        # Hit max_tokens during reasoning — no answer yet
                        pass

    t1 = time.time()
    thought_text = "".join(thought_tokens)
    answer_text = "".join(answer_tokens)
    phase1_time = t1 - t0

    if not transition_detected or not answer_text:
        # Model didn't finish thinking within token budget
        return {
            "label": "Hybrid (no split)",
            "time": phase1_time,
            "tokens": len(thought_tokens) + len(answer_tokens),
            "thought_chars": len(thought_text),
            "answer_chars": len(answer_text),
            "note": "no transition detected",
        }

    # Phase 2: Save KV slot and continue with DFlash
    # Since we can't change speculation mode mid-stream, we need to
    # re-issue a request with speculation params. But slot save/restore
    # means the model state is preserved.

    # Actually, slot save/restore doesn't change speculation mode.
    # We need to RESTORE the slot and then issue a NEW request with
    # different params. The new request will use the restored KV state
    # but can have different speculative settings.

    # Issue: Bee doesn't support per-request speculation toggling.
    # So this approach is blocked unless we modify Bee's server code.

    return {
        "label": "Hybrid (slot attempt)",
        "time": phase1_time,
        "tokens": len(thought_tokens) + len(answer_tokens),
        "thought_chars": len(thought_text),
        "answer_chars": len(answer_text),
        "note": "slot save/restore blocked: no per-request spec toggle",
    }


def main():
    print("=" * 60)
    print("Think-Split Prototype")
    print("=" * 60)
    print()

    # Warmup
    for _ in range(2):
        try:
            _req_json({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 2, "temperature": 0})
        except Exception:
            pass
    time.sleep(2)

    # Strategy 1: Baseline (normal, DFlash everywhere)
    print("1. Baseline (normal request)...")
    t0 = time.time()
    r1 = strategy_baseline()
    t_baseline = time.time() - t0
    tok1 = r1["tokens"]
    reasoning1 = r1["response"]["choices"][0].get("message", {}).get("reasoning_content", "") or ""
    content1 = r1["response"]["choices"][0]["message"].get("content", "") or ""
    print(f"   {tok1} tokens in {t_baseline:.1f}s = {tok1/t_baseline:.1f} tok/s")
    print(f"   reasoning: {len(reasoning1)} chars, content: {len(content1)} chars")
    print()

    time.sleep(3)

    # Strategy 2: No reasoning (thinking disabled, pure DFlash on answer)
    print("2. No reasoning...")
    t0 = time.time()
    r2 = strategy_no_reasoning()
    t_noreason = time.time() - t0
    tok2 = r2["tokens"]
    content2 = r2["response"]["choices"][0]["message"].get("content", "") or ""
    print(f"   {tok2} tokens in {t_noreason:.1f}s = {tok2/t_noreason:.1f} tok/s")
    print(f"   content: {len(content2)} chars")
    print()

    time.sleep(3)

    # Strategy 3: Hybrid with streaming detection
    print("3. Hybrid (stream + detect thinking)...")
    r3 = strategy_hybrid_slot()
    print(f"   {r3['tokens']} tokens in {r3['time']:.1f}s = {r3['tokens']/r3['time']:.1f} tok/s")
    print(f"   thought: {r3.get('thought_chars',0)} chars, answer: {r3.get('answer_chars',0)} chars")
    print(f"   note: {r3.get('note','')}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1. Baseline:        {tok1/t_baseline:.1f} tok/s ({t_baseline:.1f}s, {tok1} tok)")
    print(f"2. No reasoning:   {tok2/t_noreason:.1f} tok/s ({t_noreason:.1f}s, {tok2} tok)")
    print(f"   Speedup:        {tok2/t_noreason / (tok1/t_baseline):.2f}x")
    if r3.get('time'):
        print(f"3. Hybrid:          {r3['tokens']/r3['time']:.1f} tok/s")
    print()
    print("Conclusion: No-reasoning mode gives the best speedup (DFlash at 2x).")
    print("The hybrid split is blocked by lack of per-request speculation toggle")
    print("in Bee's server API. One of these is needed:")
    print("  a) Add draft_n_max parsing to server-task.cpp::params_from_json_cmpl")
    print("  b) Or run separate AR + DFlash server instances")
    print("  c) Or accept that thinking is slow and use no-reasoning when possible")


if __name__ == "__main__":
    main()
