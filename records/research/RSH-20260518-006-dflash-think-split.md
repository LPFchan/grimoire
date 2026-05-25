# DFlash Think-Split Prototype and Per-Request Spec Toggle

Opened: 2026-05-18 21-00-00 KST
Recorded by agent: mimo-v2.5-pro-precision

## Question

Can we speed up DFlash during thinking/reasoning blocks by splitting the request into "think phase (AR)" + "answer phase (DFlash)"?

## Background

Production throughput dips to 7-17s per request. The model generates thinking tokens (`` blocks) before answering. DFlash speculative decoding was suspected to be ineffective during thinking because reasoning tokens are unpredictable.

## Method

Built a streaming prototype (`tests/proto_think_split.py`) that:
1. Sends a streaming chat completion request
2. Parses SSE chunks to detect `reasoning_content` → `content` transition (the think/answer boundary)
3. Saves the KV slot state at the boundary
4. Re-issues a new request with different speculative params to continue with DFlash

## Key Findings

### 1. Per-request speculation toggle already exists

Bee's `server-task.cpp` at lines 659-668 already parses speculative overrides from the JSON body:

```cpp
params.speculative.n_max = json_value(data, "speculative.n_max", defaults.speculative.n_max);
params.speculative.branch_budget = json_value(data, "speculative.branch_budget", defaults.speculative.branch_budget);
```

The field name is `speculative.n_max: 0` (not `draft_n_max: 0`). No patch needed.

### 2. DFlash is ~1.55x faster than AR regardless of thinking mode

| Config | Tokens | Time | tok/s | vs AR |
|--------|--------|------|-------|-------|
| DFlash + thinking (default) | 2048 | 34.2s | **59.9** | 1.55x |
| AR + thinking | 2048 | 52.9s | 38.7 | 1.00x |
| AR + no thinking | 1582 | 41.2s | 38.4 | 0.99x |
| DFlash + no thinking | 1649 | 28.2s | 58.5 | 1.51x |

DFlash provides a consistent ~1.55x speedup whether the model is thinking or not. The speculation is not "useless during thinking" — it works at the same factor, just the base rate is slower for both.

### 3. The "split" approach doesn't beat baseline

The hybrid (AR → DFlash split) was ~52s vs baseline ~34s. The reason: Phase 2 must re-prefill the assistant's full thinking output (~1500 tokens) before it can start generating the answer. This prefill cost negates the DFlash speedup.

### 4. Streaming detection works

The SSE streaming parser correctly identifies the `reasoning_content` → `content` transition. The think/answer boundary is detectable via delta fields in the streaming chunks.

### 5. The 7-17s production concern

The production slowdown is not DFlash being slow during thinking. It's the token budget allocation: with `max_tokens=256`, the model burns all 256 tokens on reasoning and produces little or no visible answer. With adequate budget (2048), the model produces both reasoning and answer at DFlash speed (~60 tok/s).

## Conclusion

- **No split needed.** DFlash works at 1.55x during thinking. Just let it run.
- **Use adequate max_tokens** (2048+) so the model has room to think AND answer.
- Per-request `speculative.n_max: 0` toggle exists and works for cases where you need pure AR (e.g., debugging, or if DFlash introduces quality issues).

## DFlash VRAM Overhead

Measured by starting llama-server with and without `--spec-type dflash` on RTX 3090 (24 GB). Both use Qwen3.6-27B Q4_K_M, q8_0 KV cache, ctx=32000.

| Config | VRAM | vs AR |
|--------|------|-------|
| AR only (no speculation) | 17,412 MiB (17.0 GB) | — |
| DFlash + draft model | 19,600 MiB (19.1 GB) | **+2,188 MiB (+12%)** |

Breakdown of the 2.14 GB overhead:
- Draft model weights (dflash-draft-3.6-q8_0.gguf): ~1.84 GB
- Cross-attention ring buffer (5 layers × 1024 slots × 5120 embd): ~100 MB
- GPU tape + rollback buffers: ~200 MB

DFlash leaves ~4.4 GB free on 24 GB for KV cache (~18K tokens at q8_0). Well within budget for typical workloads.

## Summary

| Aspect | Finding |
|--------|---------|
| DFlash speedup (thinking) | 1.55x vs AR |
| DFlash speedup (no thinking) | 1.55x vs AR |
| Per-request toggle | `speculative.n_max: 0` works, no patch needed |
| Think-split feasibility | Not beneficial — re-prefill cost > DFlash gain |
| VRAM overhead | +2.14 GB (+12%) |
| Fits on 24 GB? | Yes (19.6 GB, 4.4 GB free) |

## Related

- `tests/proto_think_split.py`: prototype script
- `tools/server/server-task.cpp:659-668`: per-request speculative param parsing
- Common/arg.cpp: CLI arg definitions for speculative params
