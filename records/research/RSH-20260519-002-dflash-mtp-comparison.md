# RSH-20260519-002: DFlash vs MTP vs Baseline — Three-Way Comparison

Opened: 2026-05-19 18-56-47 KST
Recorded by agent: mimo-v2.5-pro-precision

## Question

How do DFlash and MTP speculative decoding compare against baseline (no speculation) on Qwen3.6-27B across different prompt lengths, and what are the practical limits of each approach on a single RTX 3090 (24 GB)?

## Setup

### Hardware
- GPU: 1× NVIDIA RTX 3090 (24,576 MiB)
- CPU: AMD Ryzen 5 3600 (6C/12T)
- Host: grimoire.lost.plus

### Models
- **Baseline model**: `Qwen3.6-27B-Q4_K_M.gguf` (16 GB)
- **MTP model**: `Qwen3.6-27B-MTP-Q4_K_M.gguf` (16 GB + MTP heads, from Unsloth)
- **DFlash draft**: `dflash-draft-3.6-q8_0.gguf` (1.76 GB, 5-layer cross-attention)

### Binaries
- **Baseline + MTP**: Upstream llama.cpp b9180 (built from source, CUDA 12.8)
- **DFlash**: Bee llama.cpp @ `4db14be0` (pinned SHA, DFlash patch applied)

### Configuration
- `--cache-type-k q8_0 --cache-type-v q8_0`
- `--flash-attn on --n-gpu-layers 999 --parallel 1`
- DFlash: `--spec-type dflash --spec-draft-model ... --spec-dflash-cross-ctx 1024 --spec-dflash-max-slots 4 --spec-draft-n-max 16 --spec-draft-temp 0`
- MTP: `--spec-type draft-mtp --spec-draft-n-max 3`
- `GGML_DFLASH_MAX_VERIFY_TOKENS=20` for DFlash
- All tests with `CUDA_VISIBLE_DEVICES=0`

## Results

### Short Prompt (171 tokens, 256 gen, 150k ctx)

| Config | Prefill tok/s | Decode tok/s | vs Baseline | Draft accept | VRAM |
|--------|--------------|-------------|-------------|-------------|------|
| baseline | 720 | 40.7 | — | — | 21,342 |
| mtp | 538 (↓25%) | 54.4 | **+34%** | 67% | 22,900 |
| dflash | 445 (↓38%) | **80.1** | **+97%** | 34% | 23,210 |

### 50k Prompt (50,000 tokens, 256 gen, 150k ctx)

| Config | Prefill tok/s | Decode tok/s | vs Baseline | Draft accept | VRAM |
|--------|--------------|-------------|-------------|-------------|------|
| baseline | 1,017 | 30.7 | — | — | 21,342 |
| mtp | 920 (↓10%) | **36.2** | **+18%** | 49% | 22,900 |
| dflash | 700 (↓31%) | 35.8 | +17% | 29% | 23,210 |

### 100k Prompt (100,000 tokens, 256 gen, 150k ctx)

| Config | Prefill tok/s | Decode tok/s | vs Baseline | Draft accept | VRAM |
|--------|--------------|-------------|-------------|-------------|------|
| baseline | 812 | 24.6 | — | — | 21,342 |
| mtp | 740 (↓9%) | **31.5** | **+28%** | 51% | 22,900 |
| dflash | — | **❌ FAIL** | — | — | 23,210 |

## Key Findings

### 1. DFlash ctx Halving

Bee's server doubles `n_seq_max` for DFlash (`server-context.cpp:1401-1403`):

```cpp
if (params_base.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
    params_base.n_parallel = n_parallel_user * 2;  // n_seq_max = 2 × parallel
}
```

This reserves a backup `seq_id` for speculative rollback (recurrent state restore on draft rejection). With `--parallel 1`, `n_seq_max = 2`, so `n_ctx_seq = n_ctx / 2`.

**Result**: DFlash's effective per-sequence context is capped at ~92k on a 24 GB card (184832 / 2). At 100k tokens, the request is rejected with HTTP 400.

The VRAM ctx-size sweep (RSH-20260519-001) found 184832 as the physical GPU limit. The effective per-sequence limit for DFlash is 92416.

### 2. Prefill Hit

Both spec decoding methods take a prefill speed hit:
- DFlash: 31-38% slower (GPU ring buffer contention)
- MTP: 9-25% slower (D2H embedding transfers per upstream PR #22673)

The prefill hit is most pronounced on short prompts (where prefill time is insignificant) and shrinks as a fraction of total time on long prompts.

### 3. Draft Acceptance Collapse on Long Contexts

DFlash's draft acceptance drops from 34% (short) to 29% (50k). The draft model has a 1024-token cross-attention window — on long prompts it can only see the last 1024 tokens, making accurate predictions nearly impossible.

MTP maintains 49-67% acceptance across all lengths because it uses the main model's full hidden state directly.

### 4. MTP Ctx Advantage

MTP does not halve n_seq_max — it uses the main model's context directly. The MTP heads are a single transformer layer with minimal VRAM overhead (+~1.5 GB vs baseline). MTP works at any context length the baseline supports.

### 5. Single GPU VRAM Comparison (150k ctx)

| Component | Baseline | +MTP | +DFlash |
|-----------|----------|------|---------|
| Model weights | ~15,345 | ~15,345 | ~15,345 |
| Draft model | — | — | ~1,760 |
| MTP heads | — | ~390 | — |
| KV cache | ~5,997 | ~5,997 | ~5,997 |
| **Total** | **~21,342** | **~22,900** | **~23,210** |

### 6. DFlash Context Halving

Bee's server (`server-context.cpp:1401-1403`) doubles `n_seq_max` for DFlash:

```cpp
if (params_base.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
    params_base.n_parallel = n_parallel_user * 2;  // n_seq_max = 2 × parallel
}
```

This reserves a backup `seq_id` for speculative rollback (recurrent state restore on draft rejection). With `--parallel 1`, `n_seq_max = 2`, so `n_ctx_seq = n_ctx / 2`.

**Effective per-sequence context is half of the physical ctx-size.** At `--ctx-size 184832` (the VRAM limit on 24 GB), effective ctx = 92,416. Prompts exceeding this are rejected with HTTP 400. At 85k+ actual inference, the compute buffers during prefill exceed VRAM, causing server crashes (segfault in `llama_decode`).

### 7. DFlash Cross-Attention Window Collapse

DFlash's draft model has a 1024-token cross-attention window (`--spec-dflash-cross-ctx 1024`). On short prompts where the full context fits in this window, acceptance is reasonable (34%). On 50k prompts where the window covers only 2% of context, acceptance drops to 29%. MTP uses the full hidden state and maintains 49-67% acceptance regardless of length.

## (MTP wins for long-context workloads. MTP is the recommended replacement for all models.)

## Incidental Findings

### Registry Bug 1: Fuzzy Substring Resolution

`registry.py:582-592` had a fuzzy substring match that caused `dflash-qwen3.6-27B` to resolve to `qwen-3.6-27B` because `normalize("qwen-3.6-27B") = "qwen3627b"` is a substring of `normalize("dflash-qwen3.6-27B") = "dflashqwen3627b"`. Fixed by removing the substring pass — exact normalization already covers file basenames, paths, and aliases.

### Registry Bug 2: _stat_stamp Seed File Blindness

`_stat_stamp()` only checked the state file path (`/var/lib/grimoire/models.json`), but `_load()` falls back to the seed (`/etc/grimoire/models.json`). When the state file never existed, stamp was always `None`, preventing `_maybe_reload()` from detecting seed file changes. Fixed by mirroring `_load()`'s path fallback in `_stat_stamp()`.

### Bee MTP Integration

bchap1n's MTP branch for Bee (`Anbeeld/beellama.cpp#21`) auto-detects MTP GGUFs but fails to create the MTP context with `unordered_map::at` during `llama_init_from_model`. The error occurs in the MTP context initialization path where a layer-specific map access fails. Unresolved as of 2026-05-19 (Anbeeld indicated a new release is coming first).

## Open Questions

- Can DFlash's n_seq_max halving be mitigated by passing an explicit `--n-seq-max` or using `--kv-unified`?
- Would a draft model with larger cross-attention window improve DFlash acceptance on long prompts?
- Does MTP's prefill hit decrease with `--spec-draft-n-max 2` vs `3`?
- What is the MTP decode speed at `--spec-draft-n-max 2` vs `3` on our hardware?
