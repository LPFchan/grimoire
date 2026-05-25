# DEC-20260519-001: Migrate to MTP and atomic-llama-cpp-turboquant

Opened: 2026-05-19 18-56-47 KST
Recorded by agent: mimo-v2.5-pro-precision

## Decision

Replace all served models with MTP variants and switch the inference backend
from Bee's `beellama.cpp` (DFlash) to `AtomicBot-ai/atomic-llama-cpp-turboquant`
(NextN/MTP + TurboQuant).

## Rationale

Benchmarks (RSH-20260519-002) showed:

1. **DFlash is worse than MTP on every dimension except short-prompt decode:**
   - Prefill: DFlash 31-38% slower vs MTP 9-25% slower (both vs baseline)
   - Decode at 50k: DFlash +17%, MTP +18% (tie)
   - Decode at 100k: DFlash fails (ctx cap), MTP +28%
   - VRAM efficiency: MTP uses less overhead (+1.5 GB vs DFlash +1.9 GB)

2. **DFlash has fundamental architectural limits:**
   - `n_seq_max = 2 × parallel` halves effective context (max ~92k on 24 GB)
   - 1024-token cross-attention window collapses on long contexts
   - Draft acceptance drops from 34% (short) to 29% (50k)

3. **MTP has no context halving** — MTP heads use the full context.

4. **Bee's MTP integration is broken** (`unordered_map::at` crash in
   `llama_init_from_model`). Upstream PR #21 is unmerged and unfixed.

5. **atomic-llama-cpp-turboquant** supports:
   - **Qwen 3.6 NextN** (`--spec-type nextn`) — shared-model drafting, no second mmap
   - **Gemma 4 MTP** (`--spec-type mtp --mtp-head`) — official assistant head
   - **TurboQuant KV cache** (`-ctk turbo3 -ctv turbo3`) — ~4.3× KV compression
   - Same combined `_MTP.gguf` files we already have / can download

## Model Plan

| Role | Model | Source | Quant |
|------|-------|--------|-------|
| General chat | Qwen3.6-27B MTP | unsloth/Qwen3.6-27B-MTP-GGUF | Q4_K_M (already downloaded) |
| Abliterated | Huihui-Qwen3.6-27B MTP | huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF | Q4_K (needs verification) |
| Gemma 4 | gemma-4-31B-it | AtomicChat/gemma-4-31B-it-assistant-GGUF | UD-Q4_K_XL (existing) |
| Gemma MTP head | gemma-4-assistant | AtomicChat/gemma-4-31B-it-assistant-GGUF | Q4_K_M (353 MB) |

## Retirement

- `dflash-qwen3.6-27B`: replaced by qwen MTP
- `dflash-huihui-qwen3.6-27B`: replaced by huihui MTP
- `qwen-3.6-27B`: replaced by qwen MTP
- `pflash-qwen3.6-27B`: retired (was PFlash, replaced by MTP)
- `pflash-park-qwen3.6-27B`: retired
- Bee binary retired in favor of atomic-llama-cpp-turboquant

## Build Plan

1. Build `atomic-llama-cpp-turboquant` from source with CUDA + flash-attn
2. Download Gemma 4 assistant MTP head (Q4_K_M)
3. Verify Huihui abliterated MTP GGUF has MTP heads
4. Update `models.json` with new model entries
5. Update Docker build for new binary
