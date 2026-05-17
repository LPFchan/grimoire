# Use Bee's binary For Native DFlash Canary

Opened: 2026-05-17 18-00-00 KST
Recorded by agent: opencode

## Decision

Use Bee's `llama-server` binary (`Anbeeld/beellama.cpp`) for the native DFlash canary instead of continuing to port DFlash support onto TheTom's binary. The production `grimoire` container keeps TheTom with turboquant for baseline models.

## Context

After Phase 1 implementation (ring buffer, Bee's `dflash_draft.cpp`, `n_enc_real`, `dflash_verify_logits`, `flush_prefill`), the TheTom-based canary produces 0% draft acceptance. Bee's binary achieves 100% acceptance with the same GGUF (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`) and same target model. The root cause is a subtle cross-data or CUDA kernel difference that would take extensive C++ debugging to isolate.

The canary only uses `cache-type-k: q8_0` — no turboquant features. Bee's binary handles Q8_0 cache fine. Turboquant is only needed by the baseline production models (qwen-3.6-27B, gemma-4-31B) which run on the TheTom-based `grimoire` container.

## Options Considered

| Option | Outcome |
| --- | --- |
| Bee binary for canary (chosen) | Working DFlash with 100% acceptance. Two binaries in deployment, clean separation. |
| Continue debugging TheTom port | Unknown time cost; root cause is deep in CUDA execution path |
| Port turboquant into Bee | Massive scope — turboquant spans ggml-core, CUDA kernels, CMake, across ~20 files |

## Rationale

- Bee's binary works today with zero additional effort
- The canary doesn't need turboquant
- Production stack (TheTom + turboquant) stays unchanged
- If Bee's DFlash integration is ever upstreamed into TheTom, we can switch back

## Consequences

- Two binaries in deployment: TheTom for production, Bee for canary
- Canary launched from `/tmp/spec-analysis/bee-shallow/build/bin/llama-server`
- Model config points to `gguf/dflash-draft-3.6-q8_0.gguf` (Buun's GGUF)
- TheTom patch (`spec-dflash-contract.patch`) is frozen — no further DFlash porting work on it
