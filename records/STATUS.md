# Current Status

**Snapshot:** 2026-05-30
**Posture:** vLLM migration prototyping active — llmcompressor quantization pipeline validated, vLLM serving confirmed with commercial AWQ models
**Focus:** Complete AWQ quantization for all 5 target models; resolve vLLM loading config for custom-quantized models

## Migration Summary

Bee (`Anbeeld/beellama.cpp`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse (1.93x verified). Legacy `backend:dflash` daemon, Lucebox code, and `/opt/dflash` fully retired. For completed phase details see git history (Phases 1-7, commits `f40874c` through `ef4fcb1`).

## Atomic CUDA FA V2 Track

Patch chain in `patches/atomic-llama-cpp/`, applied in order by `Dockerfile`:

| Default-served | File | Concern |
| --- | --- | --- |
| ✓ | `0002-cuda-fa-v2-scratch-owner.patch` | V2 graph-safe FA scratch owner with recoverable failure |
| ✓ | `0004-pool-flush-on-oom.patch` | Legacy pool flush-and-retry on OOM (backport of upstream PR #22155) |
|  | `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | V1, rollback only |
|  | `0003-cuda-fa-view_src-sizing.patch` | view_src sizing (closed upstream PR #23620), kept for slow-VRAM-creep regime if it appears |

`GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON` is the Dockerfile default. The trail of investigation and the decision rationale are in `records/research/RSH-20260523-001`, `RSH-20260526-001` (validation matrix), `RSH-20260526-002` (V3 design, deferred), and `RSH-20260526-003` (V2+ON shipping decision with the bench data and upstream survey).

## Recent Changes

- 2026-05-30: **vLLM migration prototype — llmcompressor quantization pipeline validated** (RSH-20260529-006, DEC-20260528-001). Driver upgraded to 580.159.04 for CUDA 13. vLLM 0.21.0 serving commercial AWQ model `Qwen3.6-27B-AWQ-INT4` on GPU 1. llmcompressor quantizes BF16 → AWQ (19.17 GB from 52 GB). vLLM loading blocked on post-quantization config patching (Marlin kernel shape requirements, missing multimodal weights).
- 2026-05-26: **Prod Dockerfile default flipped to V2+ON + 0004** (RSH-20260526-003). Pending: rebuild + recreate `grimoire:local` container.


- 2026-05-18: **DFlash MAX_VERIFY_TOKENS cap fixed** — `LLAMA_DFLASH_MAX_VERIFY_TOKENS=25` was silently breaking DDTree tree-mode (3-5% acceptance). Patched with env-var `GGML_DFLASH_MAX_VERIFY_TOKENS` following existing `GGML_DFLASH_MAX_CTX` pattern. Default stays 25. Auto-derived in `model_manager.py` from n_max + branch_budget. (RSH-20260518-005, commit `2f6a345`)
- 2026-05-18: **DDTree tree-mode benchmarked** — Tree-mode works correctly after cap fix (85% acceptance) but adds only +1% throughput over flat mode for greedy decode. Draft is already well-matched; tree branches have nothing to rescue. (RSH-20260518-005)
- 2026-05-18: **Gateway overhead profiled** — Instrumented `proxy/llama.py` with `time.perf_counter()`. Gateway adds 0-60ms per request (<1%). The ~1.3s "gateway overhead" from earlier measurements was a measurement error (comparing warm bench vs production through reasoning mode). (RSH-20260518-006)
- 2026-05-18: **DFlash think-split prototyped** — Streaming SSE parser detects think/answer boundary via `reasoning_content` delta fields. Hybrid split (AR for thinking, DFlash for output) is not beneficial because re-prefilling the assistant message costs more than the DFlash speedup saves. DFlash provides consistent ~1.55x speedup regardless of thinking mode. Per-request `speculative.n_max: 0` toggle already works — no patch needed. (RSH-20260518-006)
- 2026-05-18: **DFlash VRAM overhead measured** — +2,188 MiB (+12%) vs AR-only (17.4 GB → 19.6 GB). Draft model weights (~1.84 GB) + ring buffer + tape buffers. Leaves ~4.4 GB free on 24 GB for KV cache. (RSH-20260518-006)
- 2026-05-18: **PFlash stale-thread deadlock fixed** — `PflashDaemon` replaced `loop.run_in_executor` with a dedicated compressor thread + async Queue. Verified: 100/100 consecutive PFlash-compressing iterations, zero failures, zero VRAM drift, zero deadlocks. (DEC-20260518-001, commit `78cd53b`)
- 2026-05-18: **VRAM drift soak** — `soak_vram_drift.py` added, 100-iteration soak confirmed zero VRAM drift across repeated PFlash compression cycles
- 2026-05-18: **PINNED_SHA enforcement** — Docker build now verifies cloned SHA matches `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (commit `c90d2a0`)
- 2026-05-18: Phase 4 — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0
- 2026-05-18: Phase 3 — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Hygiene cleanup — stale patches deleted, SPEC/README updated, dead BACKEND_DFLASH code removed, .dockerignore expanded
- 2026-05-18: Phase 5 — doc cleanup + speedup verification
- 2026-05-18: Phase 7 — lucebox/ deleted, pflash_daemon extracted, dflash model files removed (3.3 GB), 87 GB disk freed
