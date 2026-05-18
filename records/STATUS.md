# Current Status

**Snapshot:** 2026-05-18
**Posture:** Migration complete — all tracks green
**Focus:** Deferred items (see PLANS.md)

## Migration Summary

Bee (`Anbeeld/beellama.cpp`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse (1.93x verified). Legacy `backend:dflash` daemon, Lucebox code, and `/opt/dflash` fully retired. For completed phase details see git history (Phases 1-7, commits `f40874c` through `ef4fcb1`).

## Recent Changes

- 2026-05-18: **DFlash MAX_VERIFY_TOKENS cap fixed** — `LLAMA_DFLASH_MAX_VERIFY_TOKENS=25` was silently breaking DDTree tree-mode (3-5% acceptance). Patched with env-var `GGML_DFLASH_MAX_VERIFY_TOKENS` following existing `GGML_DFLASH_MAX_CTX` pattern. Default stays 25. Auto-derived in `model_manager.py` from n_max + branch_budget. (RSH-20260518-009, commit `2f6a345`)
- 2026-05-18: **DDTree tree-mode benchmarked** — Tree-mode works correctly after cap fix (85% acceptance) but adds only +1% throughput over flat mode for greedy decode. Draft is already well-matched; tree branches have nothing to rescue. (RSH-20260518-009)
- 2026-05-18: **Gateway overhead profiled** — Instrumented `proxy/llama.py` with `time.perf_counter()`. Gateway adds 0-60ms per request (<1%). The ~1.3s "gateway overhead" from earlier measurements was a measurement error (comparing warm bench vs production through reasoning mode). (RSH-20260518-010)
- 2026-05-18: **DFlash think-split prototyped** — Streaming SSE parser detects think/answer boundary via `reasoning_content` delta fields. Hybrid split (AR for thinking, DFlash for output) is not beneficial because re-prefilling the assistant message costs more than the DFlash speedup saves. DFlash provides consistent ~1.55x speedup regardless of thinking mode. Per-request `speculative.n_max: 0` toggle already works — no patch needed. (RSH-20260518-010)
- 2026-05-18: **DFlash VRAM overhead measured** — +2,188 MiB (+12%) vs AR-only (17.4 GB → 19.6 GB). Draft model weights (~1.84 GB) + ring buffer + tape buffers. Leaves ~4.4 GB free on 24 GB for KV cache. (RSH-20260518-010)
- 2026-05-18: **PFlash stale-thread deadlock fixed** — `PflashDaemon` replaced `loop.run_in_executor` with a dedicated compressor thread + async Queue. Verified: 100/100 consecutive PFlash-compressing iterations, zero failures, zero VRAM drift, zero deadlocks. (DEC-20260518-001, commit `78cd53b`)
- 2026-05-18: **VRAM drift soak** — `soak_vram_drift.py` added, 100-iteration soak confirmed zero VRAM drift across repeated PFlash compression cycles
- 2026-05-18: **PINNED_SHA enforcement** — Docker build now verifies cloned SHA matches `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (commit `c90d2a0`)
- 2026-05-18: Phase 4 — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0
- 2026-05-18: Phase 3 — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Hygiene cleanup — stale patches deleted, SPEC/README updated, dead BACKEND_DFLASH code removed, .dockerignore expanded
- 2026-05-18: Phase 5 — doc cleanup + speedup verification
- 2026-05-18: Phase 7 — lucebox/ deleted, pflash_daemon extracted, dflash model files removed (3.3 GB), 87 GB disk freed
