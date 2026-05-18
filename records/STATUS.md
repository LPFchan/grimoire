# Current Status

**Snapshot:** 2026-05-18
**Posture:** Migration complete — all tracks green
**Focus:** Deferred items (see PLANS.md)

## Migration Summary

Bee (`Anbeeld/beellama.cpp`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse (1.93x verified). Legacy `backend:dflash` daemon, Lucebox code, and `/opt/dflash` fully retired. For completed phase details see git history (Phases 1-7, commits `f40874c` through `ef4fcb1`).

## Recent Changes

- 2026-05-18: **PFlash stale-thread deadlock fixed** — `PflashDaemon` replaced `loop.run_in_executor` with a dedicated compressor thread + async Queue. Verified: 100/100 consecutive PFlash-compressing iterations, zero failures, zero VRAM drift, zero deadlocks. (DEC-20260518-001, commit `78cd53b`)
- 2026-05-18: **VRAM drift soak** — `soak_vram_drift.py` added, 100-iteration soak confirmed zero VRAM drift across repeated PFlash compression cycles
- 2026-05-18: **PINNED_SHA enforcement** — Docker build now verifies cloned SHA matches `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (commit `c90d2a0`)
- 2026-05-18: Phase 4 — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0
- 2026-05-18: Phase 3 — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Hygiene cleanup — stale patches deleted, SPEC/README updated, dead BACKEND_DFLASH code removed, .dockerignore expanded
- 2026-05-18: Phase 5 — doc cleanup + speedup verification
- 2026-05-18: Phase 7 — lucebox/ deleted, pflash_daemon extracted, dflash model files removed (3.3 GB), 87 GB disk freed
