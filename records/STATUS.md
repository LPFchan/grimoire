# Current Status

**Snapshot:** 2026-05-18
**Posture:** Phase 4 complete — all tracks green
**Focus:** Deferred items (VMM park, multi-spec, tape recording)

## Current State Summary

Bee (`Anbeeld/beellama.cpp`, SHA `2b9aa77`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Legacy `backend:dflash` daemon retired. `pflash_daemon` extracted from `lucebox/` to `src/grimoire/pflash/` with trimmed CMakeLists. `lucebox/` and all dormant Python modules deleted.

## Active Tracks

### Track A: Canonical DFlash (Phase 1-2) ✅
Complete. Bee binary deployed, GPU ring fix upstreamed, web UI patches ported.

### Track B: Content-hash KV Caching (Phase 3) ✅
- **Goal:** Cross-conversation sysprompt caching for coding-agent use-cases
- **Result:** Verified — 1.93x prompt speedup (289ms → 150ms) with 18 cached tokens across different conversation_ids
- **Exit criteria met:** Same sysprompt across different conversation_ids → cache hit → skip re-prefill

### Track C: Preserved PFlash (Phase 4) ✅
- PFlash compression on `backend:llama` via `pflash_daemon`. Models start and serve.
- **Bug (fixed):** Async race — daemon/pcfg captured before first `await` in `_proxy_chat`
- **Enhancement:** Catastrophic HTTP 413 when compression required but declines
- Standard `pflash-qwen3.6-27B` and `pflash-park-qwen3.6-27B` both verified.

## Recent Changes

- 2026-05-18: Phase 3 complete — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Phase 5 complete — doc cleanup + speedup verification
- 2026-05-18: Hygiene cleanup — deleted stale patches (native in Bee), fixed SPEC/README, removed dead BACKEND_DFLASH code, PINNED_SHA enforcement in Dockerfile, expanded .dockerignore
- 2026-05-18: Phase 4 complete — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0

## Immediate Next Steps

1. ✅ Phase 1-2: DFlash pipeline + server integration (DONE)
2. ✅ Phase 7: Legacy cleanup (DONE)
3. ✅ Phase 3: Content-hash KV caching (DONE — 1.93x speedup verified)
4. ✅ Phase 5: Remaining optimization (DONE)
5. ✅ Phase 4: Preserved PFlash parity (DONE — daemon propagation fixed, 413 on compression failure)
