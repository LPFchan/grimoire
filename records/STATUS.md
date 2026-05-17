# Current Status

**Snapshot:** 2026-05-18
**Posture:** Phase 3 active — content-hash KV caching
**Focus:** Docker rebuild + canary verification

## Current State Summary

Bee (`Anbeeld/beellama.cpp`, SHA `2b9aa77`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Legacy `backend:dflash` daemon retired. `pflash_daemon` extracted from `lucebox/` to `src/pflash/` with trimmed CMakeLists. `lucebox/` and all dormant Python modules deleted.

## Active Tracks

### Track A: Canonical DFlash (Phase 1-2) ✅
Complete. Bee binary deployed, GPU ring fix upstreamed, web UI patches ported.

### Track B: Content-hash KV Caching (Phase 3) 🔄
- **Goal:** Cross-conversation sysprompt caching for coding-agent use-cases
- **Status:** Code complete — `KVCacheStore` class, wired into `proxy/llama.py` with content-hash based save/restore, canary model config updated, test suite rewritten (`test_kv_cache_store.py`)
- **Remaining:** Docker rebuild + deploy, verify coding-agent sysprompt caching works across different conversation_ids
- **Exit criteria:** Same 20K sysprompt across different conversation_ids → cache hit → no re-prefill

### Track C: Preserved PFlash (Phase 4) 🔜
- PFlash compression on `backend:llama` via `pflash_daemon`. Unchanged — park/unpark, slot save/restore, warm/cold split all intact.
- Depends on: Phase 3 sign-off

## Recent Changes

- 2026-05-18: Legacy cleanup — extracted `pflash_daemon` → `src/pflash/`, deleted `lucebox/`, `proxy/dflash.py`, `snapshot_swap.py`, `session_kv.py`, `prefix_cache.py`, `DflashDaemon`
- 2026-05-18: Phase 3 — `KVCacheStore` class with RAM→disk tiering, LRU eviction, TTL, manifest; wired into `proxy/llama.py` with content-hash based save/restore; test rewrite (`test_kv_cache_store.py`)
- 2026-05-18: Dockerfile updated — `dflash-build` → `pflash-build` (from `src/pflash/`)
- 2026-05-18: models.json — removed `dflash-pflash-qwen3.6-27B`, added `kv-cache-*` config to canary
- 2026-05-18: Disk cleanup — deleted `/home/yeowool/models/dflash/` (3.3 GB), old temp files
- 2026-05-18: Full disk: 91% → 54% across all cleanup rounds (~87 GB freed)

## Immediate Next Steps

1. ✅ Phase 1-2: DFlash pipeline + server integration (DONE)
2. ✅ Phase 7: Legacy cleanup (DONE)
3. 🔄 Phase 3: Content-hash KV caching (code done, needs rebuild + verification)
4. Phase 4: Preserved PFlash parity
