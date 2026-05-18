# Plans

## Approved Directions

### Canonical Stack
- **Outcome:** Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash, PFlash, and non-DFlash traffic
- **Why accepted:** Bee has both turboquant and DFlash built in. GPU ring + turbo4 hang fixed upstream. Legacy `backend:dflash` daemon retired.
- **Value:** One binary to build, deploy, and maintain. Content-hash KV caching for cross-conversation sysprompt reuse.

## Sequencing

### Near term

- [x] Phase 1: Core DFlash decode pipeline — ring buffer, Bee's dflash_draft.cpp
- [x] Phase 1.5: Binary comparison — Bee 100%, TheTom 0%. Decision: full Bee stack.
- [x] Phase 1.6: Fix GPU ring + turbo4 hang — merged upstream as PR #19 (commit `0ef12a5`)
- [x] Phase 2: Server integration — Bee binary deployed, web UI patches, Docker build
- [x] Phase 7: Legacy cleanup
  - [x] Extract pflash_daemon → `src/grimoire/pflash/` (trimmed CMakeLists, removed DFlash-only sources)
  - [x] Remove `lucebox/`, `proxy/dflash.py`, `snapshot_swap.py`, `session_kv.py`, `prefix_cache.py`
  - [x] Remove `DflashDaemon` from `daemon.py`, update config.py, Dockerfile
  - [x] Remove `dflash-pflash-qwen3.6-27B` model entry, delete dflash model files (3.3 GB)
- [x] Phase 3: Content-hash KV caching
  - [x] 3.1 KVCacheStore class (RAM→disk tiering, LRU eviction, TTL, manifest)
  - [x] 3.2 Wire into proxy/llama.py (hash-based save/restore, prompt tokenization)
  - [x] 3.3 Canary model config (kv-cache-disk-dir, budget, cap)
  - [x] 3.4 Test suite rewrite: test_kv_cache_store.py, fixed e2e/stress tests
  - [x] 3.5 Docker rebuild and deploy
  - [x] 3.6 Verify coding-agent sysprompt caching across conversations (1.93x speedup)

### Mid term

**Phase 4 — Preserved PFlash Parity** ✅
- [x] Models start and serve basic chat completions
- [x] pflash_daemon runs from `/opt/pflash/`
- [x] Standard and park-unpark variants verified
- [x] PFlash compression path: async race fix (capture daemon/pcfg before first await)
- [x] Catastrophic failure: HTTP 413 when compression required but declines

**Phase 5 — Remaining Optimization**
- [x] Block-aware long-prompt integration — done (in `prefill.py`)
- [x] Model registry test-harness cleanup — done (SPEC.md, README.md, test fixes)
- [x] Verify: short-prompt decode speedup >1.5x — confirmed 1.93x

### Deferred

- GPU tape recording (`dflash_tape_*`) — only needed for tree-mode DDTree verify
- Multi-spec batched decode — single-spec sufficient for MVP
- VMM-based park/unpark, warm-turn detection, KV slot reuse

## Final Gates

1. Canonical base: Bee (`Anbeeld/beellama.cpp`)
2. DFlash decode parity green for `dflash-canary-qwen3.6-27B`
3. Content-hash KV caching green (cross-conversation sysprompt reuse, disk mirror, restart resilience) — ✅ verified 1.93x speedup
4. Preserved PFlash parity green (.kv slot, warm/cold, reconstruction) — ✅ verified
5. Served runtime free of `/opt/dflash` and legacy code
