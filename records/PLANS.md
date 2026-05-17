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
  - [x] Extract pflash_daemon → `src/pflash/` (trimmed CMakeLists, removed DFlash-only sources)
  - [x] Remove `lucebox/`, `proxy/dflash.py`, `snapshot_swap.py`, `session_kv.py`, `prefix_cache.py`
  - [x] Remove `DflashDaemon` from `daemon.py`, update config.py, Dockerfile
  - [x] Remove `dflash-pflash-qwen3.6-27B` model entry, delete dflash model files (3.3 GB)
- [ ] Phase 3: Content-hash KV caching (current)
  - [x] 3.1 KVCacheStore class (RAM→disk tiering, LRU eviction, TTL, manifest)
  - [x] 3.2 Wire into proxy/llama.py (hash-based save/restore, prompt tokenization)
  - [x] 3.3 Canary model config (kv-cache-disk-dir, budget, cap)
  - [x] 3.4 Test suite rewrite: test_kv_cache_store.py, fixed e2e/stress tests
  - [ ] 3.5 Docker rebuild and deploy
  - [ ] 3.6 Verify coding-agent sysprompt caching across conversations

### Mid term

**Phase 4 — Preserved PFlash Parity**
- Warm/cold split, FIFO park/unpark, slot save/restore on `backend:llama`
- VRAM drift check, repeated cold/warm runs
- Verification: `test_pflash_pipeline.py`
- Depends on: Phase 3 sign-off

**Phase 5 — Remaining Optimization**
- Block-aware long-prompt integration
- Model registry test-harness cleanup
- Verify: short-prompt decode with positive `#gen drafts` and speedup >1.5x

### Deferred

- GPU tape recording (`dflash_tape_*`) — only needed for tree-mode DDTree verify
- Multi-spec batched decode — single-spec sufficient for MVP
- VMM-based park/unpark, warm-turn detection, KV slot reuse

## Final Gates

1. Canonical base: Bee (`Anbeeld/beellama.cpp`)
2. DFlash decode parity green for `dflash-native-qwen3.6-27B-canary`
3. Content-hash KV caching green (cross-conversation sysprompt reuse, disk mirror, restart resilience)
4. Preserved PFlash parity green (.kv slot, warm/cold, reconstruction)
5. Served runtime free of `/opt/dflash` and legacy code
