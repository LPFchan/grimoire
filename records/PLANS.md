# Plans

## Approved Directions

### Canonical Stack Migration (Phases 1-7)
- **Outcome:** DFlash decode, compact-full persistence, and preserved PFlash all run on the canonical Bee base (`Anbeeld/beellama.cpp`); Lucebox retired
- **Why accepted:** Single canonical llama.cpp fork with both turboquant and DFlash built in; no `/opt/dflash` in served runtime; unified control plane
- **Value:** Simpler build, smaller runtime image, reduced operational surface
- **Preconditions:** All upstream repos pinned by SHA (done), Bee native binary builds (done), GPU ring + turbo4 hang fixed upstream (PR #19, merged)

### Adopt Bee as Canonical Engine
- **Outcome:** Single Bee binary serves both DFlash and non-DFlash production traffic. TheTom retired.
- **Why accepted:** TheTom's custom turboquant work is either upstreamed into Bee's base or irrelevant for CUDA Linux. Bee has DFlash built in. GPU ring + turbo4 hang fixed upstream.
- **Value:** One binary to build, deploy, and maintain. No TheTom-specific patches to carry forward.
- **Preconditions:** Bee tested with Gemma 4 + turbo4 + multimodal (done). All production flags supported (done). Docker build with web UI patches (in progress).

## Sequencing

### Near term

- [x] Phase 1: Core DFlash decode pipeline — ring buffer, Bee's dflash_draft.cpp
- [x] Phase 1.5: Binary comparison — Bee 100%, TheTom 0%. Decision: full Bee stack.
- [x] **Phase 1.6: Fix GPU ring + turbo4 hang** — merged upstream as PR #19 (commit `0ef12a5`)
  - Root cause: `ggml_backend_cuda_buffer_get_tensor` uses `cudaStreamPerThread` but ggml uses private per-context stream
  - Fix: `ggml_backend_sched_synchronize` after `process_ubatch()` when DFlash active
  - Verified: 100% acceptance, draft 6-14ms, verify 48-58ms
- [x] **Phase 2: Server integration** — Bee binary deployed as canonical engine (TheTom retired)
  - [x] 2.1 Bee binary tested with Gemma 4 + turbo4 + multimodal — all flags supported
  - [x] 2.2 GPU ring + turbo4 hang fixed upstream (PR #19)
  - [x] 2.3 Sparse V skip warp fix cherry-picked from TheTom
  - [ ] 2.4 Docker build with Bee (webui patches need porting)
  - [ ] 2.5 Gateway routing for DFlash canary
- [ ] Phase 3: compact-full Persistence Implementation (on Bee base)
  - [ ] 3.1 DFlash snapshot save/load for KV + recurrent state + `target_feat`
  - [ ] 3.2 Prompt-boundary inline snapshots, staging-slot semantics
  - [ ] 3.3 RAM-first writes, async disk mirror, TTL/budget enforcement, hash invalidation
  - [ ] 3.4 Prefix-cache compatibility
  - [ ] 3.5 Preserved llama-side PFlash path on Bee
- [ ] Verify: short-prompt decode with positive `#gen drafts` and speedup >1.5x

### Mid term

**Phase 3 — compact-full Persistence Implementation**
- Native save/load for KV + recurrent state + `target_feat` equivalents
- Prompt-boundary inline snapshots, staging-slot semantics
- RAM-first writes, async disk mirror, TTL/budget enforcement, session hash invalidation
- Prefix-cache compatibility
- Verification: `tests/test_dflash.py`, `test_e2e_smoke.py`, `test_stress_dflash.py`
- Exit: save/stop/load/continue without corruption; repeated-turn restore stable; bounded growth

**Phase 4 — Preserved PFlash Parity**
- Preserved `Qwen3.5-0.8B Q8_0` compressor, standalone `pflash_daemon`, raw compress protocol
- Warm/cold split, FIFO park/unpark, `/slots/0` save/restore parity
- Token -> text -> message reconstruction
- All required `b4ed333` + `e4d4e32` native fixes
- Verification: `test_pflash_pipeline.py`, repeated cold/warm runs, VRAM drift check
- Exit: compression works without Lucebox decode; no VRAM drift; effective-prompt contract frozen for Phase 3 sign-off

**Phase 5 — Integration and Artifact Cleanup**
- Block-aware long-prompt integration on canonical stack
- Model registry cleanup, test-harness cleanup
- `.safetensors` -> GGUF draft transition
- Text-only PFlash config cleanup
- Removal of hidden `/opt/dflash` dependencies
- Verification: full required suite + manual runtime-isolation checks
- Exit: no served path relies on `/opt/dflash`; test defaults map to real model IDs

### Deferred

**GPU ring buffer** (`cross-ring-interleave.cu`) — CPU fallback works for initial decode
**GPU tape recording** (`dflash_tape_*`) — only needed for tree-mode DDTree verify
**Multi-spec batched decode** (`common_speculative_draft_batch()`) — single-spec is sufficient for MVP

**Phase 6 — Optional Runtime Optimizations**
- VMM-based park/unpark (preferred only if isolated measurement proves gain)
- Warm-turn detection + KV slot reuse
- Selective Bee runtime helpers
- Verification: isolate each optimization, measure before/after
- Exit: every retained optimization has a measurable win

**Phase 7 — Cutover and Lucebox Retirement**
- Collapsed runtime layout, final model configs
- Rollback instructions, production checklist
- Removal of `/opt/dflash` from served runtime image
- Quarantine/removal of legacy Lucebox-specific served path
- Verification: full suite + explicit rollback test + `/opt/dflash` removal audit
- Exit: A + B + C all green on canonical stack; rollback tested; Lucebox no longer needed
- Decision gate: do not retire Lucebox until all three tracks are green. Keep split deployment if one track remains red.

## Final Gates

1. Canonical base: TheTom + buun core + selective Bee, not Bee-first
2. DFlash decode parity green for `dflash-pflash-qwen3.6-27B`
3. compact-full persistence parity green (restart resilience, staging-slot, hash invalidation, bounded growth)
4. Preserved PFlash parity green (.kv slot, warm/cold, reconstruction, native fixes)
5. `pflash-qwen3.6-27B` text-only, no multimodal
6. Served runtime free of `/opt/dflash`
7. Every upstream input and artifact pinned by SHA
8. Lucebox retirement blocked until A + B + C all green
