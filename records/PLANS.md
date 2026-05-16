# Plans

## Approved Directions

### Canonical Stack Migration (Phases 1-7)
- **Outcome:** DFlash decode, compact-full persistence, and preserved PFlash all run on the canonical TheTom base; Lucebox retired
- **Why accepted:** Single canonical llama.cpp fork, no `/opt/dflash` in served runtime, unified control plane
- **Value:** Simpler build, smaller runtime image, reduced operational surface
- **Preconditions:** All upstream repos pinned by SHA (done), TheTom native binary builds (pending)

## Sequencing

### Near term

- [ ] Build TheTom native binary with `patches/spec-dflash-contract.patch`
- [ ] Launch `dflash-native-qwen3.6-27B-canary` on isolated GPU
- [ ] Prove or disprove the native DFlash decode path on real hardware
- [ ] Phase 2 hardware sign-off: decode, TTFT, correctness

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
