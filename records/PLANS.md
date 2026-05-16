# Plans

## Approved Directions

### Phase 1-4: Canonical Stack Migration
- **Outcome:** DFlash decode, compact-full persistence, and preserved PFlash all run on the canonical TheTom base
- **Why accepted:** Retire Lucebox dependency, simplify build and deployment
- **Value:** Single canonical llama.cpp fork, no `/opt/dflash` in served runtime, unified control plane
- **Preconditions:** All upstream repos pinned by SHA (done), TheTom native binary builds (pending)

## Sequencing

### Near term
- [ ] Build TheTom native binary with `patches/spec-dflash-contract.patch`
- [ ] Launch native canary on isolated GPU
- [ ] Prove or disprove the native DFlash decode path on real hardware

### Mid term
- [ ] Phase 2 hardware sign-off (decode, TTFT, correctness)
- [ ] Phase 3 persistence on native path (save/load KV + recurrent state)
- [ ] Phase 4 PFlash compression parity on canonical stack

### Deferred
- Phase 5: Integration and artifact cleanup
- Phase 6: Runtime optimizations (VMM park/unpark, warm-turn reuse)
- Phase 7: Lucebox retirement and `/opt/dflash` removal from served runtime
