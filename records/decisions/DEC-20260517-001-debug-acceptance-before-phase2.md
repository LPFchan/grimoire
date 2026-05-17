# Debug Draft Acceptance Quality Before Phase 2

Opened: 2026-05-17 02-15-00 KST
Recorded by agent: opencode

## Decision

Debug the 0% draft acceptance rate before starting Phase 2 server integration work. Phase 2 items (reduced verifier, multi-slot, rollback) are moot without at least some accepted drafts.

## Context

Phase 1 completed: native DFlash pipeline generates valid draft token IDs (579, 264, 7047...) but 0% are accepted by the verifier. Bee's binary achieves 38% acceptance with the same GGUF and same target model. The pipeline structure is correct (models load, graph builds, tokens emerge) but the cross-data signal reaching the draft graph builder is scrambled — the draft predicts tokens that never match what the target would generate.

Bee's `build_cross_data()` interleaves layers as `cross_buf[layer * n_embd + t * n_target_features]`. Our `ring_write()` stores per-layer data in a circular buffer, and `build_cross_data()` reads from that ring to produce `cross_buf`. The graph builder's `set_input()` reads from `cross->v_embd` (set by `llama_set_cross_data_seq`) and expects this exact layout. Any mismatch in stride, offset, or interleave order silently scrambles the cross-attention signal.

## Options Considered

| Option | Outcome |
| --- | --- |
| Debug acceptance now (chosen) | Fixes the core data pipeline; Phase 2 then has real drafts to verify |
| Skip to Phase 2 | Would implement multi-slot/rollback on a broken foundation — drafts still get rejected 100% |
| Deploy Bee's binary as production reference | Works (38% acceptance) but lacks TheTom's turboquant features |

## Rationale

- Bee's 38% acceptance proves the GGUF and approach are sound
- The bug is in our port of Bee's ring/cross-data code, not in Bee's original
- Fixing now avoids wasted Phase 2 work that would never fire

## Suspects (ordered by likelihood)

1. **`build_cross_data()` interleave order** — the `layer * n_embd + t * n_target_features` formula in our code might differ from what Bee's graph builder expects
2. **`ring_write()` slot placement** — circular buffer offsets might misalign if `ring_write_pos` isn't advancing correctly
3. **`set_input()` reading from `cross->v_embd`** — the graph builder's CPU-side tensor-set copies data from `cross->v_embd` with a `win_offset` sliding window; if `n_real` vs `n_enc` mismatch, the wrong slice is used
4. **`dflash_draft_ctx_len()` calculation** — if `ctx_len` differs between graph reservation and input setting, the target_hidden tensor is mis-sized

## Plan

1. Add per-layer cross-data validation: dump first/last few floats from ring_buf and cross_buf, compare with Bee's
2. Compare `n_real` vs `n_enc` vs `ctx_len` at graph input time
3. If still unresolved, add a binary comparison test: run both Bee and TheTom servers on the same prompt, dump the first draft tokens and cross-data checksums
