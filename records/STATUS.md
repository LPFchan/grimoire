# Current Status

**Snapshot:** 2026-05-16
**Posture:** Active migration
**Focus:** Phase 2 — canonical DFlash decode on TheTom base
**Blocker:** Fresh-state turn-1 OOM on 3090 above ~319 prompt tokens (lucebox path)

## Current State Summary

The served DFlash/PFlash stack runs on a Lucebox dflash base in production (`grimoire` container, port 9001). A parallel `grimoire-refactor` branch migrates the stack onto the canonical `TheTom/llama-cpp-turboquant` base. All five pinned upstream repos are cloned at verified SHAs under `/tmp/spec-analysis/`.

## Active Tracks

### Track A: Canonical DFlash Decode (Phase 2)
- **Goal:** DFlash speculative decode on TheTom without the Lucebox decode daemon
- **Software status:** Green. Canary control-plane validator accepts `qwen35` arch. TheTom patch set in `patches/spec-dflash-contract.patch`. Canary configured at `dflash-native-qwen3.6-27B-canary` with `Qwen3.5-0.8B-Q8_0.gguf` draft.
- **Grimoire checkpoint:** Full llama-server canary launch contract (Bee-style `--spec-type dflash`, `--spec-draft-model`, `--spec-dflash-cross-ctx`). TheTom patch recognizes DFlash flags, loads `dflash-draft` GGUF metadata, exposes `llama_model_share_tensors()`, auto-detects DFlash drafters via `LLM_ARCH_DFLASH_DRAFT`. Python-side GGUF conversion via `DFlashDraftModel`. Reduced-logit verifier consumer path wired through `common/sampling.*`, `src/llama-context.*`, `tools/server/server-context.cpp`. Minimal native DFlash draft runtime: `llama_set_dflash_capture()`, target hidden state capture at each verify layer, and draft generation on canonical TheTom.
- **Hardware status:** Short-prompt DFlash decode works on isolated GPU 1 (turn1 ~5.64s, turn2 ~0.87s). Fresh-state turn-1 fails above ~319 tokens (rollback-cache allocation ceiling). Native canary not yet tested on hardware.
- **Work:** Apply TheTom patch, build `llama-server`, run canary on isolated GPU
- **Exit criteria:** Text-only speculative decode works end-to-end, no Lucebox decode daemon needed
- **Risks:** TheTom patched server may not accept `qwen35` arch as draft; C++ runtime may hit rollback memory issues

### Track B: compact-full Persistence (Phase 3)
- **Goal:** DFlash snapshot save/load parity on canonical TheTom
- **Software status:** Green. `tests/test_dflash.py` and `tests/test_drop_in_blockers.py` cover stale session/prefix invalidation, zero-token cleanup, disk TTL/budget cleanup, model-scoped snapshot RAM paths, async mirror invalidation, stray disk-file cleanup, pending-mirror flush. Real restart-resilience gated on hardware.
- **Known hardware issue:** 2026-05-15 staging-slot leak fixed in `proxy/dflash.py`. After fix, sequential non-session requests pass. Memory pressure on session restore remains the blocker.
- **Dependencies:** Track A hardware sign-off
- **Exit criteria:** Save/stop/load/continue without semantic corruption; repeated-turn restore stable; bounded RAM/disk growth

### Track C: Preserved PFlash Path (Phase 4)
- **Goal:** Llama-side PFlash compression on canonical TheTom with full contract parity
- **Software status:** Green. `tests/test_llama_proxy.py` and `tests/test_drop_in_blockers.py` cover exception-safe park/unpark, validated slot naming, restore/save parity, shim-side FIFO control. Per-model shim FIFO bases, model-scoped `.kv` slot filenames, per-model serialization to avoid slot collisions.
- **Hardware progress:** 2026-05-16 PFlash compression baseline: 13-message / ~150K chars prompt compressed 43,584→23,296 (1.87x) at ~21s wall-clock with VRAM steady at ~20.4 GiB.
- **Dependencies:** Track A hardware sign-off
- **Exit criteria:** Long-prompt compression works without Lucebox decode; no VRAM drift; warm/cold reproduced; reconstruction preserves message semantics

## Phase 1 Status: Canonical Base Bring-Up

- **Software isolation:** `tests/test_drop_in_blockers.py` covers normal llama startup env without `/opt/dflash` in `LD_LIBRARY_PATH`, park-model shim scoping via `LD_PRELOAD`, fail-closed prerequisite validation.
- **Outstanding:** The runtime image still carries `/opt/dflash` for preserved PFlash components and the legacy DFlash daemon. Final removal is a Phase 7 gate, not a Phase 1 outcome.

## Recent Changes

- 2026-05-16: Adopted LPFchan/repo-template — records/, skills/, commit standards
- 2026-05-16: All 5 pinned repos cloned at `/tmp/spec-analysis/` at verified SHAs
- 2026-05-16: Canary draft switched from `Qwen3-0.6B-BF16.gguf` to `Qwen3.5-0.8B-Q8_0.gguf`; validator accepts `qwen35` arch
- 2026-05-16: Rollback-cache migration fix (`F16`→`Q8_0` in `migrate_prefill_cache`)
- 2026-05-15: Isolated GPU 1 validation container proved short-prompt DFlash decode works
- 2026-05-15: Snapshot staging-slot leak fixed in `src/grimoire/proxy/dflash.py`
- 2026-05-15: Native GGUF draft validator tightened (streaming parse, block_size=16 / n_target_layers=5 enforcement)
- 2026-05-15: Harness updates: `/history` conversation creation, `GRIMOIRE_LONG_PROMPT_MIN_CHARS`, `GRIMOIRE_SMOKE_CONTAINER`, `GPU_INDEX`

## Active Blockers

1. **Fresh-state memory wall (lucebox path):** Served DFlash OOMs on 3090 above ~319 prompt tokens on fresh-state turn-1, even after Q8_0 rollback fix. Next failure band at ~385 tokens. Hardware-gating Phase 2/3 sign-off.
2. **Native canary untested:** The TheTom native binary is now built at `/home/yeowool/grimoire-refactor/tmp/thetom-bin/bin/llama-server` (2026-05-16). DFlash flags confirmed (`--spec-type dflash`, `--spec-draft-model`, `--spec-dflash-cross-ctx`). Ready for GPU launch testing.

## Remaining Baseline Gaps

- No trustworthy cold moderate/long-prompt restore baseline (served DFlash sessions hit GPU memory failures).
- No final stable five-run median set (intermittent daemon/OOM failures even on short runs).

## Active Blocker

1. **Native canary produces garbage draft tokens** (confirmed: `dflash draft decode failed: -1` → `invalid token[1] = -1079354870`). Root cause: our simplified DFlash patch omitted Bee's `prepare_batch_draft()` and ring buffer. Plan: port Bee's pipeline (~2000 lines, estimated 14 days total over 3 phases).

## Immediate Next Steps

1. ✅ Phase 1 core pipeline — ring buffer, Bee's dflash_draft.cpp, all bugs fixed
2. ✅ Binary comparison test — Bee gets 100% acceptance (11/11), we get 0%
3. 🔴 Root cause: missing `flush_prefill()` — ring only has 4 tokens vs Bee's 12
4. Implement `flush_prefill()` and `prepare_batch_draft()` from Bee
5. Then: Phase 2 server integration (reduced verifier, multi-slot, rollback)
