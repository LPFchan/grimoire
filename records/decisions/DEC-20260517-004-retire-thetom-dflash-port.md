# Retire TheTom DFlash Port — Full Bee Stack Adopted

Opened: 2026-05-17 20-48-30 KST
Recorded by agent: opencode

## Decision

Retire the TheTom DFlash port as the served runtime. Bee's `llama-server` binary (`Anbeeld/beellama.cpp`) is the single runtime for both the native DFlash canary and the baseline production stack. TheTom remains the canonical base for normal (non-DFlash) llama.cpp serving and turboquant KV cache support.

## Context

The decision to port Bee's DFlash pipeline onto TheTom (DEC-20260516-005) was motivated by a desire to keep a single canonical fork. After extensive debugging:
- TheTom's port produced 0% draft acceptance (garbage tokens from uninitialized memory)
- Bee's implementation achieves 100% draft acceptance with Buun's GGUF
- The GPU ring + turbo4 hang was the last blocker for full Bee adoption — now fixed upstream (PR #19, commit `0ef12a5`)

Keeping TheTom's DFlash port alive would require maintaining ~2000 lines of speculative decoding code with no working path to production. The cost of maintaining dual DFlash codebases exceeds the benefit of "one canonical fork."

## What "Retired" Means

- TheTom DFlash patch (`spec-dflash-contract.patch`) is frozen — no further development
- TheTom's turboquant base (feature-turboquant-kv-cache) remains the canonical baseline for non-DFlash production serving
- Bee's binary is the served runtime for DFlash speculative decoding
- The `tmp/spec-analysis/thetom-shallow/` clone is preserved for reference but not actively maintained

## What This Unlocks

- Phase 2 (server integration) targets Bee's `llama-server` directly
- No need to port reduced verifier, multi-slot, or rollback to TheTom — Bee already ships them
- DFlash `compact-full` persistence work can target Bee's snapshot format directly

## Related Decision Records

- DEC-20260516-005: Port Bee's DFlash pipeline (superseded by this decision for the served runtime)
- DEC-20260517-002: Use Bee binary for canary (confirmed by this decision)
- DEC-20260517-003: Full Bee stack with GPU ring debug (blocker resolved, decision validated)
