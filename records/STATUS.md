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
- **Status:** Software verification green. Canary control-plane validator unblocked (now accepts `qwen35` arch). TheTom patch set is in `patches/spec-dflash-contract.patch`. The native canary is configured at `dflash-native-qwen3.6-27B-canary` but has not been tested on hardware.
- **Work:** Apply TheTom patch, build `llama-server`, run canary on isolated GPU
- **Exit criteria:** Text-only speculative decode works end-to-end, no Lucebox decode daemon needed
- **Dependencies:** TheTom native binary build
- **Risks:** TheTom patched server may not accept `qwen35` arch as a speculative draft; C++ runtime may fail on rollback memory

### Track B: compact-full Persistence (Phase 3)
- **Status:** Grimoire-side software hardening complete. Hardware-gated.
- **Dependencies:** Track A hardware sign-off

### Track C: Preserved PFlash Path (Phase 4)
- **Status:** Grimoire-side software hardening complete. Hardware-gated.
- **Dependencies:** Track A hardware sign-off

## Recent Changes

- 2026-05-16: All 5 pinned repos cloned at `/tmp/spec-analysis/` at verified SHAs
- 2026-05-16: Canary draft switched from `Qwen3-0.6B-BF16.gguf` to `Qwen3.5-0.8B-Q8_0.gguf`; validator accepts `qwen35` arch
- 2026-05-16: Rollback-cache migration fix (`F16`→`Q8_0` in `migrate_prefill_cache`)
- 2026-05-15: Isolated GPU 1 validation container proved short-prompt DFlash decode works
- 2026-05-15: Snapshot staging-slot leak fixed in `src/grimoire/proxy/dflash.py`

## Active Blockers

1. **Fresh-state memory wall (lucebox path):** Served DFlash OOMs on 3090 above ~319 prompt tokens on fresh-state turn-1, even after the Q8_0 rollback fix. Next failure band at ~385 tokens. Hardware-gating Phase 2/3 sign-off.
2. **Native canary untested:** The TheTom native binary has never been built and tested with the canary config on GPU.

## Immediate Next Steps

1. Build TheTom native binary with the DFlash patch
2. Launch `dflash-native-qwen3.6-27B-canary` on isolated GPU
3. If native path works, begin hardware verification of the full verification suite
4. If native path hits memory wall, investigate TheTom's rollback/verify allocation vs lucebox path
