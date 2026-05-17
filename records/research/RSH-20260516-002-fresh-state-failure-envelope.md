# Fresh-State DFlash Failure Envelope on RTX 3090

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Question

What is the actual hardware envelope for fresh-state (cold, no snapshot) DFlash turn-1 on the current 3090 configuration?

## Method

Disposable-container sweep on isolated GPU 1 (port 9003). One brand-new container + brand-new state volume per candidate. Each candidate received a single synthetic single-message prompt at a specific token count. No prior session state existed.

## Findings

### Pre-fix envelope (without Q8_0 rollback patch)

| Token count | Result |
| --- | --- |
| 64 | Pass |
| 127 | Pass |
| 193 | Pass |
| 256 | Fail: `cache migration: ggml_backend_alloc_ctx_tensors failed for rollback cache` |
| 289 | Fail: same error |

Failure band: ~193-256 prompt tokens.

### Post-fix envelope (with Q8_0 rollback patch, `budget=18`)

| Token count | Result |
| --- | --- |
| 64 | Pass |
| 127 | Pass |
| 193 | Pass |
| 256 | Pass (TTFT ~6411ms, prefill ~0.87s, migrate ~122ms, decode ~70 tok/s) |
| 289 | Pass (TTFT ~6070ms, prefill ~1.00s, migrate ~119ms, decode ~69 tok/s) |
| 319 | Pass (TTFT ~6191ms, prefill ~1.06s, migrate ~157ms, decode ~70 tok/s) |
| 385 | Fail: zero-token client error after prefill (~1.30s) and migrate (~125ms) |
| 448 | Fail: `cache migration: ggml_backend_alloc_ctx_tensors failed for rollback cache` |
| 511 | Fail: same error |

Failure band: ~319-385 prompt tokens.

### Session restore behavior

A hand-written real session (16 prompt tokens turn 1, 49 prompt tokens turn 2) passed cleanly on a fresh container: TTFT ~5315ms then ~981ms. The restore path itself is functional for short prompts.

A six-turn capped session reached three successful turns before failing on turn 4 with `model produced zero tokens`. Observed snapshot growth:
- Turn 1 (23 tokens): tmpfs ~351 MiB, disk ~968 MiB
- Turn 2 (47 tokens): tmpfs ~352 MiB, disk ~933 MiB  
- Turn 3 (71 tokens): tmpfs 0 MiB, disk ~703 MiB
- Turn 4: zero-token failure

Multiple sequential non-session requests (after the staging-slot leak fix) pass cleanly, showing the failure is specific to session restore + memory pressure.

## Conclusion

The current 3090 configuration with `budget=18` has a hard fresh-state turn-1 ceiling at approximately 319 prompt tokens on single-message prompts. Rollback/verify cache allocation is the bottleneck. Session restore adds additional memory pressure that can cause earlier failures.

## Next Steps

- Investigate further memory optimization in the verify/rollback allocation path.
- Test the native TheTom path which uses a different code path (no `gguf_draft_loader.cpp`, no lucebox daemon, no rollback cache migration).
- Consider whether a staged approach (shorter ctx, lower budget for fresh-state, then budget scaling after restore) is acceptable.
