# Rollback-Cache F16 to Q8_0 Fix

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Question

Why did fresh-state DFlash turn-1 fail at ~256 prompt tokens with `cache migration: ggml_backend_alloc_ctx_tensors failed for rollback cache`?

## Investigation

Source inspection of `lucebox/dflash/src/qwen35_target_graph.cpp` revealed that `migrate_prefill_cache(...)` allocated rollback `ssm_intermediate` tensors as `GGML_TYPE_F16`, while the full-cache creation path already used `GGML_TYPE_Q8_0`. The F16 allocation was inflating rollback intermediates relative to a cold full-cache alloc, pushing past the 3090's 24 GiB VRAM budget.

## Fix

Changed the migration path to use `Q8_0` (commit `c08fcf6`). The fix was minimal — one tensor type changed in `migrate_prefill_cache()`:

```cpp
// Before:
ggml_tensor * Si = ggml_new_tensor_4d(cache.rollback_ctx, GGML_TYPE_F16, ...);
// After:
ggml_tensor * Si = ggml_new_tensor_4d(cache.rollback_ctx, GGML_TYPE_Q8_0, ...);
```

## Result

After rebuilding the native dflash daemon in a CUDA-capable build container and bind-mounting the patched `/opt/dflash` into a disposable fresh-state container, previously failing prompts passed:

| Prompt tokens | Before fix | After fix |
| --- | --- | --- |
| 256 | OOM | OK (TTFT ~6411ms, decode ~70 tok/s) |
| 289 | OOM | OK (TTFT ~6070ms, decode ~69 tok/s) |

The fresh-state failure threshold moved from ~193-256 tokens to ~319-385 tokens.

## Rejected Paths

- Reducing `budget` below 18 — would change served contract.
- Decreasing `ctx-size` — would change served contract.
- A larger rollback allocation redesign — the Q8_0 change matches the full-cache path exactly; anything more is premature optimization.

## Open Questions

- The next failure band at ~319-385 tokens is still present. Further memory optimization in the verify/rollback path may be needed.
- The native TheTom path may not have this issue since it uses a different code path entirely.
