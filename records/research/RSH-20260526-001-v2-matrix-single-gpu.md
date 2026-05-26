# RSH-20260526-001: V2 Matrix on Single-GPU 24 GB — Adapted, Boundary-Case Reproduction, and Honest Limits of V2

Opened: 2026-05-26 KST
Recorded by agent: claude-opus-4-7
Revision: 2 (2026-05-26) — corrected after the boundary case was reproduced on GPU 0. Earlier draft overclaimed V2's throughput win and missed the persistent-reservation regression. Both are now in §Findings and §V2 Limits.

## Question

Does the V2 patch (`patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch`) fix the second-turn FA scratch crash described in RSH-20260523-001? What is V2's measured behavior compared to V0 (unpatched) and V1 (raw-cudaMalloc patch) under graphs ON and OFF — and where does V2 *not* help?

## Setup

| Knob | Value |
| --- | --- |
| Host | grimoire (2× RTX 3090 24 GB; **different VBIOS revisions per GPU — see §VBIOS Asymmetry**) |
| Model | qwen3.6-mtp-27B (Q4_K_M, 27B params, NextN draft, 16 attention layers + SSM/recurrent) |
| KV cache | `q8_0 / q8_0` |
| Repro | `tool-result-large.txt` (22,081 chars) injected on turn 2 (RSH-20260523-001 agentic loop) |
| Variants | V0 (no patch) / V1 (`0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch`) / V2 (`0002-cuda-fa-v2-scratch-owner.patch`), each ON and OFF for `GGML_CUDA_GRAPHS` |

Three matrix passes were run on this rig:

| Pass | GPU | ctx | `n-gpu-layers` | Free at startup | Purpose |
| --- | --- | --- | --- | --- | --- |
| matrix-180k | 1 (V1 prod live on 0) | 180000 | **63** (2 layers on CPU) | ~24 GiB | Apples-to-apples behavior, draft fits, no crash trigger |
| matrix-170k | 1 | 170000 | 999 | ~24 GiB | Full GPU offload, draft fits, ctx tight |
| matrix-999/210k | 1 | 210000 | 999 | ~24 GiB | Draft fails to init, AR-only |
| matrix-gpu0-210k | 0 (V1 prod stopped) | 210000 | 999 | 24,068 MiB free, 24,100 MiB total | **Boundary case — the RSH-20260523-001 crash environment** |

## VBIOS Asymmetry

Investigating "why doesn't GPU 1 reproduce the crash that GPU 0 reproduces?" surfaced this:

```
GPU 0 (08:00.0):  Total 24576 MiB,  Reserved 476 MiB,  VBIOS 94.02.26.C0.34
GPU 1 (09:00.0):  Total 24576 MiB,  Reserved 442 MiB,  VBIOS 94.02.59.00.42
```

Both cards physically have **24,576 MiB**. The firmware reserves a different sliver of VRAM for its own bookkeeping. **34 MiB asymmetry** — and that is exactly the band that determines whether the `cuMemCreate` crash fires. Operationally relevant: "identical SKU" is not "identical down to firmware". When a build straddles the VRAM boundary, the same workload reproduces on one card and not the other.

## Result Matrix (GPU 0, 210k, n-gpu-layers=999 — the crash environment)

| Variant | repro | Crash signals | Outcome |
| --- | --- | --- | --- |
| **V0 + graphs ON** | `ok=False`, HTTP 502 | `cuMemCreate=1`, `ggml_abort=1` | **CRASH** at `launch_fattn<256,8,8> → ggml_cuda_pool_vmm::alloc → cuMemCreate`. Identical signature to the original RSH-20260523-001 stack. Process dies. |
| V0 + graphs OFF | `ok=True`, 19.8 s | none | **Workload fits.** Pool's transient allocations rotate among non-FA ops between FA calls. |
| **V1 + graphs ON** | `ok=False`, HTTP 409 | `stream_cap=1`, `ggml_abort=1` | Capture/sync incompatibility (V1's `cudaStreamSynchronize` + `cudaFree` inside `launch_fattn` is illegal during graph capture). Process dies. |
| V1 + graphs OFF (current prod image) | `ok=False`, HTTP 502 | `ggml_abort=1`, no `cuMemCreate`, no `stream_cap` | Third crash signature — some other allocation failure (under investigation). Note: V1+OFF *did* succeed at this exact config on GPU 1 (boundary-sensitive). |
| **V2 + graphs ON** | `ok=False`, HTTP 500 | **none** | `V2: cudaMalloc(V f16 scratch, 2097152 bytes) failed: out of memory` → `GGML_STATUS_ALLOC_FAILED` (status -2) → `srv send_error` → 500 → server keeps serving. **Recoverable per-request failure, no `ggml_abort`.** |
| V2 + graphs OFF | `ok=False`, HTTP 500 | **none** | Same recoverable failure path as V2+ON. |

## Result Matrix (GPU 1, 180k & 170k & 999, summary)

At configs where the workload fits in available VRAM, all five non-V1+ON variants pass and throughput sits in the same 11–20 s band on the repro payload. **V1+ON deterministically crashes at every config tested** (capture/sync is config-independent). See `/mnt/MX500/grimoire-ab/results/matrix-{180k,170k,999}-...` for raw data.

## Findings

1. **The original RSH-20260523-001 crash is reproducible on GPU 0 at 210k+999, on V0+ON only.** Confirmed signature: `cuMemCreate(&handle, reserve_size, &prop, 0)` inside `ggml_cuda_pool_vmm::alloc`, called from `launch_fattn<256,8,8>` during turn-2 prompt processing at ~6 k tokens of an 8.5 k-token batch.
2. **The crash window is a 34 MiB knife-edge** between GPU 0 and GPU 1, set by the firmware-reserved VRAM diff between the two cards.
3. **V1+ON deterministically crashes everywhere** with `operation not permitted when stream is capturing`. V1 is unusable with `USE_GRAPHS=1`. Confirmed at every config and on both GPUs.
4. **V2 does NOT prevent the OOM at the boundary case** — V2 *converts* the abort into a recoverable failure (`GGML_STATUS_ALLOC_FAILED` → HTTP 500), and `cuMemCreate` no longer fires inside `ggml_cuda_pool_vmm::alloc` (the FA pool path), but V2's own `cudaMalloc` for the FA scratch fails instead. The server continues running.
5. **V0+OFF succeeds at the same boundary case that V2 fails.** This is the central honest finding: V2 is *strictly worse than V0+OFF* on raw "does the workload fit" in this tight-VRAM case.
6. **V2 is unambiguously better than V0 only on:** (a) graph-safe — supports `USE_GRAPHS=1` where V0+ON's pool growth deterministically crashes; (b) recoverable — converts process-killing aborts into per-request 500s.
7. **My earlier "5-10× faster than V1+OFF" claim is wrong and retracted.** That measurement was at `n-gpu-layers=63` where 2 CPU-offloaded layers dominated the per-token cost; the per-FA `cudaStreamSynchronize` cost in V1 is real but small compared to CPU offload. At `n-gpu-layers=999`, V2 and V1+OFF run in the same throughput band when both succeed.

## Why V2 Has a Boundary Regression

V2's design choice that pays back the regression: **scratch reservation is persistent across FA calls.** V2 reserves `max(K_i)` and `max(V_i)` over all FA nodes in the cgraph and holds those pointers stable. The borrower in `launch_fattn` reads the pointer, never allocates.

V0's pool allocator, by contrast, sees each `ggml_cuda_pool_alloc<half>` as a transient lease. After `launch_fattn` returns, the lease ends, and the pool can hand the same physical bytes to the next op (matmul, softmax, etc.). Peak VRAM occupancy is dominated by the largest *single op* with all its scratch alive at once, not the cumulative footprint.

V2 trades that flexibility for stability (which it needs because captured graphs bake the scratch pointer into kernel arguments — see §3 step 5 in RSH-20260523-001 §Patch V2 Authoritative Plan). The trade is correct *during capture*. The trade is *unnecessary* on the direct-execute path, and that's where the regression bites.

## V2 Limits — Honest List

| Aspect | V2 status |
| --- | --- |
| Original `cuMemCreate` crash | Eliminated by removing FA scratch from the VMM pool. Pool growth is smaller because the largest pool consumer is gone. |
| Capture/sync crash (V1's failure mode) | Eliminated. V2 never allocates / frees / synchronizes inside `launch_fattn`. |
| Recoverable allocation failure | Implemented per RSH §5. `GGML_STATUS_ALLOC_FAILED` propagates cleanly; server stays up. |
| **Tight-VRAM-budget workloads that V0+OFF handles** | **Regressed.** V2's persistent reservation steals VRAM from non-FA ops. V0+OFF at 210k+999 on GPU 0 fits and serves; V2 fails gracefully. |
| Throughput vs V1+OFF | **No measurable advantage** at the configs where both succeed. The earlier 5-10× claim was a CPU-offload artifact and is retracted. |

## What Would A V3 Look Like

Proposal in `RSH-20260526-002-v3-adaptive-fa-scratch.md`. Short version: **adaptive scratch lifetime** — transient pool-borrow on the direct-execute path (matches V0+OFF's working-set behavior); persistent V2-style reservation only when CUDA-graph capture is active (because that's the only path that needs a stable pointer baked into kernel arguments). Combines V0+OFF's memory efficiency with V2's graph safety. Sketched, not yet implemented.

## Artifacts

- `patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch` — V2 patch (the one tested)
- Per-pass results:
  - GPU 1 at 180k (n-gpu-layers=63): `/mnt/MX500/grimoire-ab/results/matrix-170k-...` (apples-to-apples, draft fits, no crash trigger)
  - GPU 1 at 170k (999): `/mnt/MX500/grimoire-ab/results/matrix-170k-20260526-030515/`
  - GPU 1 at 210k (999): `/mnt/MX500/grimoire-ab/results/matrix-999-20260526-020350/`
  - GPU 0 at 210k (999, V1 prod stopped) — **the boundary case**: `/mnt/MX500/grimoire-ab/results/matrix-gpu0-210k-20260526-040207/`
- Yesterday's matrix (`/mnt/MX500/grimoire-ab/results/{patched,unpatched}-graphs-{on,off}-{180k,210k}/`) — independently reproduces the same V0+ON `cuMemCreate` crash from 2026-05-24.
- Dockerfile: `GRIMOIRE_LLAMA_CPP_PATCH_FILE` build-arg added so the matrix can swap among V0 (`APPLY_PATCHES=0`), V1, V2 patches without changing Dockerfile source.
- `compose.gpu0.yml`, `compose.v2-smoke.yml`, `configs/models-matrix-{180k,210k,999,v2-smoke,gpu0}.json` — single-GPU test harness.
- `run-matrix-gpu0.sh`, `run-matrix-170k.sh`, `run-matrix-999.sh`, `run-matrix-v2.sh` — matrix runners.

## Open Items

- V1+OFF's HTTP 502 + `ggml_abort=1` (no `cuMemCreate`, no `stream_cap`) on GPU 0 at 210k+999 is a third crash signature not yet classified. Either V1 has another allocation failure mode at this boundary, or it's a downstream effect of partial allocation cleanup.
- Long-soak (≥ 1000 decode cycles) not yet run on V2.
- F32 K/V VEC dispatch case not yet exercised (qwen3.6 with q8_0 KV doesn't hit it).
- 210k matrix on dual-GPU prod layout not run (operator-time gated).
- Forced failure injection (`GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1`) hook is in the patch but not exercised.

## Verdict

V2 should be considered a **safety upgrade** over V0, not a performance upgrade or a "makes everything fit" upgrade. Adopting V2 in prod buys: graph-ON safety, recoverable per-request failure, no process aborts on FA scratch OOM. Adopting V2 *costs*: a tight-VRAM regression band where V0+OFF still fits and V2 doesn't. V3 (next RSH) addresses that cost.
