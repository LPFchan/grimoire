# RSH-20260526-002: V3 Proposal — Adaptive FA Scratch Lifetime + Recoverable VMM Pool

Opened: 2026-05-26 KST
Recorded by agent: claude-opus-4-7
Status: design proposal, not implemented
Revision: 2 — pool-recoverability folded into the core scope (was an option in rev 1)

## Motivation

V2 (per RSH-20260526-001) closes V1's capture/sync abort and converts V0's `cuMemCreate → ggml_abort` into a recoverable `GGML_STATUS_ALLOC_FAILED`. But the operator wants **graphs ON by default** ("free perf, leave nothing on the table") and V2 has two costs that block that adoption:

1. **Persistent FA scratch reservation regresses tight-VRAM workloads vs V0+OFF.** V2 holds K_max/V_max bytes for the lifetime of the cuda_ctx; V0+OFF's pool lets non-FA ops reuse those bytes between FA calls. On the 210k+999+GPU 0 boundary, V0+OFF serves the workload while V2 returns HTTP 500.
2. **Non-FA pool consumers (KV_max, dst_tmp, dst_tmp_meta, others) still go through `ggml_cuda_pool_vmm::alloc` which aborts on `cuMemCreate` OOM.** V2 only protected the FA path. Any other pool consumer that hits the VMM-budget wall on tight VRAM kills the server.

V3 solves both: adaptive FA scratch lifetime *plus* recoverable VMM pool. After V3, the *only* way the daemon can be killed by VRAM pressure is a process-level hard limit (model load too large from the start). Per-request OOMs on the steady-state hot path always become recoverable.

## V3 Has Two Parts

### Part A: Adaptive FA Scratch Lifetime

Keyed on `ctx->active_capture_graph` set by `graph_compute` immediately after `cudaStreamBeginCapture` and cleared by an RAII guard at function exit:

| Path | Lifetime | Borrower behavior |
| --- | --- | --- |
| Capture active (`active_capture_graph != nullptr`) | V2 persistent (`fattn_scratch[d][s]`) | Stable pointer baked into captured kernel arguments; required for replay correctness. Step-5 invalidation + retired-list reclaim from V2 apply unchanged. |
| Direct execute (`active_capture_graph == nullptr`) — warmup, properties-changed reset, `GGML_CUDA_DISABLE_GRAPHS=1`, post-warmup decision blocks capture | Pool-borrow via `ggml_cuda_pool_alloc<half>` | Same lifetime as V0+OFF — alloc at FA entry, RAII free at FA exit. Pool can rotate bytes among non-FA ops between FA calls. |

Implementation: discriminated borrower in `fattn-common.cuh`, and graph_compute's preflight scratch reservation block moves *inside* the existing `if (use_cuda_graph && cuda_graph_update_required)` guard so it runs only on capture-bound calls.

### Part B: Recoverable VMM Pool

`ggml_cuda_pool_vmm::alloc()` currently calls `CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0))` which expands to `ggml_abort` on failure — killing the process. V3 changes the pool to expose a *recoverable* allocation path while preserving the existing aborting path for backward compatibility.

Concretely:

1. **Pool base class adds `try_alloc`:**
   ```cpp
   class ggml_cuda_pool {
   public:
       virtual void * alloc(size_t size, size_t * actual_size) = 0;          // existing — keeps aborting on failure
       virtual cudaError_t try_alloc(size_t size, void ** out, size_t * actual_size) = 0;  // new — returns cudaErrorMemoryAllocation on failure
       virtual void free(void * ptr, size_t size) = 0;
   };
   ```
2. **`ggml_cuda_pool_vmm::try_alloc` body:** same logic as `alloc` but every `CU_CHECK(cuMem*)` call is replaced with a raw `CUresult` check that returns `cudaErrorMemoryAllocation` on failure. `alloc` remains as a thin wrapper that calls `try_alloc` and aborts if the result isn't `cudaSuccess` — that preserves source compatibility for every existing caller.
3. **`ggml_cuda_pool_leg::try_alloc` body:** same pattern. `cudaMalloc` returning out-of-memory propagates as `cudaErrorMemoryAllocation` instead of aborting.
4. **New thin RAII wrapper for the FA direct-execute path:**
   ```cpp
   template<class T> class ggml_cuda_pool_try_alloc {
       cudaError_t alloc(size_t n);   // returns cudaSuccess or cudaErrorMemoryAllocation
       half * ptr;                    // nullptr on failure
       ~ggml_cuda_pool_try_alloc();   // returns to pool, identical to existing pool_alloc
   };
   ```
5. **V3's FA direct-execute borrower uses the try_alloc wrapper.** On failure, `graph_compute` returns `GGML_STATUS_ALLOC_FAILED` exactly like V2.
6. **Other pool consumers stay on the existing aborting `alloc`** for V3. Wider audit (KV_max, dst_tmp, dst_tmp_meta, every non-FA caller) is part of *full pool recoverability* and is deferred to a follow-up RSH. V3 unlocks the audit but doesn't ship it.

The net of Part B: the FA path becomes recoverable on the *pool side* too, not just on V2's own `cudaMalloc`. Combined with Part A, V3+ON's direct-execute steady state behaves identically to V0+OFF, and V3+ON's capture path behaves identically to V2.

### Why Part B Is Necessary (Not Optional)

In V3 rev 1 the pool patch was an "open question, recommend defer". Re-examining the matrix evidence: in the 210k+999+GPU 0 reproduction, V0+OFF's pool *did not* hit `cuMemCreate` failure — the workload fits there. But it fits *barely*. Any of:

- a different VBIOS reservation on a different card
- a slightly larger ctx-size
- another VRAM consumer on the host (`nvidia-smi`, X server, monitoring agent)
- a larger tool-result payload

…can shift the pool growth past the device-free-memory boundary at any of the *non-FA* pool consumers (KV_max in `launch_fattn`, dst_tmp, dst_tmp_meta). Without Part B, V3 has Part A's nice V0+OFF behavior but inherits V0+OFF's `ggml_abort` failure mode on those calls. The whole point of V3 over V2 is "graphs ON by default with no new server-killing risks" — that requires the pool itself to be recoverable on the path V3 uses.

Part B is bounded scope (~50-100 lines of pool plumbing + new RAII wrapper) and changes no existing call signatures. Existing callers continue to abort exactly as today; only V3's FA borrower opts into the recoverable path. Wider migration of other pool consumers to `try_alloc` is a separate RSH-level decision.

## Behavior Matrix (V3 vs V0, V1, V2)

At the boundary case (210k+999+GPU 0):

| | Workload fits | Server survives OOM | Graphs ON safe | Direct-exec working set ≈ V0+OFF |
| --- | --- | --- | --- | --- |
| V0+ON | ✗ (`cuMemCreate` abort) | ✗ | ✗ | n/a |
| V0+OFF | ✓ | ✗ (pool aborts on growth) | n/a | ✓ |
| V1+ON | ✗ (capture/sync abort) | ✗ | ✗ | n/a |
| V1+OFF | varies (boundary-sensitive) | ✗ | n/a | partly |
| V2+ON | ✗ (V2 cudaMalloc fail → 500) | ✓ | ✓ | ✗ |
| V2+OFF | ✗ (same) | ✓ | n/a | ✗ |
| **V3+ON** | ✓ during direct-execute; ✗ on capture if persistent reservation can't fit | ✓ everywhere | **✓** | ✓ on direct-execute |
| **V3+OFF** | ✓ | ✓ everywhere | n/a | ✓ |

V3+ON self-heal note: when capture reservation fails, V3 should mark the cgraph entry graph-disabled (set a new bit or reuse `disable_due_to_gpu_arch`) so subsequent calls stay on direct-execute permanently for that entry. That converts a recurring per-capture-attempt 500 into a single 500 followed by steady-state success. Recommend including this in V3 since it's a few lines of code on top of Part A.

## What V3 Does NOT Solve

- Workloads that genuinely don't fit (model + KV + non-FA buffers > device VRAM): V3 can't conjure memory. The first request will fail with HTTP 500, but the server stays up. Same as V2.
- Non-FA pool consumers that grow past the device limit: V3 leaves them on the aborting `alloc` path. If KV_max or dst_tmp grows past available VRAM during a future workload, the server still dies. Migrating those to `try_alloc` is a follow-up RSH (call it V4 or "full pool recoverability") and is unblocked by V3's `try_alloc` plumbing.
- Capture-time persistent reservation OOM: V3+ON inherits V2's failure here. The self-heal converts it into a graceful degradation to direct-execute.

## Implementation Effort

| Part | Files | Estimated lines | Notes |
| --- | --- | --- | --- |
| A: adaptive lifetime | `fattn-common.cuh`, `ggml-cuda.cu`, `common.cuh` | ~100 (mostly the discriminated borrower + conditional preflight) | Inherits V2's machinery; no new data structures |
| B: pool `try_alloc` | `common.cuh` (base class), `ggml-cuda.cu` (vmm + leg derived classes), small RAII wrapper | ~80-120 | Adds new method, doesn't change existing |
| C: self-heal on capture failure | `ggml-cuda.cu` in `ggml_backend_cuda_graph_compute` | ~10 | Set graph-entry disabled flag on persistent reservation failure |
| Tests / matrix | runner scripts already exist | n/a | Same 4-matrix protocol; pass criterion = V3+OFF behaves like V0+OFF and V3+ON serves on direct-execute |

Total: ~200-250 net lines on top of V2. About a day of careful implementation + a re-run of the 4-matrix.

## Validation Plan

Same 4-matrix passes as RSH-20260526-001:

1. **matrix-180k+63 on GPU 1** — apples-to-apples, draft fits, no crash trigger. Expect V3 throughput parity with V2 within run-to-run noise.
2. **matrix-170k+999 on GPU 1** — full GPU offload, tight ctx. Expect parity.
3. **matrix-999/210k on GPU 1** — draft fails to init, AR-only. Expect parity.
4. **matrix-gpu0-210k+999 on GPU 0 (V1 prod stopped)** — *the load-bearing row*. **Pass criterion: V3+OFF returns `ok=true` matching V0+OFF; V3+ON serves on direct-execute (first request `ok=true`) and gracefully degrades on persistent reservation OOM (subsequent capture attempts return HTTP 500 with the self-heal bit set, the entry permanently disables capture for that cgraph, and subsequent calls return `ok=true` on direct-execute).**

Additional V3-specific cases:

- **Forced reservation failure during capture** via `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1`: confirm self-heal disables capture for the cgraph entry and subsequent calls succeed on direct-execute.
- **Forced pool failure during direct-execute** via a new `GGML_DEBUG_FORCE_POOL_FAIL=1` (small addition to Part B): confirm `try_alloc` returns `cudaErrorMemoryAllocation`, FA borrower returns ALLOC_FAILED, server returns HTTP 500, server stays up.
- **Long-soak (≥ 1000 decode cycles)** with frequent capture / direct-execute alternation: persistent scratch from prior captures reclaimed cleanly via §3 step 5; pool's high-water mark stable; retired-list size bounded.
- **`GGML_CUDA_DISABLE_GRAPHS=1`** with V3 image: every call should pool-borrow; persistent fattn_scratch[d][s] should never be allocated; behavior identical to V0+OFF.
- **F32 K/V VEC dispatch case**: still pending from V2 validation; should work identically on V3.

## Open Decisions

- **Self-heal scope**: should capture-failure auto-disable be per-cgraph entry (proposed) or per-cuda_ctx (broader)? Per-entry is finer-grained; per-ctx is more conservative. Recommend per-entry.
- **Wider pool migration**: should V3 also migrate KV_max / dst_tmp / dst_tmp_meta to `try_alloc`? Strictly speaking that's a separate effort that V3's Part B unblocks. Recommend keep V3 narrow on this dimension; ship the migration as a follow-up after V3 lands and is stable in prod.
- **Upstream alignment**: before implementing, probe `AtomicBot-ai/atomic-llama-cpp-turboquant` and `ggml-org/llama.cpp` for in-flight work on (a) graph-safe FA scratch lifetime, (b) recoverable VMM pool, (c) any existing PR/issue tracking this exact failure mode. If upstream is already addressing this, V3 should land as a thin shim over upstream's approach rather than a parallel design. (Probe is the next step; see end of doc.)

## Decision Required Before Implementing

- Operator approve V3 scope (Part A + Part B + the self-heal in Part C).
- Operator approve schedule (~1 day implementation + matrix re-run).
- If approved, V3 supersedes RSH-20260523-001's "V2 Authoritative Plan" as the current canonical design. V2 stays in the patches/ directory as historical reference; production flips to V3.

## Upstream Probe Results (probed 2026-05-26)

Canonical survey lives in `RSH-20260526-003 §Upstream Survey`. Re-stating only the items that directly shape V3's design:

| Upstream PR / issue | What it does | Status | Relevance to V3 |
| --- | --- | --- | --- |
| `ggml-org/llama.cpp` issue **#23446** "llama-server vram usage gradually increasing each run until OOM" | The user-facing bug report. RTX 2080Ti, 3090, 5090, Vulkan, all show monotonic VRAM growth with quantized KV cache | **OPEN** since 2026-05-20, multiple confirmations | Same symptom as RSH-20260523-001 |
| PR **#23620** "cuda: optimize KV cache dequant workspace to eliminate VRAM growth in flash attention" | **2-line fix:** `K_f16.alloc(ggml_nelements(K->view_src ? K->view_src : K))` — pool reuses one max-sized allocation instead of growing-by-one allocations each step | **CLOSED, not merged**, 2026-05-24. Maintainer am17an: "Wrong fix." (no further explanation) + closed for AI-generated content policy. 5 community testers confirmed it eliminates VRAM growth | **Highly relevant — and unappreciated by upstream.** This is dramatically simpler than V2/V3. Worth testing in isolation against grimoire's workload before committing to V3. |
| PR **#22155** "ggml-cuda: flush legacy pool on OOM and retry" | Legacy pool: on cudaMalloc OOM, flush cached buffers and retry once | **MERGED** 2026-04-20 | Likely **not in Atomic's pinned SHA** (`0a635dcd`, predates the merge by ~4 days as far as I can tell). Recoverability for legacy pool only; VMM pool path still aborts. V3's Part B is broader. |
| PR **#22193** "cuda: add partial eviction on pool OOM" | Builds on #22155 — LRU eviction + `bool overallocate` flag (2x lookahead for FA buffers) | **OPEN** since 2026-04-21 | Adjacent to V3 Part B but different approach — keeps the aborting API, just retries-then-aborts. V3's `try_alloc` is cleaner because it doesn't require the caller to predict pool behavior. |
| PR **#22207** "cuda: LRU eviction + overalloc for legacy pool" | Variant of #22193 | **OPEN** | Similar |
| PR **#21054** "cuda: fall back to VEC attention when quantized K/V F16 scratch exceeds free VRAM" | Use `cudaMemGetInfo` before MMA kernel selection; pick VEC (no scratch) if MMA scratch wouldn't fit | **CLOSED, not merged** 2026-03-27 | An alternative to V2/V3 — avoid the allocation entirely. Could co-exist with V3 as a "best kernel for memory budget" heuristic. |
| PR **#22094** / **#22185** "hip: bypass memory pool for FA f16 temp buffers" | Direct alloc via raw `cudaMalloc` + `cudaStreamSynchronize` for HIP only (V1 pattern) | **CLOSED, not merged** 2026-04-18 / 2026-04-20 | The V1 patch we ship is the CUDA generalization of this. Atomic has the HIP version in its tree (`0757ff4ee fix(hip): bypass pool for FA f16 temp buffers to prevent OOM`). |
| Issue **#22032** "Flash attention crash (MUL_MAT failed / cudaStreamSynchronize) on Pascal GPUs" | Same `cudaStreamSynchronize` failure on Pascal GPUs with MiniMax-M2.7 | **OPEN** since 2026-04-17 | Possibly the same V1+ON capture/sync failure mode in a different model, on Pascal. Suggests V1's failure mode is more widespread than V0's. |

### Maintainer Position On The Underlying Bug

am17an (`ggml-org/llama.cpp` maintainer) comment on #23446:

> Using the quantized KV-cache, it is expected the VRAM usage will grow as context size grows larger because of the q8->f16 de quantization inside the FA kernel. It should also happen without any spec decoding

Upstream considers monotonic VRAM growth with quantized KV cache **by design**. Filed-and-confirmed fixes for the resulting OOMs (#23620, #21054, #22094, #22185) are getting closed without merge. Upstream's view: if your workload OOMs because of FA dequant scratch growth, your workload doesn't fit. Don't enable quantized KV cache on tight VRAM.

This is a **divergence point**. Grimoire's operator wants the workload to serve under tight VRAM with quantized KV; upstream's position is that's not a supported configuration. V2 / V3 are therefore not blocked on upstream — they cannot be upstreamed as-is — they ship as private Atomic-side patches.

### Implications For V3 Decision

1. **The 2-line view_src fix should be tested first.** It's vastly simpler than V2/V3. If it eliminates the growth on grimoire's actual workload (qwen3.6-mtp-27B, quantized KV, 180k+ ctx, agentic two-turn), it might solve the operator's problem with ~5 minutes of patch effort. **Open question:** does view_src sizing apply to MMA + VEC + WMMA + TILE dispatches uniformly, or only some of them? Worth one matrix pass at the boundary case (210k+999+GPU 0).
2. **#22155 (pool flush-and-retry, merged upstream) is probably absent from our pinned SHA.** Confirm with a `git log` against the pin; if absent, port the patch — it's ~30 lines and gives recoverability on the legacy pool path for free.
3. **V3 Part B (`try_alloc`) is novel relative to #22193's approach.** #22193 makes the existing alloc retry with LRU + flush; V3 introduces a separate `try_alloc` API. V3's is cleaner for V3's FA path use case; #22193's is a drop-in for everyone. If we decide to widely migrate non-FA pool consumers later, the #22193 approach is friendlier. If we keep recoverability scoped to FA, V3's `try_alloc` is the better surface.
4. **The capture/sync incompatibility of V1+ON has no upstream fix.** #22032 reports it on Pascal, still open. V2's design (no allocations inside capture) remains the only known approach.
5. **Operator priority alignment:** "graphs ON by default for free perf" requires V2 or V3. V0/V1 cannot deliver graphs-ON safely. If we test the 2-line view_src fix and it works, we have *V0+view_src*: simple, narrow, but still V0+ON would crash on graph capture if VRAM is tight. So V2/V3 still needed for graphs-ON safety, but the path could be: **view_src fix as the working-set optimizer + V2/V3 as the graph-safety + recoverability layer on top**. Stacked.

## Recommended Sequence Before Committing To V3

1. **Add a "V0 + view_src 2-line fix" image to the patch set.** `0003-cuda-fa-view_src-sizing.patch`. Test it at the boundary case (matrix-gpu0-210k+999). If V0+view_src+ON or V0+view_src+OFF serves the workload, the OOM is solved by the cheaper fix and V3 reduces to "graph-safety layer that doesn't bring its own VRAM regression" — much narrower scope.
2. **Port #22155 to Atomic's pinned SHA** if not already present. Cheap recoverability on legacy pool.
3. **Only then revisit V3 scope.** If view_src + #22155 + V2 (or V2+OFF) covers all the boundary cases, V3 may be unnecessary. If gaps remain (specifically: capture path under tight VRAM where view_src alone doesn't help), the V3 design as described above is the right shape.

## Pointers

- V2 boundary-case proof: `records/research/RSH-20260526-001-v2-matrix-single-gpu.md`.
- V2 patch (the artifact V3 evolves from): `patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch`.
- Matrix harness: `/mnt/MX500/grimoire-ab/{compose.gpu0.yml, compose.v2-smoke.yml, configs/, run-matrix-v2.sh, summarize-matrix.sh}`.
- Upstream PRs/issues: #23446, #23620, #22155, #22193, #22207, #22094, #22185, #21054, #22032 (see table above).
