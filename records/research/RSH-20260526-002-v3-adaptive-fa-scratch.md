# RSH-20260526-002: V3 Proposal — Adaptive FA Scratch Lifetime

Opened: 2026-05-26 KST
Recorded by agent: claude-opus-4-7
Status: design proposal, not implemented

## Motivation

V2 (per RSH-20260526-001) closes the V1 capture/sync crash and converts V0's `cuMemCreate` abort into a recoverable `GGML_STATUS_ALLOC_FAILED`. But it introduces a regression on tight-VRAM workloads: V2's persistent FA scratch reservation steals memory from non-FA ops between FA calls. At the 210k+999 boundary on GPU 0, V0+graphs-OFF *succeeds* serving the agentic repro while V2 *fails gracefully* — i.e., the user-visible outcome is "V2 lost the workload that V0+OFF could serve". V3's job is to keep V2's safety properties while restoring V0+OFF's working-set efficiency.

## Diagnosis

The trade V2 made — persistent reservation in exchange for capture-safe stable pointers — is only *needed* during CUDA graph capture, because the captured graph bakes the scratch pointer into kernel arguments. Outside capture, the pointer doesn't need to be stable: each `launch_fattn` is a single dispatch, and the scratch can be born, used, and released within that call exactly like V0's pool allocator does. By forcing the capture-safe lifetime on the direct-execute path too, V2 pays a regression it doesn't need to pay there.

## Design

**V3 = V2 + adaptive scratch lifetime, conditioned on `active_capture_graph`.**

| Path | V2 behavior | V3 behavior |
| --- | --- | --- |
| Capture active (`active_capture_graph != nullptr`) | Borrow from `fattn_scratch[d][s]` reserved up front by graph_compute preflight | **Same** — captured graphs need stable pointers; V2's preflight + retired-list + step-5 invalidation stays |
| Direct execute (`active_capture_graph == nullptr`) — warmup, properties_changed reset, `GGML_CUDA_DISABLE_GRAPHS=1`, post-warmup decision says "no capture this call" | Borrow from the same persistent slot (this is the regression source) | **Pool-borrow via `ggml_cuda_pool_alloc<half>`** — alloc at FA entry, free on FA exit (RAII), exactly like V0+OFF |

Implementation sketch (changes vs V2):

1. `launch_fattn`'s scratch borrower (in `fattn-common.cuh`) becomes a discriminated union:
   ```cpp
   struct fa_scratch_borrower {
     ggml_backend_cuda_context * ctx;
     fa_scratch_side side;
     half * ptr = nullptr;
     // ONE of these is populated, never both:
     ggml_cuda_pool_alloc<half> * transient = nullptr;
     // (persistent path uses ctx->fattn_scratch[d][s] directly; no extra storage)

     void alloc(size_t nelements) {
       if (ctx->active_capture_graph != nullptr) {
         // V2 path: persistent borrow
         ptr = side == K ? ctx->fattn_scratch[d][s].k : ctx->fattn_scratch[d][s].v;
         ctx->active_capture_graph->fa_borrowed_streams.insert(s);
       } else {
         // V0 path: pool borrow
         transient = new ggml_cuda_pool_alloc<half>(ctx->pool());
         transient->alloc(nelements);
         ptr = transient->ptr;
       }
     }
     ~fa_scratch_borrower() {
       delete transient; // returns to pool; nullptr on persistent path
     }
   };
   ```
2. `graph_compute` preflight (in `ggml-cuda.cu`) only reserves persistent scratch when the call will actually capture — move the reservation block *inside* the existing `if (use_cuda_graph && cuda_graph_update_required) { ... cudaStreamBeginCapture(...); }` block. Outside that block, fattn_scratch stays unused and zero-sized for this compute window.
3. Step-5 invalidation on growth and the retired-list lifecycle remain unchanged — they only ever fire under capture, which is exactly the path that still uses persistent scratch.
4. `§7` predictor still runs on the direct-execute path so the §3 step 3 stream-slot resolution stays accurate for the assert at the FA enqueue site — but its output is unused for the direct-execute borrower.

## What V3 Gains Over V2

| Property | V0+ON | V0+OFF | V1+ON | V1+OFF | V2+ON | V2+OFF | **V3+ON** | **V3+OFF** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original `cuMemCreate` crash absent | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | **✓** |
| Graph-ON safe (no `stream is capturing`) | n/a | n/a | ✗ | n/a | ✓ | n/a | **✓** | n/a |
| Recoverable on OOM (no `ggml_abort`) | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | **✓** | **✓** |
| Direct-execute working-set ≈ V0+OFF | n/a | ✓ | n/a | partly | ✗ | ✗ | n/a | **✓** |
| Workload fits at 210k+999 GPU-0 boundary | crashes | **fits** | crashes | crashes | fails | fails | **target: fits** | **target: fits** |

The two bottom rows are V3's reason to exist.

## What V3 Does NOT Promise

- On the **capture path** (V3+ON when graphs are being captured), V3 is equivalent to V2 in memory behavior. If the workload is so tight that even capture-time persistent scratch can't fit, V3 still returns `GGML_STATUS_ALLOC_FAILED`. V3 does not make every workload fit — it just restores V0+OFF's flexibility on the direct-execute path.
- V3 does not eliminate the §8 `fattn_compute_mu` lock cost. The lock is still needed to serialize `cuda_graphs` map mutation between target and MTP host threads regardless of which scratch path is used. This was already established as small in V2's measurements.
- V3 does not lower the pool's high-water mark below V0+OFF's — when V3's direct-execute path uses the pool, the pool can grow exactly as V0+OFF's pool does. The cuMemCreate path inside `ggml_cuda_pool_vmm::alloc` is theoretically reachable again on the direct-execute path. **However**, since V3's direct-execute path matches V0+OFF — which empirically does *not* crash at the boundary we tested — V3's direct-execute path should be equally safe. If a future workload pushes V0+OFF over the cliff, V3 will go with it. That's a known and accepted constraint.

## Open Questions Before Implementing

1. **Pool exposure cost.** Reintroducing `ggml_cuda_pool_alloc<half>` into `launch_fattn`'s direct-execute path means re-importing the failure modes V2 walked away from (specifically: `CUDA_CHECK(cuMemCreate(...))` inside `ggml_cuda_pool_vmm::alloc` still aborts). Should V3 also patch the pool to return `cudaError_t` recoverable failures? That's a wider scope (touches a hot path used by KV_max, dst_tmp, dst_tmp_meta, and every non-FA pool consumer) and probably out of scope for V3 — but worth flagging.
2. **Capture-state visibility.** The borrower checks `ctx->active_capture_graph`. That pointer is set by `graph_compute` after `cudaStreamBeginCapture` and cleared by the RAII guard at graph_compute exit. On the *first* `graph_compute` call of a session, capture hasn't started yet when scratch is first reserved (preflight runs *before* BeginCapture). The current V2 logic sidesteps this by reserving unconditionally. V3 must move the reservation *inside* the `if (use_cuda_graph && cuda_graph_update_required)` block — but launch_fattn's borrower then has to see `active_capture_graph != nullptr` reliably *only* during the captured region. The set-and-clear sequence around `cudaStreamBeginCapture` already exists in V2; V3 inherits it. Double-check that warmup-incomplete computes (the `:4237-4243` keep-`use_cuda_graph=false` path) leave the pointer null — they do in V2.
3. **What about `properties_changed` reset within an otherwise-capturing session?** When properties change, V2 sets `warmup_complete = false` and the call falls through direct-execute, then the *next* call resumes capture. V3's transient-pool borrow on the reset call is fine; the persistent slot from prior captures stays alive until §3 step 5 invalidates it on the next growth. No additional changes.
4. **Borrower allocation cost.** Each FA call on the direct-execute path will pay one `pool_alloc<half>` construct/destruct pair (i.e., one pool lease, no `cudaMalloc` in the steady-state pool warm path). V0+OFF pays this exact cost and serves the workload at this matrix's throughput band. V3+OFF should be equivalent. V3+ON during the direct-execute branches (warmup, properties_changed reset) will pay it too — these are rare, so the cost is amortized.
5. **§3 step 5 invariant.** The invariant "`(fa_borrowed_streams.size() > 0) ⇒ (instance != nullptr)`" must continue to hold. V3 only records into `fa_borrowed_streams` on the capture path — direct-execute borrower must NOT touch the set. That falls out of the conditional naturally; no code change needed beyond moving the `insert(s)` call inside the `active_capture_graph != nullptr` branch (V2 already does this; V3 inherits).

## Validation Plan for V3

Once implemented, the same 4-matrix protocol from RSH-20260526-001 should be re-run:

1. matrix-180k on GPU 1 (n-gpu-layers=63) — apples-to-apples, expect all variants behaviorally identical to V2 row.
2. matrix-170k on GPU 1 (999) — full GPU offload, expect throughput parity with V2 row.
3. matrix-999/210k on GPU 1 (999) — draft fails to init, AR-only, expect parity.
4. **matrix-gpu0-210k on GPU 0 (999, V1 prod stopped)** — boundary case. **Pass criterion: V3+OFF returns `ok=true` (matching V0+OFF), V3+ON either passes or fails gracefully like V2+ON.** This is the row where V3 has to earn its keep.

Additional V3-specific cases:

- `GGML_CUDA_DISABLE_GRAPHS=1` row: V3 should be indistinguishable from V0+OFF (no capture ever, always pool-borrow).
- Long-soak with frequent capture / direct-execute alternation (e.g., property changes triggering warmup resets): the persistent scratch from prior captures must be reclaimed cleanly via the §3 step 5 path when the slot is grown by a later capture; the retired-list size must stabilize. Same long-soak case the V2 RSH listed.
- `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` at the persistent-path allocation (preflight): should produce `GGML_STATUS_ALLOC_FAILED` just like V2.
- `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` at the transient-path allocation (pool alloc): should produce a `ggml_abort` (since the pool is still abort-on-failure). Document this as the known scope boundary unless the pool is also patched.

## Scope Decision Required Before Coding

The narrow question for the operator: **should V3 also patch the VMM pool to make it return `cudaError_t` recoverable failures?**

- **Yes:** wider blast radius (touches every pool consumer), longer implementation, but the only way to make V3 truly recoverable on the direct-execute path. V3 becomes "V2's safety + V0+OFF's footprint, end-to-end recoverable".
- **No:** narrower scope, V3 ships faster. V3+OFF would still abort on a pool growth that exceeds VRAM — same risk profile as V0+OFF today. But since V0+OFF is what currently fits the workload, the practical risk is low.

Recommendation: **No, defer pool patch.** Ship V3 with the V2 capture-path safety + V0+OFF direct-execute path. Document the pool-abort residual as a known V3 limitation. Pool recoverability is a separate effort (would unlock several other RSH-class crashes), warrants its own decision record, and shouldn't gate V3.

## Decision To Make Before Implementing V3

- **Operator approve scope** (V3 narrow, no pool patch).
- **Operator approve schedule** — V3 implementation effort: probably one day of careful coding plus a matrix re-run.
- If V3 is the path, supersede RSH-20260523-001's "V2 Authoritative Plan" with a V3 plan, mark V2 as historical.

## Pointers

- Why V2 has the regression: §V2 Limits in RSH-20260526-001.
- V2 patch (the artifact to evolve into V3): `patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch`.
- Boundary-case proof: `matrix-gpu0-210k-...` run on 2026-05-26.
