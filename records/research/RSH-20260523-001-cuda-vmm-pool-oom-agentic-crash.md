# RSH-20260523-001: CUDA VMM Pool OOM Crash During Agentic MCP Tool Result Injection

Opened: 2026-05-23 02-00-00 KST
Recorded by agent: deepseek-v4-pro-precision

## Question

When the webui's agentic loop injects a large MCP tool result (~10k+ chars of real text) into the LLM context for the second turn, why does the server crash with "Error in input stream"?

## Evidence

### Reproduction

1. Send a user query that causes the model to call `web_search_exa` (exa MCP)
2. First turn completes successfully: model generates tool call
3. MCP returns a lengthy result (real text, >10k chars, not 'x' repetition)
4. Second turn: inject tool result into context and stream → **502 Model server unavailable**
5. Server log shows crash in CUDA VMM pool during flash attention

### Crash Stack Trace

```
ggml_abort()  ← server terminates with SIGABRT
  → CU_CHECK(cuMemCreate(...)) → CUDA_ERROR_OUT_OF_MEMORY
    → ggml_cuda_pool_vmm::alloc()  [ggml-cuda.cu:505]
      → launch_fattn<256,8,8>()     [flash attention temp buffer alloc]
        → ggml_cuda_flash_attn_ext_mma_f16_case<256,256,8,8>()
          → ggml_backend_sched_graph_compute_async()
            → llama_context::process_ubatch()
              → llama_context::decode()
                → llama-server request handler
```

### Failure Mode

The VMM pool's `alloc()` method (line 485-548 of `ggml-cuda.cu`) manages virtual memory via CUDA VMM (`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`). When the pool needs more physical backing:

```cpp
if (size > avail) {
    size_t reserve_size = size - avail;
    reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);
    GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);  // 32 GB limit

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));  // ← OOM here
    ...
}
```

`cuMemCreate` fails with `CUDA_ERROR_OUT_OF_MEMORY` when the GPU has insufficient free VRAM. The `CU_CHECK` macro expands to `ggml_abort()` on any error — no retry, no OOM fallback, no graceful degradation.

### Why Tool Results Trigger It

The flash attention kernel allocates f16 dequantization temp buffers (`K_f16`, `V_f16`) that scale with KV cache length. With quantized KV cache (`cache-type-k=q8_0`), these temp buffers are allocated FROM the VMM pool during prompt processing. When the tool result (~10k+ chars ≈ 2500+ tokens) is injected into context, the total prompt increases, requiring larger FA temp buffers that exceed the remaining VRAM budget.

### Why 'x'-Repetition Doesn't Trigger It

Tested with `'x' * 15000` (same char count) — works fine. Real text tokenizes into more tokens than repeated 'x' (which collapses into few repeating BPE tokens), meaning real text produces a much longer effective prompt and larger FA temp buffers.

## Upstream Context

### This is an Upstream Bug, Not Atomic-Fork-Specific

| Item | Link | Status |
|------|------|--------|
| Exact same crash with MTP | [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) (May 16) | Open |
| Same crash with different GPUs | [#23245](https://github.com/ggml-org/llama.cpp/issues/23245) (May 18) | Open |
| Legacy pool OOM during prefill | [#22075](https://github.com/ggml-org/llama.cpp/issues/22075) (Apr 18) | Closed |
| VRAM leak in legacy pool + FA | [#22107](https://github.com/ggml-org/llama.cpp/issues/22107) (Apr 19) | Closed as duplicate |

Downstream: grimoire uses `AtomicBot-ai/atomic-llama-cpp-turboquant` at SHA `0a635dcd92ba66c75fccfef91c3e106f4668f367` (tag `turboquant-kv-cache-b9019`, based on upstream before any OOM fixes).

### Existing Fixes

| PR | Approach | Scope | Status |
|----|----------|-------|--------|
| [#22155](https://github.com/ggml-org/llama.cpp/pull/22155) | Flush legacy pool on OOM & retry | Legacy pool only | ✅ Merged |
| [#22094](https://github.com/ggml-org/llama.cpp/pull/22094) | Bypass pool for FA f16 temp buffers via RAII | HIP only | ❌ Open |
| [#22207](https://github.com/ggml-org/llama.cpp/pull/22207) | LRU eviction + overalloc for legacy pool | Legacy pool only | ❌ Open |

**None fix the VMM pool path** — that's why #23154 and #23245 still crash on latest upstream HEAD (`1acee6b`).

## Peer Review

Reviewed by: subagent (deepseek-v4-pro-precision)
Review timestamp: 2026-05-23 02-10-00 KST

### Accuracy

All factual claims verified against source code (`ggml-cuda.cu`, `fattn-common.cuh`, `common.cuh`). Upstream issue citations are correct. One nuance: #23154 shows the crash occurs specifically **with MTP enabled**, which is the grimoire config — the MTP speculator creates additional FA calls that accelerate VMM pool OOM.

### Option A Fix: Unsafe as Written

**Problem:** The proposed `cuMemUnmap`/`cuMemAddressFree` without prior `cudaDeviceSynchronize` will cause GPU page faults. The VMM pool may contain allocations still referenced by in-flight GPU work. The RSH's safety claim ("GPU has not yet submitted work") is incorrect — previous FA calls in the same graph computation chain may have queued work referencing pool allocations.

**Required correction:**
```cpp
// Corrected Option A — always sync before unmap
CUresult result = cuMemCreate(&handle, reserve_size, &prop, 0);
if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    CUDA_CHECK(cudaDeviceSynchronize());  // MUST sync before unmapping
    if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
        for (auto & m : mappings) {
            CU_CHECK(cuMemUnmap(m.first, m.second));
        }
        mappings.clear();
#else
        CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
        CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
    }
    pool_addr = 0;
    pool_used = 0;
    pool_size = 0;
    CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
    result = cuMemCreate(&handle, reserve_size, &prop, 0);
}
CU_CHECK(result);
```

Mirrors the `cudaDeviceSynchronize` + `clear_pool()` pattern from the merged #22155 (legacy pool fix).

### Option B Fix: Wrong API Call

The RSH uses `cuMemFree()` which frees allocation handles (`CUmemGenericAllocationHandle`), not device pointers. Should use `cudaFree` (runtime API) like the HIP pattern.

**Corrected:**
```cpp
struct ggml_cuda_fa_temp_buf {
    half *ptr = nullptr;
    cudaStream_t stream;
    ggml_cuda_fa_temp_buf(cudaStream_t s) : stream(s) {}
    ~ggml_cuda_fa_temp_buf() {
        if (ptr) {
            cudaStreamSynchronize(stream);
            cudaFree(ptr);
        }
    }
    void alloc(size_t nelements) {
        CUDA_CHECK(cudaMalloc(&ptr, nelements * sizeof(half)));
    }
};
```

### `CUDA_POOL_VMM_MAX_SIZE` Assessment

The 32 GB limit is **fine** for dual-RTX-3090 on Linux:
- VMM pool is per-device. Each 24 GB GPU gets its own 32 GB virtual address reservation.
- 32 GB VA is trivially small on 64-bit Linux (47-bit user VA space).
- The assertion at line 497 is a correctness guard, not a runtime OOM trigger under normal operation.

### Additional Upstream Issues Found

| Issue | Relevance |
|-------|-----------|
| [#22583](https://github.com/ggml-org/llama.cpp/issues/22583) | System RAM exhaustion from VRAM overcommit — proposes 256 MB defensive VRAM margin before OOM |
| [#23181](https://github.com/ggml-org/llama.cpp/issues/23181) | `cudaGraphInstantiate` OOM from context checkpoints — different root cause, same symptom |
| [#580](https://github.com/ggml-org/llama.cpp/issues/580) (node-llama-cpp) | `cuMemAddressReserve` failure on Windows WDDM — Windows-specific, not applicable |

### Additional Approach: VMM → Legacy Pool Fallback

If `cuMemCreate` fails with OOM, destroy the VMM pool and fall back to `ggml_cuda_pool_leg` for the remainder of the process lifetime. The legacy pool already has its own OOM retry mechanism (#22155). This avoids the complexity of VMM cleanup during active computation, but requires propagating the error past `ggml_cuda_pool_alloc` to the pool owner (`ggml_backend_cuda_context`), making it architecturally more invasive.

### Recommendations

| Priority | Action | Rationale |
|----------|--------|-----------|
| P0 | Fix Option A: add `cudaDeviceSynchronize` before unmap | Without sync, Option A causes GPU page faults |
| P0 | File upstream PR/issue | Bug affects all VMM-path CUDA users (#23154, #23245) |
| P1 | Test Option B on grimoire stack | RAII bypass proven on HIP; quick to implement, zero risk |
| P1 | Consider Option C (both A + B) | Option A as safety net, Option B prevents FA temps from entering pool |
| P2 | Verify with conversation UUIDs from RSH | Validate fix resolves the specific agentic crash cases |

## Proposed Fix (Revised)

### Option A (Corrected): OOM Retry with Synchronization

Mirror the #22155 pattern: sync first, free existing VMM pages, retry once.

```cpp
// In ggml_cuda_pool_vmm::alloc(), replace:
//   CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));
// With:
CUresult result = cuMemCreate(&handle, reserve_size, &prop, 0);
if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    CUDA_CHECK(cudaDeviceSynchronize());
    if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
        for (auto & m : mappings) {
            CU_CHECK(cuMemUnmap(m.first, m.second));
        }
        mappings.clear();
#else
        CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
        CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
    }
    pool_addr = 0;
    pool_used = 0;
    pool_size = 0;
    CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
    result = cuMemCreate(&handle, reserve_size, &prop, 0);
}
CU_CHECK(result);
```

### Option B (Corrected): RAII Bypass for FA Temp Buffers (Recommended First Fix)

Port the existing HIP pattern (#22094) to CUDA/VMM. Smallest, safest, most localized change.

Target file: `ggml/src/ggml-cuda/fattn-common.cuh`

```cpp
// Remove the #ifdef GGML_USE_HIP guard from the RAII struct
struct ggml_cuda_fa_temp_buf {
    half *ptr = nullptr;
    cudaStream_t stream;
    ggml_cuda_fa_temp_buf(cudaStream_t s) : stream(s) {}
    ~ggml_cuda_fa_temp_buf() {
        if (ptr) {
            cudaStreamSynchronize(stream);
            cudaFree(ptr);
        }
    }
    void alloc(size_t nelements) {
        CUDA_CHECK(cudaMalloc(&ptr, nelements * sizeof(half)));
    }
};
```

Then replace the pool-based `K_f16`/`V_f16` allocations (currently lines ~1322-1324 of `fattn-common.cuh`) with the RAII wrapper, just like the existing HIP path does.

### Option C: Combine Both

Apply Option A as a safety net for any VMM pool OOM across all allocation types, and Option B as a targeted fix that prevents FA temp buffers from entering the pool at all. The HIP benchmarks from #22094 show combining both is optimal.

## Open Questions

1. Should this be filed as an upstream PR against llama.cpp, or is an atomic-fork-local patch sufficient?
2. Does the MTP speculator's additional FA calls significantly accelerate the OOM? (Per #23154, yes — worth investigating for grimoire's config)

## Follow-up Routes

- [ ] File upstream issue or PR referencing this RSH and related issues (#23154, #23245, #22075)
- [ ] Apply Option B (RAII bypass) to atomic fork's `fattn-common.cuh` — safest first fix
- [ ] Test Option B on grimoire stack with real tool results from `e2d2b083`, `33bcee6a`
- [ ] Apply Option A (OOM retry) as deeper safety net if Option B alone is insufficient
- [ ] Verify fix with exact conversation messages from `e2d2b083-dad7-4e29-a38c-9a4e9e1736fa` and `33bcee6a-0062-4438-8ffc-8a9eebf5375e`

## Verification Update — 2026-05-23 02-32-32 KST

Verified by agent: openai/gpt-5.5

### Result

The core diagnosis holds.

Live grimoire logs already contain multiple matching failures, including `33bcee6a-0062-4438-8ffc-8a9eebf5375e` with `mcp.exa.ai` tool activity. The relevant failure path is:

```text
CUDA error: out of memory
current device: 0, in function alloc at ggml-cuda.cu:505
cuMemCreate(&handle, reserve_size, &prop, 0)
ggml_cuda_pool_vmm::alloc -> launch_fattn -> ggml_cuda_flash_attn_ext_mma_f16_case -> llama_decode
```

The HTTP-side symptom also matches: the llama-server peer closes the response stream, producing incomplete chunked read / disconnected response errors and downstream `502 Bad Gateway` responses.

### Evidence Rechecked

| Claim | Verification |
| --- | --- |
| Deployed server is the pinned Atomic fork | Container `llama-server --version` reports `version: 1 (0a635dc)` |
| Runtime uses CUDA VMM | Startup reports both RTX 3090 devices with `VMM: yes` |
| Current model shape is risky | `qwen3.6-mtp-27B` uses NextN/MTP, `ctx-size: 210000`, `cache-type-k: q8_0`, `cache-type-v: q8_0` |
| Crash happens in FA temp allocation path | Live logs show `ggml_cuda_pool_vmm::alloc -> launch_fattn<256,8,8> -> flash_attn_ext_mma_f16_case` |
| Real tool text is materially larger than repeated `x` after tokenization | `/tmp/tool-result-large.txt`: 21,692 chars, 6,204 Qwen tokens; same-length `x` string: 2,712 tokens |
| Atomic pinned VMM path lacks retry/fallback | `0a635dc` still calls `CU_CHECK(cuMemCreate(...))` directly |
| Current upstream VMM path lacks retry/fallback | `ggml-org/llama.cpp` master `1acee6bf8939948f9bcbf4b14034e4b475f06069` still calls `CU_CHECK(cuMemCreate(...))` directly |
| Current upstream FA f16 temp buffers still enter the pool on CUDA | Current upstream `fattn-common.cuh` still uses `ggml_cuda_pool_alloc<half> K_f16/V_f16` |

The direct repro script was not rerun during this verification because the live logs already prove the failure path, and rerunning it would likely intentionally crash or disrupt the currently healthy model server.

### Upstream Status Correction

As of this verification:

| Item | Current state |
| --- | --- |
| `ggml-org/llama.cpp#23154` | Open |
| `ggml-org/llama.cpp#23245` | Open |
| `ggml-org/llama.cpp#22155` | Merged; legacy pool only |
| `ggml-org/llama.cpp#22094` | Closed, not open |
| `ggml-org/llama.cpp#22207` | Open; legacy pool only |

### PR Posture

This is worth fixing locally in the Atomic/grimoire stack because it is a production server abort under realistic agentic tool-result traffic.

This is also worth taking upstream only after a local patch is proven with before/after evidence. The preferred PR shape is the targeted FA temp-buffer fix: keep CUDA FA f16 dequant temp buffers (`K_f16`, `V_f16`) out of the VMM pool by using scoped `cudaMalloc` / `cudaFree`, analogous to the HIP-side approach. The upstream pitch should be narrow: MTP/NextN + quantized KV + long real-text prompt/tool-result traffic can abort CUDA VMM users via pooled FA temp allocations.

Do not submit Option A as the first upstream PR in its current form. Retrying VMM allocation by unmapping/freeing the whole VMM pool is risky if any allocation from that pool is still logically live. `cudaDeviceSynchronize()` handles in-flight kernels, but it does not make active pointers safe to invalidate. Option A may be acceptable only as a carefully designed allocator-level safety net after proving lifetime constraints, not as a quick patch.

Recommended sequence:

1. Patch Atomic locally with the targeted FA temp-buffer bypass.
2. Reproduce the current crash case before the patch if a controlled maintenance window is available.
3. Verify after the patch with the exact `33bcee6a...` / `e2d2b083...` message shapes or equivalent real tool-result payloads.
4. Measure VRAM and throughput before/after on the same model/config.
5. Comment on upstream issues `#23154` and `#23245` with the evidence.
6. Open a focused upstream PR only if the local fix resolves the crash without meaningful throughput regression.

## Mitigation Verification Update - 2026-05-23 10-21-28 KST

Verified by agent: openai/gpt-5.5

### Local Patch Applied

The grimoire image now applies a local Atomic patch during Docker build:

- Source: `AtomicBot-ai/atomic-llama-cpp-turboquant` pinned at `0a635dcd92ba66c75fccfef91c3e106f4668f367`
- Patch: `patches/atomic-llama-cpp/0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch`
- Build guard: patch SHA is tracked in `/app/.cache/llama-cpp-build/.atomic_patch_hash`; a changed patch forces rebuild
- CUDA graphs: disabled with `-DGGML_CUDA_GRAPHS=OFF` so scoped raw `cudaMalloc` / `cudaFree` is not captured inside CUDA graphs

The patch keeps FA f16 dequant temp buffers (`K_f16`, `V_f16`) out of the ggml CUDA pool and allocates them directly for the duration of `launch_fattn()`.

### Build Result

Docker build command:

```text
nohup /usr/bin/docker compose build grimoire > /tmp/grimoire-vmm-fix-build.log 2>&1 &
```

Result:

- Image built successfully as `grimoire:local`
- Image ID: `sha256:e9d79213dc0a903d438f8af525b92023d183051f037baab83897558c16064bc4`
- Binary check passed under NVIDIA runtime
- `llama-server --version` reported `version: 1 (0a635dc)`
- Both RTX 3090 devices were detected with `VMM: yes`

### Runtime Result

The live `grimoire` service was recreated from the patched `grimoire:local` image and became healthy on port `9001`.

At the original `qwen3.6-mtp-27B` `ctx-size: 210000`, the original VMM-pool `cuMemCreate` abort was no longer observed. The failure moved to direct FA temp-buffer `cudaMalloc`, showing the targeted patch removed the VMM retention failure but the 210k all-GPU configuration still does not have enough RTX 3090 VRAM headroom for the repro shape.

The active mitigation is therefore:

- keep `n-gpu-layers: 999`
- reduce `qwen3.6-mtp-27B` `ctx-size` from `210000` to `180000`
- keep `cache-type-k: q8_0` and `cache-type-v: q8_0`

With `ctx-size: 180000`, the long-timeout real tool-result repro succeeded:

```text
Output: 60660 bytes, 241 SSE events
Last: data: [DONE]
Content: True, Reasoning: True, Finish: True, Error: False
```

Evidence log: `/tmp/grimoire-vmm-fix-repro-180k-long.log`.

### Conclusion

The local mitigation is effective for the tested production shape when paired with the 180k context limit. It should not be described as making the original 210k all-GPU shape safe on 24 GB GPUs. The corrected claim is narrower: bypassing the VMM pool prevents the crash from being a retained-pooled-FA-temp-buffer abort, and the remaining requirement is sufficient raw VRAM headroom for the selected context size.

## Backtracking Matrix Update - 2026-05-24

Verified by agent: openai/gpt-5.5

### Test Shape

The original `/tmp/tool-result-large.txt` was lost across reboot, so the backtracking matrix used a replacement real-text Sony alpha 7 article payload stored at `/mnt/CT500MX500SSD1/grimoire-ab/tool-result-large.txt`.

Payload size: 22,081 characters as observed by the repro harness.

Harness and artifacts:

- Runner: `/mnt/CT500MX500SSD1/grimoire-ab/run-matrix.sh`
- Repro: `/mnt/CT500MX500SSD1/grimoire-ab/repro.py`
- Results: `/mnt/CT500MX500SSD1/grimoire-ab/results/`
- Configs: `/mnt/CT500MX500SSD1/grimoire-ab/configs/models-180k.json` and `models-210k.json`

### Matrix Results

| Variant | ctx-size | Result | Key evidence |
| --- | ---: | --- | --- |
| Unpatched, CUDA graphs ON | 210000 | Fail | HTTP 502; original `cuMemCreate(&handle, reserve_size, &prop, 0)` VMM OOM reproduced in `unpatched-graphs-on-210k/docker.log` |
| Unpatched, CUDA graphs ON | 180000 | Pass | Stream completed with `[DONE]`; draft-context compute allocation warnings were present but non-fatal |
| Unpatched, CUDA graphs OFF | 210000 | Pass | Stream completed with `[DONE]`; draft-context KV allocation warnings were present but non-fatal |
| Unpatched, CUDA graphs OFF | 180000 | Pass | Stream completed with `[DONE]`; draft-context compute allocation warnings were present but non-fatal |
| Patched, CUDA graphs ON | 210000 | Fail | HTTP 409; `CUDA error: operation not permitted when stream is capturing` |
| Patched, CUDA graphs ON | 180000 | Fail | HTTP 409; `CUDA error: operation not permitted when stream is capturing` |
| Patched, CUDA graphs OFF | 210000 | Fail | HTTP 502; direct FA temp-buffer `cudaMalloc((void **) &ptr, nelements * sizeof(half))` OOM |
| Patched, CUDA graphs OFF | 180000 | Pass | Stream completed with `[DONE]` |

### Corrected Backtracking Conclusions

The first-pass interpretation overclaimed the V1 patch. The matrix is useful, but it is confounded and does not prove that V1 is a production-ready mitigation.

All matrix variants that reached model startup logged draft-context initialization failure, for example:

```text
failed to create draft context
slot   load_model: id  0 | task -1 | speculative decoding context not initialized
```

That means the successful stream completions were base-context runs, not full NextN/MTP-equivalent validations. A result with those log lines must be marked invalid for proving speculative/MTP behavior.

The patched CUDA-graphs-ON failures at both 180k and 210k are not context-size failures. They are V1 patch correctness failures. The log shows the patched RAII destructor calling `cudaStreamSynchronize(stream)` while CUDA graph capture is active:

```text
CUDA error: operation not permitted when stream is capturing
current device: 0, in function ~fa_f16_alloc at .../fattn-common.cuh:1310
cudaStreamSynchronize(stream)
```

Therefore V1 is CUDA-graph-incompatible by construction: it performs allocation/free/synchronization in a path that can execute during graph capture.

The 210k patched-graphs-OFF failure is also important. V1 changed the original failure from VMM `cuMemCreate` OOM to direct FA temp-buffer `cudaMalloc` OOM:

```text
CUDA error: out of memory
current device: 0, in function alloc at .../fattn-common.cuh:1315
cudaMalloc((void **) &ptr, nelements * sizeof(half))
```

This means V1 is not a pure improvement. By bypassing the pool with per-call raw allocations, it can lose pool reuse and increase peak raw allocation pressure. In this replacement-payload matrix, unpatched + CUDA graphs OFF completed at 210k while patched + CUDA graphs OFF did not, so V1 can be worse than unpatched graphs-off for that shape.

The safest corrected conclusion is:

- V1 demonstrates that FA f16 temp buffers are implicated in the VMM crash path, but V1 is not acceptable as a final fix.
- Disabling CUDA graphs appears to avoid the original VMM crash for the replacement 210k payload, but the run is not full MTP validation because draft context failed to initialize.
- Lowering `qwen3.6-mtp-27B` to 180k remains a conservative VRAM-margin mitigation, but the current matrix does not prove that the V1 patch is required.
- Any future fix must be CUDA-graph-compatible and must validate with NextN/MTP draft context actually initialized.

## Restored Runtime After Matrix - 2026-05-24

After the matrix, the live `grimoire` service was restored using the base compose file with image `grimoire:local` and repo `etc/models.json` (`qwen3.6-mtp-27B` at `ctx-size: 180000`). The service became healthy on port `9001`.

## Patch V2 Authoritative Plan

Status: definitive source of truth for the V2 fix. Supersedes every earlier "Proposed Fix" and option-letter discussion above; those remain only as historical context for how the diagnosis evolved.

Revision: iteration 15 (2026-05-25) — converged. Multiple consecutive independent reviews (iterations 8, 9, 12, 13) returned "converged with minor wording only" with no source-code contradictions; remaining findings are wording/comment additions to protect against future drift. This revision is the source of truth. Line numbers below are approximate (±a few lines); verify by symbol when implementing.

### Scope

V2 is a targeted CUDA flash-attention f16 K/V scratch fix for the original Grimoire crash path:

```text
ggml_cuda_pool_vmm::alloc -> launch_fattn -> flash_attn_ext -> llama_decode
```

It must remove CUDA FA f16 K/V dequant scratch from the VMM pool without:

- reintroducing V1's per-call raw allocation, capture-time synchronization, or CUDA graph incompatibility;
- creating a host-thread data race between the target decode path and the async MTP worker, which share one `ggml_backend_cuda_context`;
- silently disabling NextN/MTP, draft context, or CUDA graphs to make a run pass;
- leaving any captured graph executable baked with a stale scratch pointer after growth.

V2 is **not** a global CUDA OOM fix. Remaining FA temporaries such as `KV_max`, `dst_tmp`, and `dst_tmp_meta` still allocate from `ctx.pool()` and can still hit VMM `cuMemCreate` OOM unless the separate allocator-robustness work described in *Candidate B Boundary* is added later. Do not claim V2 makes every VMM OOM recoverable.

### Source Facts

The exact Atomic patch target is `/mnt/MX500/grimoire-ab/atomic-src`, pinned to `0a635dcd92ba66c75fccfef91c3e106f4668f367`. Relevant facts from that source:

- `ggml_cuda_pool_vmm::alloc()` still calls `CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0))` directly (`ggml/src/ggml-cuda/ggml-cuda.cu` ~line 505).
- CUDA `launch_fattn()` uses `ggml_cuda_pool_alloc<half> K_f16(pool)` and `V_f16(pool)` for non-HIP builds (`ggml/src/ggml-cuda/fattn-common.cuh:1323-1324`). The allocation is gated by `need_f16_K && K->type != GGML_TYPE_F16` (`:1340`) and the symmetric V gate (`:1367`).
- HIP has a scoped raw allocation workaround at `fattn-common.cuh:1320-1321`, but that pattern is not acceptable for CUDA because allocation/free/sync can run during CUDA graph capture.
- CUDA graph update detection compares graph tensor/source properties, not hidden scratch pointers passed as kernel arguments.
- CUDA execution can change `curr_stream_no` inside graph-optimized concurrent regions, and temporary pools are per device and per stream slot.
- `cuda_graphs` is `std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>>` keyed by `cgraph->nodes[0]` (map declared in `common.cuh`; keying helper `ggml_cuda_graph_get_key` in `ggml-cuda.cu` ~3167-3169). Multiple distinct graphs can be resident in a single cuda_ctx — distinct `nodes[0]` values from `sched`, from `sched_mtp`, from different cgraph shapes, or from CPU/GPU split changes all coexist. Invalidation reasoning must address the whole map, not a single entry.
- `ggml_cuda_graph::node_props` (`ggml-cuda.cu` ~3179-3199) currently stores `ggml_tensor` properties. It does **not** store the captured FA stream slot or the scratch pointer that was baked into kernel arguments. V2 must add the missing state explicitly (see §1).
- The cuda_ctx destructor body (`ggml-cuda.cu` ~580-597) acquires `ggml_cuda_lock` and waits on `ggml_cuda_lock_cv` for `ggml_cuda_lock_counter == 0` before any cleanup. It then destroys `copy_event`, `streams`, and `cublas_handles`. It does **not** explicitly destroy `cuda_graphs`; the map is currently freed by member destruction after the body returns, in reverse declaration order.
- `cudaGraphLaunch(graph->instance, cuda_ctx->stream())` (`ggml-cuda.cu` ~4190, **inside `ggml_cuda_graph_evaluate_and_capture`**) launches the captured graph through the **launch stream** — the stream returned by `cuda_ctx->stream()` at the call site, which is `streams[device][curr_stream_no]`. `cudaStreamEndCapture` (~4169) and `cudaGraphInstantiate` (~4184) also live inside `evaluate_and_capture`, **not** inside `graph_compute`. `graph_compute` (~4220-4272) only handles preflight decision-making, `cudaStreamBeginCapture` (~4266), the call into `evaluate_and_capture` (~4269), and the trailing `return GGML_STATUS_SUCCESS` (~4271). The §8 lock therefore holds across `evaluate_and_capture` and covers all of begin-capture, the per-node body, end-capture, instantiate, and launch.
   The captured graph's internal stream-fork topology is preserved on replay, but from the host's synchronization perspective the launch stream is the wait point for "the graph completed". A device-level event recorded on the capture-time slot stream `s` does **not** synchronize against a graph replay that was launched on stream 0; replays must be drained via the launch stream.
- `ggml_backend_cuda_graph_compute()` is invoked **once per split** by `ggml_backend_sched_graph_compute_async` (`ggml-backend.cpp` ~1671, ~1693). On configurations with CPU/GPU splits (e.g., `--n-cpu-moe`), one decode produces multiple CUDA-backend calls, each with its own sub-cgraph keyed by its own `nodes[0]` in `cuda_graphs`. V2's per-call reservation, §3 step 5 traversal, and §8 lock acquire/release therefore run once per split, not once per decode.
- `ggml_backend_cuda_graph_optimize()` (`ggml-cuda.cu` ~4299-4324) runs during scheduler split planning (`ggml-backend.cpp` ~1415), **before** `graph_compute_async`. It calls `ggml_cuda_graph_set_enabled()` → `cuda_ctx->cuda_graph(graph_key)`, which **inserts into** the shared `cuda_graphs` `unordered_map`. With `sched` and `sched_mtp` sharing one cuda_ctx, target's `graph_optimize` (target host thread) and MTP's `graph_compute_async` (MTP worker thread) can run concurrently and both touch the same map. The §8 lock as originally drafted covered only compute; §8 must extend to `graph_optimize` as well — see §8 below.
- `ggml_backend_cuda_context` lives in libggml (`ggml/src/ggml-cuda/`); `mtp_worker` is a `llama_context` member in libllama (`src/llama-context.cpp` ~405). The two layers are intentionally independent. The destructor in libggml therefore cannot reference `mtp_worker.joinable()`. The "no concurrent compute submission at backend teardown" contract is enforced at the libllama layer: `llama_context::~llama_context()` (`src/llama-context.cpp` ~403-431) joins `mtp_worker` near the top of its body (before any member destruction); subsequent body work only iterates `backend_ptrs` for buffer-size logs and calls `ggml_opt_free(opt_ctx)`. Member declaration order in `src/llama-context.h` is `sched` (~389), `backends` (~394), `sched_mtp` (~421); reverse-declaration-order teardown therefore destroys **`sched_mtp` first, then `backends` (which owns the cuda_ctx), then `sched` last**. What matters for V2's correctness is (a) the MTP worker is joined before any member destruction, so MTP cannot still be submitting compute; and (b) the destroying thread is `~llama_context()` itself, so the target host thread is not inside `graph_compute`/`graph_optimize` either. The cuda_ctx destructor therefore runs while no other thread submits compute on it.
- Beyond the canonical readers in `ggml-cuda.cu`, `cuda_graphs` is also read by `mean.cu` at ~37, 38, 41, 42 via the helpers `any_cuda_graph_has_instance()` and `any_cuda_graph_enabled()` (defined in `common.cuh` ~1381, ~1391). Those helpers range-iterate the map. They run inside `ggml_cuda_compute_forward` → `ggml_cuda_graph_evaluate_and_capture` → `ggml_backend_cuda_graph_compute()`, so under V2 they are already on the §8-locked path. V2 must not introduce or tolerate any new `cuda_graphs` reader that does not hold `fattn_compute_mu` on the owning cuda_ctx — this is the documented invariant in §8 below.
- `cuda_graphs`, `~ggml_cuda_graph()`, `fa_borrowed_streams`, and the `active_capture_graph` pointer all live under `#ifdef USE_CUDA_GRAPH` (`common.cuh` ~1174-1202, ~1365-1399). V2 must guard every site that references them with the same macro. The non-graph build path still allocates FA scratch and still runs the §3 sizing/reservation/borrow flow, but skips steps 5 (`cuda_graphs` traversal) and the capture-related lifecycle around `active_capture_graph`.
- `cuda_graphs` is declared inside `struct ggml_backend_cuda_context` (`common.cuh` ~1355, ~1368). C++ struct default access is public; the map and `cuda_graphs.size()` are already directly accessible from any TU that includes the header. V2 does not need a new accessor.
- **Shared backend across schedulers.** `sched_mtp` is constructed with the same `backend_ptrs` array as the main `sched` (`src/llama-context.cpp` ~1261). The two schedulers therefore share one `ggml_backend_cuda_context` per device, the same `streams[device][stream_no]` table, the same `cuda_graphs` collection, and (under the V2 design) the same `fattn_scratch[device][stream]` table.
- **Concurrent submission is intentional.** `llama_context::graph_compute_mtp()` (MTP worker, function at ~1412) acquires `backend_cfg_mu` at ~1425 and submits via `ggml_backend_sched_graph_compute_async(sched_mtp.get(), ...)` at ~1439. `llama_context::graph_compute()` (target, function at ~2858) acquires `backend_cfg_mu` at ~2869 and submits via `ggml_backend_sched_graph_compute_async(sched.get(), ...)` at ~2882. Both functions drop `backend_cfg_mu` before the submit. The dropped lock is documented in source as enabling target/MTP overlap. The MTP worker is one dedicated thread (`mtp_worker_loop`); two host threads therefore concurrently enter `ggml_backend_cuda_graph_compute()` on the same cuda_ctx.
- **`ggml_cuda_lock` scope.** The mutex defined at `ggml/src/ggml-cuda/ggml-cuda.cu` ~573-578 is documented to guard cuBLAS handle destruction against another thread's capture. It is acquired in the cuda_ctx destructor (which then waits on `ggml_cuda_lock_cv` for `ggml_cuda_lock_counter == 0`), and around the counter increment/decrement (~4262-4264 and the symmetric block at capture end ~4172-4175). `cudaStreamBeginCapture` itself is called *outside* the lock (~4266). The lock therefore does not cover scratch reserve, scratch borrow, scratch grow, scratch free, `cudaMalloc`, or the body of a capture region. It cannot be relied on as an FA-scratch concurrency primitive.

### Required Design

V2 has eight requirements. All must hold; an implementation that satisfies seven of eight is not V2.

#### 1. CUDA FA scratch owner on the backend context

Add a scratch owner to `ggml_backend_cuda_context`. It tracks stable `half * k` and `half * v` pointers, byte capacities, a retired-allocation list (see §6), and the per-context concurrency primitive defined in §8. Scratch is keyed by device and by the stream slot that will actually execute FA:

```text
fattn_scratch[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]
```

The `[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]` shape matches the existing `streams[]`/`pools[]` declarations in `common.cuh` as a stylistic convention. A given cuda_ctx only reads/writes its own `[device][...]` row; the other rows are intentionally dead memory and must not be referenced by any borrower or preflight code path. Collapsing the shape to `[GGML_CUDA_MAX_STREAMS]` would also be correct; V2 keeps the wider shape to match the surrounding members and reduce diff churn, not because multi-row indexing is needed.

V2 also adds a `std::set<int> fa_borrowed_streams` member (or equivalent fixed-width bitset over `GGML_CUDA_MAX_STREAMS`) to `ggml_cuda_graph` so the patch can track, per graph entry, which scratch slots were borrowed during the most recent capture. §3 step 5 uses this set to decide which entries need invalidation on growth — without per-graph slot tracking the invalidation predicate is unimplementable, because `node_props` does not currently record the captured slot.

`ggml_backend_cuda_context` is per-device, so the scratch table, retired list, `fattn_compute_mu`, and any per-context bookkeeping are per-device. There is no global lock across devices; multi-GPU workloads keep their device-parallelism across `cuda_ctx` instances.

#### 2. Dispatch-accurate scratch sizing helper

Declare the helper in `fattn.cuh`; define it in `fattn.cu` where it can call `ggml_cuda_get_best_fattn_kernel` (which is `static` in that TU and must stay so). Do not hoist the implementation into the header.

```text
ggml_cuda_flash_attn_ext_f16_scratch_size(device, dst, &k_bytes, &v_bytes)
```

The helper is **read-only** over the cgraph node — it must not mutate `graph->node_props`, `warmup_complete`, or any graph state. It calls `ggml_cuda_get_best_fattn_kernel(device, dst)` (already side-effect-free; reads only tensor metadata) and sizes per the returned `BEST_FATTN_KERNEL_*`, mirroring the allocation gates in `launch_fattn()` (`fattn-common.cuh` ~1340, ~1367). Do **not** roll a hand-rolled "minimum predicate" to identify VEC — the selector at `fattn.cu:360-599` returns `BEST_FATTN_KERNEL_VEC` from many branches (`Q->ne[1] <= 2`, `gqa_opt_applies`, MMA/WMMA availability fallbacks, etc.); any shortcut will misclassify and produce wrong sizing.

Per-kernel sizing rules (each side computed independently; byte size is `ggml_nelements(K_or_V) * sizeof(half)`):

- **TILE, WMMA, MMA.** `launch_fattn(..., true, true, ...)` — `need_f16_K = need_f16_V = true`. Each side allocates when the corresponding K/V tensor is **not** `GGML_TYPE_F16` (any of F32, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, ...). Worst-case sizing follows directly from K/V metadata; the kernel enum does not change the answer.
- **VEC.** `need_f16_K = (type_K == GGML_TYPE_F16)` and `need_f16_V = (type_V == GGML_TYPE_F16)` at `fattn-vec.cuh` ~562-563 — `type_K` / `type_V` here are the **template** types selected by the `FATTN_VEC_CASE` macro at `fattn.cu:223-231`. That macro accepts the runtime tensor by `K->type == type_K || (K->type == GGML_TYPE_F32 && type_K == GGML_TYPE_F16)`, so a runtime `K->type == F32` is legitimately routed through the F16 template. In that case `need_f16_K = true` **and** `K->type != F16`, so the allocation gate fires and VEC allocates `ggml_nelements(K) * sizeof(half)`. Symmetric for V, with one further reduction: when `V_is_K_view(V, K)` returns true, `launch_fattn` at `fattn-common.cuh:1367-1372` short-circuits and V reuses K's scratch — sizing must return 0 bytes for V in that aliased case even when V would otherwise allocate. **The earlier "VEC contributes zero" shortcut is wrong and must not be reintroduced.** Correct VEC rule: K allocates when `K->type == GGML_TYPE_F32`; V allocates when `V->type == GGML_TYPE_F32` **and** `!V_is_K_view(V, K)`.
- **`BEST_FATTN_KERNEL_NONE`.** Return 0; the abort fires at execution anyway.

Sizing must use the **same** `V_is_K_view` predicate that `launch_fattn()` uses, by sharing a small inline helper in `fattn.cuh`. Do not copy-paste `V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs))` — copy-pasted predicates drift on upstream rebase and produce wrong-sized scratch.

Do **not** add a `GGML_ASSERT(!(need_f16_K && K->type != GGML_TYPE_F16))` at any site, in any file. The VEC F32 dispatch makes that expression legitimately true; an assertion there would fire on valid workloads. V2 carries **no** VEC-related runtime invariant assertion; consistency is enforced through the debug-build sizing-vs-actual drift assertion (see *Validation Requirements*) instead. Any earlier wording suggesting a VEC invariant assertion belongs in the CUDA sizing helper has been retracted; it referenced the buggy form above.

#### 3. Pre-execution reservation in `ggml_backend_cuda_graph_compute()`

V2 inserts sizing and reservation **before** the existing decision block (around `ggml-cuda.cu` ~4233) that calls `ggml_cuda_graph_update_required()` and derives `use_cuda_graph` / `cuda_graph_update_required`. The existing function's call site is **not** moved; V2 leaves the decision block's internal semantics unchanged.

Reservation must run before direct execution, graph warmup, graph capture, and graph replay. Reserving only before `cudaStreamBeginCapture()` is insufficient.

Required order, executed inside the §8 critical section (acquired at function entry, before step 0):

0. normalize `cuda_ctx->curr_stream_no = 0`. The field lives on cuda_ctx and persists across calls; the live join-reset at `:3750` returns it to 0 under normal flow. `curr_stream_no` is mutated only inside `evaluate_and_capture` (`:3762`/`:3771` set, `:3750` reset), and `evaluate_and_capture` only runs under §8 lock (V2's invariant), so the field is effectively thread-private within the lock window — the other thread is blocked at §8 acquire and cannot observe or mutate it. In the **current** source, every path between set-nonzero and reset-to-zero is `CUDA_CHECK`-guarded; the only way to leave `curr_stream_no` nonzero is a process abort. Step 0 is therefore **defensive against future refactors** that introduce a softer error path (e.g., a `GGML_STATUS_FAILED` return from inside the concurrent region). The pre-existing `GGML_ASSERT(cuda_ctx->curr_stream_no == 0)` at `:3666` corroborates the intended invariant on the fork-entry path. Document this rationale in the in-source comment so a future reader does not delete the normalization assuming it is dead code;
1. drain the §6 retired list by polling `cudaEventQuery()` on **every** event in each retired entry's event set; once all events signal, `cudaFree` the K and V pointers, `cudaEventDestroy` each event in the set, and remove the entry. The expected steady-state list size is bounded by the number of growths in the previous compute window (typically 0–1). Bootstrap regression threshold is **4 entries**; the long-soak validation case must record the observed steady-state list size so the threshold can be retuned on data. If the list exceeds the threshold, first log a regression-class signal; then choose **one** of two recovery paths and document the choice: (a) block once with `cudaStreamSynchronize()` on each populated stream slot of device `d` — paying the long pause inside §8 lock — or (b) return `GGML_STATUS_ALLOC_FAILED` and let the caller decide. Either path may be taken; the default until validation data exists is (a). `cudaDeviceSynchronize()` remains reserved for the destructor. The §6 rule ("destructor is the only place a blocking sync on retired memory is permitted") applies to **device-wide** sync; per-stream sync used here as a regression escape hatch is explicitly permitted;
2. determine FA scratch needs for the graph by sweeping `cgraph->nodes[]`: for each `node` with `node->op == GGML_OP_FLASH_ATTN_EXT`, call the §2 helper with `dst = node`. Read-only sweep — no mutation of `node_props`, `warmup_complete`, or any graph state;
3. resolve the FA execution stream slot for each FA node (§7);
4. reserve or grow `fattn_scratch[d][s]` outside any active capture. Reservation failure must be detected here, **before** step 5 mutates any graph entry; never enter step 5 with partial state. Before any growth `cudaMalloc`, assert `cudaStreamIsCapturing(streams[d][s]) != cudaStreamCaptureStatusActive`. The §8 lock already prevents cross-thread interference; this assertion guards against a future edit that moves reservation **below** `cudaStreamBeginCapture` (`:4266`) in the same function. Without the assert, that mis-ordering would surface only as a downstream `operation not permitted when stream is capturing`, far from the actual cause;
5. on growth (pointer change) for slot `(d, s)`, traverse the entire `cuda_graphs` map and for every entry where `fa_borrowed_streams` contains `s` (including the current graph entry if it matches): if `instance != nullptr`, call non-aborting `cudaGraphExecDestroy(instance)` and set `instance = nullptr` (skip the call entirely when `instance` is already null — calling `cudaGraphExecDestroy(nullptr)` returns `cudaErrorInvalidValue`; the captured-but-not-instantiated state where set is populated but instance is null is reachable if a prior `cudaGraphInstantiate` aborted between capture-completion and instantiation success); if `graph->graph != nullptr`, call non-aborting `cudaGraphDestroy(graph->graph)` and set it to `nullptr`; **also reset `warmup_complete = false`** so the entry re-enters warmup on its next compute (defensive: the existing `:4252` path reads `instance == nullptr` and sets `cuda_graph_update_required = true`, but a future edit to that decision block that gates recapture on `properties_changed` would silently skip recapture and `cudaGraphLaunch` on a null instance — resetting warmup forces the entry through the post-warmup decision again from a known state); clear that entry's `fa_borrowed_streams`. The traversal uses raw `cudaError_t` for both destroy calls, logs non-success returns, and continues so iteration cannot abort mid-map. (`graph->node_props` is intentionally **not** cleared — preserving it costs nothing and avoids re-running property detection on the next compute; recapture is forced by the instance/warmup state.)

   Then move the old pointer `A` into the retired list along with `(d, s)` and a set of fresh `cudaEvent_t` records that proves no in-flight work still references `A` on completion. Event recording rule: record one event on **each populated stream slot** for device `d` — i.e., for every `s' ∈ [0, GGML_CUDA_MAX_STREAMS)` where `streams[d][s']` has been created. `cudaGraphLaunch` enqueues a replay through the **launch stream** (`cuda_ctx->stream()` at the call site, typically slot 0 in production, possibly nonzero under `GGML_CUDA_GRAPH_OPT=1` with one visible GPU), not through the capture-time slot. A single event on slot `s` would miss past replays launched on slot 0. Recording on all populated slots is cheap (a few events per growth, a rare operation) and covers replays and direct-execute kernels on any stream uniformly. The retired entry holds the event set; step 1 reclaim frees `A` only after **every** event in the set has signalled `cudaSuccess`.

   Streams are lazy-created (`ggml-cuda.cu` ~587-592 iterates `streams[i][j] != nullptr`). Event-recording approach: record events on **populated slots only** at growth time. To close the race where a stream slot comes into existence between growth and the next drain, V2 also records an event on each newly-created stream the **first time** it is used and tracks the latest "first-use event" per slot on cuda_ctx. On retired-entry reclaim (step 1), the drain check polls each retired event AND the per-slot first-use events for any slot created since the retire moment. This avoids the alternative ("record on all 8 slots at growth, lazy-creating them") which would call `cuda_ctx->stream(d, s')` for unused slots — and that call invokes `CUDA_CHECK(cudaStreamCreateWithFlags(...))` (`common.cuh:1410-1416`), which aborts the process on creation failure, inside the §8 lock, at the moment we're trying to engineer recoverable growth. The populated-slots-plus-lazy-tracking design avoids any unconditional stream creation under the lock.

   The reclaim loop in step 1 must call `cudaEventDestroy` on each event after the entry is freed so handles don't leak.

   Invalidating only the graph being computed now is insufficient: other entries (notably MTP's own graph, keyed by its own `cgraph->nodes[0]`) can hold an executable that baked the old pointer into its captured kernel arguments; replay would then read stale K/V dequant memory and silently produce wrong results;
6. if reservation fails, **return early** from `ggml_backend_cuda_graph_compute()` with `GGML_STATUS_ALLOC_FAILED`. The function currently returns `GGML_STATUS_SUCCESS` unconditionally (`ggml-cuda.cu` ~4271); V2 changes the return on the failure branch only — the success branch and `ggml_cuda_graph_evaluate_and_capture()` itself (which remains `void`-returning and `CUDA_CHECK`-aborting) are unchanged. No `node_props` advance, no warmup-state change, no entry into the existing `:4233+` decision block. `fa_borrowed_streams` on the current entry is **not** modified on failure — its existing state still corresponds to the previous successful capture's instance, which has not been invalidated;
7. on success, fall through to the existing decision/capture/launch block. The current entry's `fa_borrowed_streams` is populated by §4's borrower while the new capture is active. Invalidations made in step 5 against **other** entries persist on those entries (`instance == nullptr`); each invalidated entry's next own `ggml_backend_cuda_graph_compute()` call is the point at which the existing `:4252` branch reads `instance == nullptr` and sets `cuda_graph_update_required = true`, forcing that entry's recapture. No global re-instantiation happens at growth time; invalidation is a per-entry persistent flag.

`fa_borrowed_streams` lifecycle (consolidated rule): the set is cleared whenever the corresponding `instance` is destroyed — by step 5 invalidation, by any other path that destroys the executable (existing recapture flow, destructor), or as part of `ggml_cuda_graph::~ggml_cuda_graph()`. The borrower in §4 only adds; it never clears. There is no "clear at capture start" — the previous instance is already destroyed before recapture, so the set is already empty when the new borrower runs.

Note on sweep duplication: the §2 read-only sweep (step 2) and the existing `ggml_cuda_graph_update_required` (`:4233`) both walk `cgraph->nodes[]`. V2 intentionally keeps them as two separate sweeps — folding them would force the size helper into the mutating `node_props` path. The cost is O(n_nodes) duplication inside the §8 lock window; validation must record sweep duration as a component of lock-hold time (see *Validation Requirements*).

Do not rely on graph property comparison or `cudaGraphExecUpdate()` to repair hidden scratch pointer changes.

#### 4. `launch_fattn()` is a pure borrower for CUDA

For CUDA builds:

- remove `ggml_cuda_pool_alloc<half> K_f16/V_f16`;
- replace them with non-owning scratch borrowers that return the stable pointer reserved in §3 by the current host thread for the current device and stream slot;
- the borrower records `s` into `cuda_ctx->active_capture_graph->fa_borrowed_streams` whenever `cuda_ctx->active_capture_graph != nullptr`. That pointer is set in `ggml_backend_cuda_graph_compute()` immediately after `cudaStreamBeginCapture` (~4266); a local RAII guard at `graph_compute` scope clears it on guard dtor. The guard's scope encloses both `evaluate_and_capture` (where borrowers run) and the subsequent launch path, so the pointer is visible throughout the capture/launch region. The borrower **does not** call `cudaStreamIsCapturing(stream)` — relying on the manual pointer is a single source of truth and avoids the race window between driver state and patch-side tracking. When `active_capture_graph == nullptr` (direct-execute path: warmup, properties-changed reset at `:4246-4249`, or graph disabled), the borrower skips the recording — direct-execute paths produce no replayable executable, so growth invalidation has nothing to do for them. (The capture-state assertion that prevents growth from running during active capture lives in §3 step 4, where growth actually happens — not here.)
- the borrower indexes `fattn_scratch[d][curr_stream_no]`. If `fattn_scratch[d][curr_stream_no].k == nullptr` or `capacity < request`, the §3 preflight resolver and the live `curr_stream_no` have diverged — this is a §3/§7 defect, not a runtime expected case. Debug builds `GGML_ASSERT` panic; release builds return a recoverable failure via the same path as §5 (`GGML_STATUS_ALLOC_FAILED` propagated back from `ggml_backend_cuda_graph_compute`). The borrower must **never** fall back to a `cudaMalloc` inside `launch_fattn`; that would revive V1's failure mode.
- Warmup-incomplete computes (`warmup_complete == false`, first call: `:4237-4243` keeps `use_cuda_graph = false`) are also direct-execute paths from the borrower's perspective. FA kernels do execute against scratch on those calls, but no capture runs, so `active_capture_graph` is null and the borrower skips `fa_borrowed_streams` recording. This is the correct behavior: warmup-incomplete compute produces no replayable executable, so growth invalidation has nothing to do for it later.
- borrower `alloc(nelements)` checks that preflight capacity is sufficient and returns the stable pointer;
- add a debug assertion that actual borrower requests fit preflight capacity (`ptr != nullptr && capacity >= request`);
- no `cudaMalloc`, `cudaFree`, `cudaStreamSynchronize`, `cuMemUnmap`, or VMM remap may run from `launch_fattn()`.

Keep the HIP scoped allocation path separate unless HIP is being deliberately refactored.

#### 5. Recoverable scratch allocation failure

Scratch reserve/grow must use raw `cudaError_t` handling, not `CUDA_CHECK(cudaMalloc(...))`.

Requirements:

- clean up partial K/V allocation if only one side succeeds;
- preserve the old scratch pointers if growth fails;
- return a recoverable failure, preferably `GGML_STATUS_ALLOC_FAILED`, from the backend compute boundary;
- never enter graph capture or graph replay after failed scratch reservation;
- never silently disable NextN/MTP to make a run pass;
- after a forced failure, the next successful decode must reach a clean reservable state — no partial K-only allocation left behind, no orphan retired entry, no advanced warmup state.

#### 6. Captured graph pointer lifetime and retired-pointer reclaim

If a captured graph may reference scratch pointer `A`, V2 must not `cudaFree(A)` while that graph instance can replay.

Use monotonic scratch growth for the local patch:

- allocate a larger scratch pointer outside capture;
- §3 step 5 destroys every graph executable whose `fa_borrowed_streams` contains the grown slot, so no future replay can read from `A`;
- §3 step 5 moves the old pointer `A` into a retired list owned by the cuda_ctx, alongside the affected `(d, s)` and a fresh `cudaEvent_t` recorded on stream `s` after the latest in-flight FA work;
- on each compute window's step 1, poll `cudaEventQuery()` for each retired entry. If complete, free `A`. If not yet complete, leave it for a future window. The retired list is also drained at destructor as a safety net (graceful shutdown);
- never block on `cudaEventSynchronize()` or `cudaDeviceSynchronize()` inside the §8 critical section to reclaim retired pointers. Reclaim is opportunistic via `cudaEventQuery`; under steady state the retired list converges to at most the set of graphs captured in the previous window.

Destructor ordering is mandatory and must be made **fully explicit in the body** rather than left to member declaration order:

1. existing `ggml_cuda_lock` acquisition + `ggml_cuda_lock_cv` wait for `counter == 0` (unchanged);
2. **explicit** `cuda_graphs.clear()` — destroys every `cudaGraphExec_t` / `cudaGraph_t` while no other thread is mid-capture. This step is guarded by `#ifdef USE_CUDA_GRAPH`; without that macro the `cuda_graphs` member does not exist and the build would not compile. The existing wait guarantees no other thread is mid-capture, closing the cuBLAS-vs-capture hazard. The current `~ggml_cuda_graph()` body calls `CUDA_CHECK(cudaGraphExecDestroy(...))` / `CUDA_CHECK(cudaGraphDestroy(...))`, which can abort the process on driver error while the destructor holds `ggml_cuda_lock`. V2 accepts this behavior at process teardown — graceful shutdown is already best-effort at this point — but the in-source comment must say so explicitly. (A non-aborting variant remains a future refactor option; do not introduce it as part of V2 to keep scope tight.)
3. **explicit** drain of the retired-pointer list: poll `cudaEventQuery` and free the pointers whose events are complete; for entries whose events are not yet complete, `cudaEventSynchronize` then free (destructor is the only place a blocking sync on retired memory is permitted). Runs unconditionally — `fattn_scratch` is used by FA regardless of `USE_CUDA_GRAPH`, so retired entries can exist in either build;
4. **explicit** `cudaFree` of the live `fattn_scratch[][]` K and V pointers; reset capacities. Runs unconditionally;
5. existing teardown of `copy_event`, `streams`, `cublas_handles`;
6. `pools[][]` and other members destroy by member dtor in reverse declaration order, which now runs after `cuda_graphs.clear()` and scratch teardown above. No member-order trick is required because steps 2–4 are explicit.

Guard the new steps with an in-source comment that explains the lifetime contract and references this RSH. Do not rely on member-declaration order; that is fragile to future header edits.

Destructor pre-condition (already true today, V2 inherits it; document it in the comment so a future contract break is loud): `llama_context::~llama_context()` (`src/llama-context.cpp` ~411) joins the MTP worker thread and destroys `sched`/`sched_mtp` before backend tear-down. By the time `ggml_backend_cuda_context::~ggml_backend_cuda_context()` runs, no other thread is submitting; step 3's blocking `cudaEventSynchronize` therefore cannot deadlock against another thread waiting on `fattn_compute_mu`. If a future refactor breaks this pre-condition (e.g., backend destroyed while the MTP worker is still alive), step 3 can deadlock — the in-source comment must call this out.

#### 7. Stream ownership resolution

Do not use one global scratch pointer for all streams. Do not reserve every possible stream slot by default, because that can multiply huge K/V scratch and recreate OOM pressure.

Required stream rule:

- reserve and borrow from the actual FA execution stream slot;
- factor the live resolver out of `ggml_cuda_graph_evaluate_and_capture` (which currently inlines stream assignment around `try_launch_concurrent_event` at ~3768 / ~4156) into a **side-effect-free** callable in `common.cuh` or `ggml-cuda.cu`. The preflight pass in §3 step 3 calls the side-effect-free variant; the live execution loop calls the existing in-place version. Copy-pasting the resolver would drift on rebase — exactly the failure mode §2 warns about for `V_is_K_view`;
- §7 implements a **parallel preflight predictor** that mirrors the live state machine in `evaluate_and_capture` — not a refactor extracting one helper. The live path's `try_launch_concurrent_event` (`ggml-cuda.cu` ~3657-3674) is a closure with `cudaEventRecord`/`cudaStreamWaitEvent` side effects and cannot be made side-effect-free by extraction. V2 ships two implementations of the same state machine: the live one (unchanged) and a side-effect-free predictor used in §3 step 3. Add a debug-build assertion that fires **at the FA enqueue site inside the live evaluate_and_capture loop, while §8 lock is held**, comparing `cuda_ctx->curr_stream_no` against `predicted_slot[fa_node_index]`. The assertion must run before the FA borrower is called, so a drift fires before stale scratch is captured into a graph — not as a post-hoc trace comparison;
- the preflight resolver **predicts** the effective stream-slot assignment by reading the same inputs the live path consumes (`event.is_valid()` results, `should_launch_concurrent_events`, fusion skips, join-node-resets-to-stream-0). It must not call `concurrent_events.clear()` itself, must not mutate `curr_stream_no`, and must not call into the live `try_launch_concurrent_event`. When the live path *would* clear `concurrent_events` this compute (i.e. `should_launch_concurrent_events == false`), the resolver predicts **every FA node executes on stream 0** for *this* compute window; it does not cache or carry over the stream-slot prediction to the next compute window, because the next window's `concurrent_events` may be repopulated by a fresh `graph_optimize`;
- `curr_stream_no` lives on `cuda_ctx` and persists across calls; the live join-reset at `:3750` returns it to 0 under normal flow, but a previous compute that exited mid-region could leave it non-zero. `graph_compute` must **normalize `curr_stream_no` to 0 at function entry inside the §8 lock**, before §3 step 3 runs. The existing `GGML_ASSERT(cuda_ctx->curr_stream_no == 0)` at `:3666` only fires on the fork path; production (no concurrent events) never reaches it, so V2 cannot rely on the assert for normalization;
- use the resolved `stream_context.concurrent_events` schedule when graph optimization assigns FA to a nonzero stream;
- if FA is the join node and execution has returned to stream 0, reserve and borrow stream 0;
- log/assert any FA node whose stream cannot be resolved;
- validation must include a graph-optimized run that logs the FA borrow stream slot.

Note on production reach: Atomic's graph optimization (`ggml-cuda.cu` ~4322) early-exits unless `ggml_backend_cuda_get_device_count() == 1`. On Grimoire's production 2-GPU host, `curr_stream_no` therefore stays at 0 and §7's nonzero-stream rules act as correctness insurance for single-visible-GPU test runs (`CUDA_VISIBLE_DEVICES=<one GPU>`). A regression in §7's nonzero-stream code path would be invisible in production logs even if `fattn_compute_mu` itself works — single-visible-GPU validation is the only surface that exercises it.

#### 8. Concurrency contract on the shared CUDA backend

Because `sched` and `sched_mtp` share one `ggml_backend_cuda_context` per device, the V2 design must close the host-thread race on shared scratch, shared streams, and the shared `cuda_graphs` collection. A scratch pointer that is stable from one thread's perspective can still be invalidated by the other thread's growth between borrow and kernel launch, and capture/replay state can be mutated by either thread.

Required rule: serialize the entire FA-touching window of one `ggml_backend_cuda_graph_compute()` invocation on the shared cuda_ctx, **and** the `cuda_graphs` map insertion that happens earlier in `ggml_backend_cuda_graph_optimize()`.

- Add a `std::mutex` member to `ggml_backend_cuda_context` (suggested name `fattn_compute_mu`). The lock is per cuda_ctx, hence per device — multi-GPU compute remains parallel across devices.
- The lock is acquired at **function entry** of `ggml_backend_cuda_graph_compute()`, before §3 step 1 (retired-list reclaim). It is released only after `cudaGraphLaunch` (or the equivalent direct-execute path) returns. The lock covers retired-pointer reclaim, sizing, reservation, growth, the §3 step 5 traversal, `cuda_graphs` mutations, the existing decision block, `cudaStreamBeginCapture` / `cudaStreamEndCapture`, `cudaGraphInstantiate` / `cudaGraphExecUpdate`, and `cudaGraphLaunch`. Acquiring the lock later would leave a window where another thread could grow scratch and invalidate an in-flight capture.
- The lock is **also** acquired at function entry of `ggml_backend_cuda_graph_optimize()`, around the `ggml_cuda_graph_set_enabled()` → `cuda_ctx->cuda_graph(graph_key)` call that inserts into `cuda_graphs`. Otherwise target's planning thread and the MTP worker thread can mutate the shared `unordered_map` concurrently. The lock window in `graph_optimize` is brief (no compute) and does not measurably affect throughput; it only serializes map mutation against the parallel scheduler thread. The acquisition happens regardless of `GGML_CUDA_GRAPH_OPT`'s env-gate value because the insert at `:4304` runs before the env check at `:4322`.
- **Shared cuda_ctx state access invariant.** Every read or write of shared cuda_ctx state from outside the destructor — `cuda_graphs` (via `cuda_ctx->cuda_graph(graph_key)`, `cuda_graphs.size()`, `cuda_graphs.clear()`, range iteration in `any_cuda_graph_enabled()`/`any_cuda_graph_has_instance()` at `common.cuh` ~1381/~1391, used by `mean.cu` ~37-42; and the §3 step 5 traversal), `concurrent_stream_context` (`common.cuh` ~1406; read by `evaluate_and_capture`, mutated by `graph_optimize` via `stream_context.reset()` at ~4321), `curr_stream_no`, `fattn_scratch[][]`, and the retired list — must occur with `fattn_compute_mu` held on the owning cuda_ctx, or run from the destructor under the documented pre-condition. All current callers are inside `graph_compute`/`graph_optimize`/`evaluate_and_capture`/destructor, so V2's lock placement covers them; any new caller must be added with the lock or the invariant breaks. Patch validation should re-grep these member names and confirm every caller is on the locked path.
- **Lock release point.** The lock is held across the **entire `evaluate_and_capture` call** invoked from `graph_compute` (~4269), which itself contains `cudaStreamEndCapture` (~4169), `cudaGraphInstantiate` (~4184), `ggml_cuda_graph_update_executable` (~4187), and `cudaGraphLaunch` (~4190). `graph_compute` itself does **not** call instantiate/launch — those all live inside `evaluate_and_capture`. After `evaluate_and_capture` returns, `graph_compute` only returns `GGML_STATUS_SUCCESS` (~4271). The lock therefore releases on `graph_compute` function exit, on both the capture branch and the direct-execute branch. Do not release earlier; do not look for instantiate/launch in `graph_compute` body.
- The simplest correct implementation holds the lock unconditionally across the whole function body. Optimizing the lock window — e.g., releasing earlier when sizing returns zero — is deferred to a follow-up only after the §Validation measurement establishes that serialization is the bottleneck.
- `fattn_compute_mu` is independent of `ggml_cuda_lock`. `ggml_cuda_lock` keeps its documented role (cuBLAS handle teardown vs. capture counter).
- `backend_cfg_mu` in `llama_context` is not changed. V2 does **not** re-hold `backend_cfg_mu` across `graph_compute_async`; the new serialization lives inside the CUDA backend.
- The destructor path must not acquire `fattn_compute_mu`. `ggml_backend_cuda_context::~ggml_backend_cuda_context()` inherits the existing contract that the caller drained pending compute before teardown. The destructor's existing `ggml_cuda_lock_cv` wait on `counter == 0` only blocks against threads **mid-capture** (the counter is only nonzero between the increment at `:4262-4264` and the decrement at `:4172-4175`); pure replay and direct-execute paths never increment the counter, so the wait does not stop them. V2 relies on the libllama-level contract (`llama_context::~llama_context()` joins the MTP worker, destroys `sched`/`sched_mtp`, then destroys backends — all in that order, ~411) to guarantee no thread is inside `ggml_backend_cuda_graph_compute()` or `ggml_backend_cuda_graph_optimize()` when this destructor runs. The in-source comment in the destructor must state this contract explicitly; debug-only instrumentation may track outstanding `fattn_compute_mu` acquire/release pairs and assert zero at destructor entry, but no source-level assertion may reference `mtp_worker` (a libllama symbol).
- **Lock order is `fattn_compute_mu` first, then `ggml_cuda_lock`.** The compute path acquires `fattn_compute_mu` at function entry, then briefly takes `ggml_cuda_lock` around the capture counter increment/decrement (existing `:4262-4264` and `:4172-4175` blocks). No code path may ever take `ggml_cuda_lock` first and then `fattn_compute_mu`; the destructor (which holds `ggml_cuda_lock` while waiting on `counter == 0`) explicitly never acquires `fattn_compute_mu`, preserving the order.

Acknowledged cost: serializing on the shared cuda_ctx erases the host-side overlap of target and MTP submission that the dropped `backend_cfg_mu` at `:1420`/`:2864` was designed to allow. The lock window includes the entire graph capture body (`ggml_cuda_graph_evaluate_and_capture` traversing the full cgraph and enqueueing every node), so the cost is real, not nominal. The decode-throughput impact must be measured (see *Validation Requirements*). If the measured regression exceeds the pre-declared threshold, the design escalates to per-generation scratch banks keyed by `(device, stream_slot, scheduler-id-or-generation)` with explicit retirement on graph destruction. That escalation path is deferred until measurement proves it is necessary, because per-bank doubling of K/V f16 scratch at 180k+ context can re-trigger the OOM V2 is meant to prevent on 24 GB cards.

### Candidate B Boundary

Allocator robustness work is separate from V2.

Do not unmap, remap, or free VMM pool memory while active kernels or captured graphs may reference it. Do not add low-level fallback allocation inside FA launch code. A future non-aborting VMM pool design should propagate allocation failure at safe backend boundaries and must not invalidate live VMM pointers.

### Patch File Plan

A local Atomic V2 patch should touch:

- `ggml/src/ggml-cuda/common.cuh`: define FA scratch owner, reserve/borrow helpers, retired-allocation ownership with per-entry **set** of `cudaEvent_t` (one per populated stream slot of the device — see §3 step 5 event-recording rule), `fattn_compute_mu`, the `std::set<int> fa_borrowed_streams` member on `ggml_cuda_graph`, an `active_capture_graph` pointer on `ggml_backend_cuda_context`, and (debug-only) the failure-injection hook described in *Validation Requirements*. `cuda_graphs` is already directly accessible (struct default access; `cuda_graphs.size()` works from any caller); no new accessor is needed. Add an in-source comment at both lock declarations stating "lock order: `fattn_compute_mu` before `ggml_cuda_lock`; never reverse" so the rule lives next to the code, not only in this RSH. Document the destructor pre-condition (caller must have drained all `graph_compute`/`graph_optimize` calls and joined any worker threads — enforced by libllama, not by `mtp_worker.joinable()` which is not visible from this layer) in a comment at the top of `~ggml_backend_cuda_context()`. A debug-only outstanding-acquire counter on `fattn_compute_mu` may assert zero at destructor entry; no libllama-symbol assertion may be used.

  `active_capture_graph` lifecycle: declare a single RAII guard **at `graph_compute` function scope**, unconditionally; its dtor unconditionally writes `nullptr` (idempotent on a nullptr state). Inside the `if (use_cuda_graph && cuda_graph_update_required)` block (immediately after `cudaStreamBeginCapture` at ~4267) set the pointer to the current entry. The guard remains in scope across both the `evaluate_and_capture` call (where borrowers run inside the capture body) and the trailing return path — across both the capture-branch and the direct-execute-branch. Do **not** place the guard inside the `if` block; its dtor would fire at line 4267 before `evaluate_and_capture` runs and defeat the purpose. Note that `CUDA_CHECK` aborts the process on driver error; "abort mid-capture" is unrecoverable regardless of guard state.
- `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - add pre-execution retired-pointer reclaim, graph scan/reserve, stream-slot resolution (via the §7 helper extraction below), and the §3 step 5 traversal that invalidates stale executables (using non-aborting `cudaError_t` returns inside the loop);
  - propagate `GGML_STATUS_ALLOC_FAILED` from the reservation boundary;
  - in the destructor: insert explicit steps 2–4 of §6 after the existing `ggml_cuda_lock` acquisition / counter wait and before the existing tail teardown. Step 2 (`cuda_graphs.clear()`) is guarded by `#ifdef USE_CUDA_GRAPH`; steps 3 (drain retired list, with `cudaEventDestroy` after each entry's pointers are freed) and 4 (`cudaFree` live scratch) run unconditionally. The §3 step 5 traversal and the `active_capture_graph`/`fa_borrowed_streams` lifecycle code in `ggml_backend_cuda_graph_compute()` are also under `#ifdef USE_CUDA_GRAPH` (those members do not exist otherwise); the non-graph build still runs §3 steps 0-4 (sizing, reservation, growth) and the §4 borrower path. The §3 step 4 `cudaMalloc` must run after the existing `ggml_cuda_set_device(cuda_ctx->device)` call at the top of `graph_compute` (~4218) — V2's reservation block is inserted below that line, not above;
  - acquire the §8 `fattn_compute_mu` at function entry of `ggml_backend_cuda_graph_compute()`;
  - **ship a separate, side-effect-free predictor** of the stream-slot assignment that mirrors the live state machine used by `ggml_cuda_graph_evaluate_and_capture` (`try_launch_concurrent_event` at ~3657-3674 and its caller sites at ~3768/~4156). This is a **parallel implementation**, not an extraction — the live closure has `cudaEventRecord`/`cudaStreamWaitEvent` side effects and cannot be made side-effect-free. The preflight predictor is used by §3 step 3; the live path stays unchanged. A debug assertion at the FA enqueue site (inside `evaluate_and_capture`, under §8 lock) compares the predictor's output against the live `curr_stream_no` and fires on drift before stale scratch is captured;
  - do **not** move the call site of `ggml_cuda_graph_update_required()`; V2 only inserts new code before it.
- `ggml/src/ggml-cuda/fattn.cuh`: expose the scratch sizing helper and the shared `V_is_K_view` predicate.
- `ggml/src/ggml-cuda/fattn.cu`: implement dispatch-accurate scratch sizing by calling `ggml_cuda_get_best_fattn_kernel()` and applying the per-kernel rules in §2. Carry no `GGML_ASSERT` on the `need_f16_K && K->type != F16` invariant — that form fires on the legitimate VEC F32 dispatch path.
- `ggml/src/ggml-cuda/fattn-common.cuh`: convert CUDA K/V f16 temp handling to non-owning scratch borrowers; record borrowed slots into the current `ggml_cuda_graph`'s `fa_borrowed_streams` when capture is active. The existing `V_is_K_view` is a *local bool computed inline* at ~1278, not a callable helper. Edit `fattn-common.cuh:1278` to call the new shared helper extracted into `fattn.cuh`; the `if (V_is_K_view)` site at ~1368 then keeps the same boolean form but the source of the bool changes from an inline expression to a helper call.

Do not update `Dockerfile` back to `GGML_CUDA_GRAPHS=ON` as part of the patch itself. Re-enable graphs in validation builds only after V2 builds and the matrix proves graphs-on behavior with active NextN/MTP. Confirm the Atomic patch hash gate (`/app/.cache/llama-cpp-build/.atomic_patch_hash`) actually busts ccache outputs from prior V1 builds, and verify via `nm`/`strings` on the built `libggml-cuda.so` that the new scratch-owner symbol is present — without that check, "patch applied" cannot be distinguished from "build linked a stale object".

### Rejected Approaches

- Continue V1's raw `cudaMalloc`/`cudaFree` inside `launch_fattn()`.
- Synchronize streams inside `launch_fattn()`.
- Allocate, free, unmap, remap, or change scratch pointers during CUDA graph capture.
- Treat graphs-OFF success as a complete fix.
- Treat runs with disabled draft/speculative context as valid proof.
- Silently disable MTP/NextN to pass tests.
- Use whole-pool VMM teardown as a quick OOM retry.
- Rely on `ggml_cuda_lock` to serialize FA scratch operations; it was not designed for that and does not bracket the right window.
- Re-hold `backend_cfg_mu` across `ggml_backend_sched_graph_compute_async` to "fix" the race; this serializes at the wrong layer and contradicts Atomic's documented design for MTP overlap.
- Key scratch only by `(device, stream)` while leaving the host-thread race open. Sharing scratch by stream slot alone is safe only when §8 serialization is in place.
- Invalidate only the currently-computing graph on growth, leaving other entries in `cuda_graphs` (e.g., MTP's) bound to a stale captured pointer.
- Rely on member declaration order in `ggml_backend_cuda_context` to sequence graph teardown before scratch teardown; sequence must be made explicit in the destructor body via `cuda_graphs.clear()`.
- Copy-paste the `V_is_K_view` predicate between sizing and `launch_fattn()`; share one helper instead.
- Move the call site of `ggml_cuda_graph_update_required()`. V2 inserts sizing+reservation **before** the existing decision block; the existing function's call site is unchanged.
- Block on `cudaEventSynchronize` or `cudaDeviceSynchronize` inside §8's critical section to reclaim retired pointers. Reclaim is opportunistic via `cudaEventQuery` at the start of each compute window.
- Use the §3 invalidation predicate without per-graph slot tracking. The plan adds `fa_borrowed_streams` to `ggml_cuda_graph` precisely because `node_props` does not currently record the captured FA stream slot; an invalidation pass that ignores this state is unimplementable on the current code.
- Call the live `try_launch_concurrent_event` from the §3 step 3 preflight resolver. That mutates `concurrent_events`/`curr_stream_no`. §3 step 3 must use the side-effect-free variant extracted per §7.
- Use `CUDA_CHECK(cudaGraphExecDestroy / cudaGraphDestroy)` inside the §3 step 5 traversal. Iteration must use non-aborting raw `cudaError_t` returns so a single bad entry cannot kill the process mid-map.
- Place a `GGML_ASSERT(!(need_f16_K && K->type != GGML_TYPE_F16))` (or any variant) anywhere in the patch. The VEC F32 dispatch makes that expression legitimately true; the assertion would fire on valid workloads. Earlier wording about a "VEC invariant assertion" has been retracted.
- Use the `BEST_FATTN_KERNEL_VEC` enum value to short-circuit sizing to zero. VEC contributes nonzero scratch when `K->type == GGML_TYPE_F32` (the macro at `fattn.cu:225` accepts F32 routed through the F16 template).

### Validation Requirements

Before the next matrix:

- Fix harness paths if `/mnt/CT500MX500SSD1` is absent; current artifacts are visible under `/mnt/MX500/grimoire-ab/`.
- Add payload token counting to `repro.py` or a companion artifact.
- Record current payload facts; visible file is 22,312 bytes, while prior Python harness results reported 22,081 characters.
- **Declare the §8 serialization regression threshold up front** (e.g., ≤ X % decode-tokens/sec drop on the second large turn vs. the unpatched graphs-OFF baseline on the same payload). A measured regression beyond the threshold escalates §8 to per-generation banks. The threshold must be recorded before measurement begins, not after seeing numbers.
- Build the debug-only failure-injection hook (e.g., env var `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` causes the next reserve/grow to return `cudaErrorMemoryAllocation`). Locate the gate inside `common.cuh` so the test surface lives entirely in V2 — no LD_PRELOAD shim required.
- Capture cache-bust evidence: snapshot `/app/.cache/llama-cpp-build/.atomic_patch_hash` before and after patch swap, and run `nm libggml-cuda.so | grep <new-scratch-symbol>` to prove the patched objects are linked.

Build and test at least:

| Variant | CUDA graphs | ctx-size |
| --- | --- | ---: |
| unpatched baseline | ON | 180000 |
| unpatched baseline | ON | 210000 |
| unpatched baseline | OFF | 180000 |
| unpatched baseline | OFF | 210000 |
| V2 patched | ON | 180000 |
| V2 patched | ON | 210000 |
| V2 patched | OFF | 180000 |
| V2 patched | OFF | 210000 |

Each run must record:

- model startup logs;
- positive evidence that draft/NextN/MTP context initialized and is active;
- first-turn tool-call result;
- second-turn large tool-result streaming result;
- exact tool payload character count and token count;
- `/` and SSD free space snapshots;
- GPU memory before load, after load, after turn 1, during large turn 2, and after idle if feasible;
- decode and prompt throughput (tokens/sec) for the second large turn — at minimum one patched run and one comparable unpatched run must report this so the §8 serialization cost is quantified against the pre-declared threshold;
- `cuda_graphs.size()` growth curve: snapshot at startup, after first decode, after the second large turn, and after N idle minutes. A monotonic climb across many decode cycles indicates leaked graph instances and weakens the retired-scratch lifetime story;
- retired-list size and total retained bytes at the same snapshot points. Steady-state must be bounded; under prolonged idle (no decode running), retired pointers may linger because §3 step 1 only fires at the start of the next compute — verify that the bound is the size set after the most recent growth, not unbounded;
- `pools[device][stream]` size snapshot at the same points. V2's success criterion is K/V f16 scratch leaving the pool; a V2 that succeeds but still drives `pools[][]` growth from `KV_max` / `dst_tmp` / `dst_tmp_meta` is the *Candidate B Boundary* handoff, not a clean K/V scratch pass;
- a scratch-borrow trace at every FA enqueue point (`fattn_scratch[d][s].k` pointer, `fattn_scratch[d][s].v` pointer, thread id). A race regression then surfaces as two threads borrowing different pointers for the same slot inside one lock window;
- an acquire/release trace on `fattn_compute_mu` showing the interleaving of target and MTP threads;
- lock-hold duration breakdown for `fattn_compute_mu`: (a) retired-list reclaim, (b) §3 step 2 sweep, (c) reservation/growth/invalidation, (d) existing decision block, (e) capture body, (f) instantiate/launch. Total throughput alone is insufficient; the breakdown tells a future optimizer which component dominates the serialization cost.

Validity rule:

- A run is invalid for proving MTP/NextN if logs contain `failed to create draft context`.
- A run is invalid for proving MTP/NextN if logs contain `speculative decoding context not initialized`.

Classify validation outcomes explicitly:

1. V2 K/V scratch pass:

   - K/V f16 scratch no longer enters the VMM pool;
   - no `cudaMalloc`, `cudaFree`, `cudaStreamSynchronize`, unmap, remap, or pointer-changing growth occurs from `launch_fattn()`;
   - CUDA graph capture/replay remains safe;
   - scratch reservation failure returns a recoverable backend status, preferably `GGML_STATUS_ALLOC_FAILED`;
   - no VMM `cuMemCreate` abort occurs in the implicated FA K/V scratch path;
   - draft/NextN context is initialized and active.

2. Candidate B block:

   - a residual abort is attributed by stack/site to `KV_max`, `dst_tmp`, `dst_tmp_meta`, or another non-K/V pool allocation;
   - logs prove K/V f16 scratch no longer enters VMM;
   - graph capture/replay remains safe.

   This is not a V2 K/V scratch design failure, but it is not an end-to-end production pass. Route it to Candidate B allocator robustness work.

3. End-to-end production pass:

   - no HTTP 502/409;
   - no process abort;
   - no capture-time CUDA allocation/synchronization error;
   - no allocation/free/sync path from `launch_fattn()` during graph capture;
   - stream completes with `[DONE]`;
   - draft/NextN context is initialized and active;
   - §8 serialization cost falls within the pre-declared threshold;
   - `cuda_graphs.size()` and the retired-list size both converge under long-soak.

Any remaining process abort must be attributed by stack/site before assigning the verdict.

Add targeted validation cases:

- capture with smaller FA scratch, then run a larger shape and prove the old graph executable is destroyed and recaptured before any replay; the §3 step 5 traversal must be observable in logs (graph key + invalidation event + the `fa_borrowed_streams` set that triggered the match).
- run with `GGML_CUDA_GRAPH_OPT=1` and log the FA borrow stream slot; on Grimoire's dual-GPU host this must run with one visible CUDA device (`CUDA_VISIBLE_DEVICES=<one GPU>`), because Atomic graph optimization returns early unless `ggml_backend_cuda_get_device_count() == 1`.
- run with `GGML_CUDA_GRAPH_OPT=1` where a concurrent event is rejected or cleared and FA falls back to stream 0; this also needs the single-visible-GPU condition above.
- force scratch reservation failure via `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` and prove it returns `GGML_STATUS_ALLOC_FAILED` without process abort, without entering capture/replay, without advancing graph warmup or property state, and without leaving a partial K-only allocation. A subsequent normal decode after the forced failure must reach a clean reservable state.
- **MTP/target concurrent stress.** During the second large-tool-result turn, the MTP worker must be in flight on `sched_mtp` while the target submits on `sched`. The run must show MTP `graph_compute_async` succeeding while V2 scratch is active for the target, with: no scratch overwrite; no `operation not permitted when stream is capturing` error; no graph replay using a stale scratch pointer (verifiable via the §3 step 5 traversal log plus the scratch-borrow trace); no abort. The `fattn_compute_mu` acquire/release trace must contain at least one instance where thread A acquires the lock *after* thread B acquired and released it within the same decode generation — purely sequential execution can fake interleaving and does not satisfy this criterion.
- **Serialization cost.** Compare decode/prompt throughput between V2-patched (with §8 lock) and the closest succeeding unpatched baseline on a payload large enough to drive MTP traffic. Record the delta against the pre-declared threshold.
- **Long-soak.** Run ≥ 1000 decode cycles with MTP active. Assert that `cuda_graphs.size()` and the retired-pointer list both stabilize. Linear growth in either signals a leaked invalidation case or a missing reclaim — V2 cannot stabilize on a long-running server otherwise.
- **Idle-after-growth.** Trigger a growth event, then idle the server for ≥ 5 minutes. Verify the retired list remains bounded (entries may linger until the next compute drains them) and that no descheduled `cudaEventQuery` polling is happening. Destructor-time drain is the safety net for shutdown; this case covers steady idle.
- **Removed-call-site grep.** Build with the patch and confirm via `grep -n "ggml_cuda_pool_alloc<half> K_f16\|ggml_cuda_pool_alloc<half> V_f16" .../fattn-common.cuh` that no pool allocation for K_f16/V_f16 remains on the CUDA path. Also grep the whole `ggml/src/ggml-cuda/` tree for any new `pool_alloc<half>` for FA scratch — a future rebase could reintroduce one in a different file. A partial merge that leaves one behind would not be caught by other validation cases.
- **F32 K/V VEC dispatch case.** Add at least one validation run whose payload routes FA through VEC dispatch with `K->type == GGML_TYPE_F32` (and a separate one for `V->type == GGML_TYPE_F32`). This exercises the corrected §2 VEC sizing rule (which the earlier "VEC contributes zero" claim got wrong) and proves the borrower's `capacity >= request` assertion does not fire on the F32 path. The existing matrix only varies ctx-size and graphs on/off — it does not naturally exercise F32 KV.
- **Sizing-vs-actual drift assertion.** Debug builds should record, at every FA execution, the actual bytes `launch_fattn` allocates for K_f16/V_f16 and the preflight helper's prediction for the same node, and assert `helper_pred >= actual`. A regression that under-sizes (e.g., a new kernel enum added to the selector but not to the sizing helper) then surfaces as a debug assert rather than a silent under-allocation.
- **Failure-injection coverage.** Exercise `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` at three points: (a) first reserve (cold start, no prior scratch); (b) growth on an existing slot (warm, scratch present but undersized for the new shape); (c) growth attempted while another thread holds the §8 lock mid-capture. Only (a) is naturally exercised by a basic forced-failure test.
- **Target-in-flight while MTP grows.** Validation case that pairs an in-flight target replay (`cudaGraphLaunch` has returned to host but the GPU is still executing the launched graph) with an MTP shape change that triggers growth on the same `(d, s)`. The §8 lock serializes the host paths, but the GPU drain depends on the §3 step 5 event set recorded on **all populated launch streams** plus any later-created slot's first-use event. Log the launch stream identity at every `cudaGraphLaunch` and the event-stream identities at every retired-entry creation. **Pass criterion (hard):** for every retired entry, the test harness must assert `event_stream_ids ⊇ {launch_stream_ids of every cudaGraphLaunch observed between the entry's growth time and its reclaim point}`. A regression that records events only on slot `s` would log fine but fail this assertion.
- **Map-iteration safety during step 5.** Force one `cudaGraphExecDestroy` call to return a non-success `cudaError_t` (e.g., via the failure-injection hook scoped to graph-destroy) during a step 5 traversal. Confirm the loop continues, all other entries are processed, the affected entry is logged, and no entry remains where `instance != nullptr && fa_borrowed_streams contains the grown slot`. The map must not be partially invalidated. Note that the C++17 `unordered_map` only guarantees iterator stability across non-rehashing erases; the step 5 loop mutates fields of mapped values but does not erase keys, so iteration is safe. Do not "simplify" the loop to erase nullptr-instance entries — that would invalidate iterators mid-traversal.
- **Per-thread lock-hold breakdown.** Report the §8 lock-hold breakdown separately for the target thread vs. the MTP worker thread. Target's cgraph and MTP's draft cgraph have very different node counts; the asymmetry is exactly what motivates §8's serialization-cost measurement.
- **Step 5 invariant post-traversal.** After step 5 returns, every entry in `cuda_graphs` must satisfy the one-way implication `(fa_borrowed_streams.size() > 0) ⇒ (instance != nullptr)`. The reverse is not required: a captured graph with no FA nodes legitimately has `instance != nullptr` and `fa_borrowed_streams.empty()`, and a freshly-inserted entry that has not yet captured has both empty/null. A debug-build assertion that walks the map and checks this one-way implication catches future error paths that destroy `instance` outside step 5 while leaving the set populated.
- **`cuda_graphs` reader grep.** After build, run `grep -rn "cuda_graphs\|any_cuda_graph_enabled\|any_cuda_graph_has_instance" .../ggml/src/ggml-cuda/` and confirm every call site is on the §8-locked path (inside `graph_compute`, `graph_optimize`, `evaluate_and_capture`, or the destructor under its pre-condition). The two `mean.cu` sites (`~37, 38, 41, 42`) must be on that list because they run inside `evaluate_and_capture`.
- **`GGML_CUDA_DISABLE_GRAPHS=1` row in matrix.** Add a validation row that disables CUDA graphs at runtime (no capture, FA always direct-execute). Confirms the non-capture path of §1/§3/§4/§6 builds and runs correctly; this is the production-relevant fallback when graphs are disabled by the operator.
- **"Decode generation" definition for traces.** The MTP/target concurrent-stress validation case asks for an out-of-order acquire on `fattn_compute_mu` "within the same decode generation". Define generation as **one `llama_decode` invocation by libllama** — not one `graph_compute_async` call (a single decode produces multiple `graph_compute` calls due to splits and MTP). The trace inspector groups acquires by decode start/end markers.
- **§7 dormant-on-dual-GPU positive test.** Run the matrix on the dual-GPU production config (no `CUDA_VISIBLE_DEVICES` override) with `GGML_CUDA_GRAPH_OPT=1` and assert that the §7 nonzero-stream code paths are never entered (the early-exit at `:4322` fires). This is the positive evidence that §7 stays dormant in production; the existing single-visible-GPU graph-opt case is the active test.
- **Built-object grep for residual pool allocs.** Run `nm libggml-cuda.so | c++filt | grep ggml_cuda_pool_alloc` restricted to the FA translation unit's symbols. Source grep only checks file text; a re-instantiation of the template for `half` via a different `#include` path would not be caught by source grep alone.
- **Direct-execute path skips `fa_borrowed_streams`.** When `properties_changed == true` at `:4246-4249` resets warmup and `use_cuda_graph` becomes false, the §4 borrower must skip recording. Confirm via the scratch-borrow trace that no entry is added to any `fa_borrowed_streams` during this compute.
- **Production reach of §7.** The single-visible-GPU constraint on `GGML_CUDA_GRAPH_OPT=1` means production (2 GPUs) never exercises §7's nonzero-stream code path. Validation must record §7 as "test-validated with `CUDA_VISIBLE_DEVICES=<one GPU>`", not as "production-validated".

The existing matrix remains invalid for full NextN proof because all startup runs that reached model load contained draft-context failure lines.
