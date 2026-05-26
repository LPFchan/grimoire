# RSH-20260526-003: V2+ON Shipping Decision — Bench Data, Upstream Survey, and the View_src False Positive

Opened: 2026-05-26 KST
Recorded by agent: claude-opus-4-7

## Question

Given V2 was matrix-validated for safety (RSH-20260526-001) but its throughput value over V1+OFF wasn't established and its boundary-case regression vs V0+OFF was, **does V2+ON deserve to be the served default?** If yes, what's the patch chain, and what alternatives did we evaluate before committing?

## Decision (TL;DR)

Yes. Ship `0002-cuda-fa-v2-scratch-owner.patch` + `0004-pool-flush-on-oom.patch` (the latter a backport of upstream PR #22155) with `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON`. Defer V3 (RSH-20260526-002). Don't ship `0003-view_src` (the upstream PR #23620 fix) — it doesn't help our specific crash and actively hurts our boundary case. Don't comment on upstream issue #23446 — the draft made claims about view_src that the test results refuted.

Evidence below.

## Decode Bench: V2+ON vs V2+OFF

Methodology: short prompt (89 tokens, a detailed technical writing instruction), long generation (2048-token cap), 5 iterations per variant, NextN/MTP draft active. Same hardware (GPU 1, RTX 3090, VBIOS `94.02.59.00.42`), same `n-gpu-layers=999`, same q8_0 KV, same temperature 0.6. Bench script at `/tmp/bench-decode.sh` (qwen) and `/tmp/bench-decode-gemma.sh` (gemma); raw JSON under `/mnt/MX500/grimoire-ab/results/bench-decode-20260526-194011/` and `bench-gemma-20260526-194936/`.

### qwen3.6-mtp-27B at ctx=60000

| Iteration | V2+ON decode tok/s | V2+OFF decode tok/s | Per-iter delta |
| --- | --- | --- | --- |
| 1 | 54.92 | 52.62 | +4.4 % |
| 2 | 55.28 | 53.23 | +3.9 % |
| 3 | 56.20 | 51.91 | +8.3 % |
| 4 | 55.31 | 53.26 | +3.9 % |
| 5 | 54.46 | 52.98 | +2.8 % |
| **Median** | **55.28** | **52.98** | **+4.34 %** |

Draft acceptance both variants: 74–81 %, parity.
Prompt processing: 349.91 vs 346.96 tok/s, +0.85 % (noise).

### gemma-4-mtp-31B at ctx=60000

| Iteration | V2+ON decode tok/s | V2+OFF decode tok/s | Per-iter delta |
| --- | --- | --- | --- |
| 1 | 44.49 | 42.44 | +4.8 % |
| 2 | 44.04 | 42.78 | +2.9 % |
| 3 | 43.89 | 42.38 | +3.6 % |
| 4 | 43.25 | 42.57 | +1.6 % |
| 5 | 43.66 | 42.45 | +2.9 % |
| **Median** | **43.89** | **42.45** | **+3.40 %** |

Draft acceptance: V2+ON 82-83 % vs V2+OFF 81-82 % (graphs ON nudges draft acceptance up by ~1 pp, likely from tighter draft+verify pipeline batching).
Prompt processing: 59.96 vs 60.61 tok/s, -1.06 % (noise).

### Analysis

Both served model families show a positive, consistent graphs-ON decode win in the **+3-5 %** band. Sub-upstream-reported (community claims 20-40 % on pure decode), explained by:
- 27B / 31B class models have large kernels — kernel-launch overhead is a small fraction of per-token cost.
- Speculative decoding (NextN for qwen3.6, MTP for gemma-4) adds per-token GPU work that runs inside the captured graph but doesn't shrink under graph capture.
- q8_0 KV dequant inside `launch_fattn` is compute-heavy.
- Atomic's turboquant kernels add additional GPU compute time.

The +3-5 % is the realistic ceiling for grimoire's hardware/model class. It's modest but real (not noise — 5/5 iterations favor ON over OFF on each model).

## Upstream Survey

Probed `ggml-org/llama.cpp` for prior art on the FA scratch / VMM pool failure class. Key artifacts:

| Artifact | State | Relevance |
| --- | --- | --- |
| Issue [#23446](https://github.com/ggml-org/llama.cpp/issues/23446) "llama-server VRAM gradually increasing each run until OOM" | **OPEN** since 2026-05-20, 5+ confirmations | Same symptom as RSH-20260523-001. Multi-user repro on RTX 2080Ti / 3090 / 5090 / Vulkan. |
| PR [#23620](https://github.com/ggml-org/llama.cpp/pull/23620) "optimize KV cache dequant workspace to eliminate VRAM growth in flash attention" | **CLOSED, not merged** 2026-05-24 | Two-line fix: size K_f16/V_f16 by `view_src` size instead of view's `n_kv`. Pool reuses one max-context-size buffer instead of growing-by-1. Closed by maintainer am17an as "Wrong fix." (no detailed reason) + AI-generated-content policy. 5 community testers confirmed it eliminates the slow VRAM creep symptom. |
| PR [#22155](https://github.com/ggml-org/llama.cpp/pull/22155) "ggml-cuda: flush legacy pool on OOM and retry" | **MERGED** 2026-04-20 (commit `97895129e5f2bde94d13dc01ca41ee79e9b629f2`) | Legacy pool only: on `cudaErrorMemoryAllocation`, flush cached buffers and retry once before aborting. **Confirmed absent from our pinned SHA** (`0a635dcd`, predates the merge); we ported as `0004-pool-flush-on-oom.patch`. |
| PR [#22193](https://github.com/ggml-org/llama.cpp/pull/22193) "cuda: add partial eviction on pool OOM" | **OPEN** since 2026-04-21 | Builds on #22155 — LRU eviction + `bool overallocate` flag for FA buffers. Adjacent to V3 Part B (RSH-20260526-002) but different API direction. Watching, not engaging. |
| PR [#21054](https://github.com/ggml-org/llama.cpp/pull/21054) "cuda: fall back to VEC attention when quantized K/V F16 scratch exceeds free VRAM" | **CLOSED, not merged** 2026-03-27 | Alternative approach — `cudaMemGetInfo` before MMA kernel selection, pick VEC (no scratch) when MMA wouldn't fit. Not adopted. |
| PR [#22094](https://github.com/ggml-org/llama.cpp/pull/22094) / [#22185](https://github.com/ggml-org/llama.cpp/pull/22185) "hip: bypass memory pool for FA f16 temp buffers" | **CLOSED, not merged** | The HIP variant of V1's pattern. The Atomic source already has the HIP version inline (`0757ff4ee`); we have the CUDA generalization in `0001-V1` (private). |
| Issue [#22032](https://github.com/ggml-org/llama.cpp/issues/22032) "Flash attention crash (MUL_MAT failed / cudaStreamSynchronize) on Pascal GPUs" | **OPEN** | Same V1+ON-style capture/sync abort, different hardware (Pascal P40). Confirms V1's failure mode is hardware-independent. |

### Maintainer Position

From am17an on #23446 (2026-05-24):

> Using the quantized KV-cache, it is expected the VRAM usage will grow as context size grows larger because of the q8→f16 de quantization inside the FA kernel. It should also happen without any spec decoding.

Upstream considers monotonic VRAM growth with quantized KV cache **by design**. Multiple fix attempts (#23620, #21054, #22094, #22185) closed without merge. **V2 is not upstreamable** as-is; ships as a private Atomic-side patch.

## The View_src False Positive

PR #23620's two-line `view_src` sizing fix was the cheapest possible fix (5 community confirmations). Tested locally as `0003-cuda-fa-view_src-sizing.patch` against the matrix-validated boundary case (210k+999+GPU 0).

Result: **same `cuMemCreate → ggml_abort → HTTP 502` crash, no improvement.** The fix made the crash happen sooner.

Reason: numerical analysis.

| Sizing strategy | K_f16 per FA call | K+V combined per call |
| --- | --- | --- |
| V0 view sizing at n_kv=6000 (V0 crash point) | 11.7 MB | 23.4 MB |
| V0 view sizing at n_kv=8500 (turn-2 end) | 16.6 MB | 33.2 MB |
| **view_src sizing** (full 210k buffer) | **410.2 MB** | **820.3 MB** |

At our boundary case, post-load free VRAM is ~87 MiB. view_src demands 820 MiB up front on the very first FA call. Instant fail.

This reveals **three distinct OOM regimes** in the FA scratch path:

| Regime | Trigger | View_src effect |
| --- | --- | --- |
| **Slow VRAM creep** | Max-ctx scratch fits, but the growing-by-1 allocation pattern strands old pool buffers → eventual OOM after N tokens | **Fixes it.** Pool reuses one max-sized buffer. This is what #23446 community testers hit and view_src eliminates. |
| **First-FA OOM** | Max-ctx scratch *doesn't* fit available VRAM | **Makes it worse.** Crashes on first FA call instead of growing toward the wall. |
| **Pool-growth-during-capture** | Graphs ON causes the pool to accumulate FA scratch beyond "one call's worth" | Doesn't help directly. V2's "FA scratch out of pool" solves this. |

Our boundary case is **regime 2+3** (V0+OFF works, V0+ON crashes). view_src is the wrong tool for our case. The maintainer's "wrong fix" comment was technically defensible — for some workloads.

**Decision:** keep `0003-view_src` in the patches directory for documentation / future slow-creep workloads, but **do not ship it** in the default chain. Don't reduce the upstream issue to "we shipped your fix and it worked" — that claim is false for our workload class.

## The Issue #23446 Comment

A draft was written at `/tmp/issue-23446-comment-draft.md` claiming view_src works for our setup. After the test refuted that claim, the draft would have to be substantially revised to "view_src fixes regime 1 but not regime 2; our case is regime 2; the bug has more than one regime."

That's accurate but invites a long thread. The maintainer position is firm ("by design"); engaging further is high-effort, low-payoff. **Decision: do not post.** The bug report is well-tracked already; our specific finding (VBIOS-reserved asymmetry, regime taxonomy) is captured here for our own use and in our private patches.

## What Ships

`patches/atomic-llama-cpp/` chain applied in order:

1. **`0002-cuda-fa-v2-scratch-owner.patch`** — private V2 patch (RSH-20260523-001 §Patch V2 Authoritative Plan). Graph-safe FA scratch ownership, recoverable `GGML_STATUS_ALLOC_FAILED`, per-context `fattn_compute_mu` mutex, retired-list reclaim, capture-time `cuda_graphs` invalidation. ~830 lines.
2. **`0004-pool-flush-on-oom.patch`** — backport of upstream PR #22155 (merged on master `97895129e5`, absent from our pinned SHA). Legacy pool flush-and-retry on `cudaErrorMemoryAllocation`. ~20 lines.

`Dockerfile` defaults updated:
- `GRIMOIRE_LLAMA_CPP_PATCH_FILE=0002-cuda-fa-v2-scratch-owner.patch,0004-pool-flush-on-oom.patch` (comma-separated, new format)
- `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON`

`Dockerfile` build logic extended to apply each patch in the comma-separated list in order, with patch hashes folded into `.atomic_build_config` so swapping busts the C++ build cache.

Available but not in the default chain:
- `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` (V1, kept for rollback)
- `0003-cuda-fa-view_src-sizing.patch` (view_src, kept for slow-creep mitigation use cases that may surface later)

## What Doesn't Ship and Why

| Candidate | Why not |
| --- | --- |
| V3 (RSH-20260526-002) | Implementation effort ~1-2 days; would unlock workloads at the V2+ON boundary that currently fail HTTP 500. Decode bench shows V3 wouldn't add to the +3-5 % throughput delta. Production workloads don't approach the boundary. Reopen if monitoring shows HTTP 500 from FA OOM becomes measurable. |
| Upstream submission of V2 | Maintainer position is "VRAM growth is by design". V2 cannot be upstreamed as-is. Ships as private Atomic-side patch. |
| `0003-view_src` in the default chain | Tested, doesn't fix our boundary, makes it worse. Useful for a different bug regime that we haven't observed in our prod traffic. |
| Comment on issue #23446 | Engagement is high-effort, low-payoff. Maintainer position firm. Our findings captured here. |

## Validation Followups (V2+ON Era)

| Item | Notes | Priority |
| --- | --- | --- |
| Live deployment | `docker compose up -d --build --force-recreate grimoire` after the commit lands. | next step |
| 24-hour HTTP 500 rate monitoring | Specifically for `GGML_STATUS_ALLOC_FAILED`-class failures from FA scratch path. Should be zero on normal production workloads; non-zero is a signal to reopen V3. | first day post-deploy |
| Long-soak (≥ 1000 decode cycles) at V2+ON | Confirm retired-list and `cuda_graphs.size()` stabilize. | week 1 |
| F32 K/V VEC dispatch case | Not exercised by qwen3.6/gemma-4 with q8_0 KV. Needs a model + cache config routing through `FATTN_VEC_CASE` with F32 → F16 template. | when a model that hits it lands |
| Forced failure injection coverage | `GGML_DEBUG_FORCE_FA_SCRATCH_FAIL=1` hook in V2 patch, three injection points (cold start, growth on existing slot, mid-capture) not yet exercised. | low priority — code path is well-reviewed |
| Predictor-drift assert inside `evaluate_and_capture` | Deferred from V2 implementation. Borrower's debug `capacity >= request` covers practical case. | low priority |

## Artifacts

- Decode bench raw data: `/mnt/MX500/grimoire-ab/results/bench-decode-20260526-194011/` (qwen), `/mnt/MX500/grimoire-ab/results/bench-gemma-20260526-194936/` (gemma)
- Bench scripts: `/tmp/bench-decode.sh`, `/tmp/bench-decode-gemma.sh`
- View_src test container logs + crash signature: same matrix infrastructure as RSH-20260526-001
- Bench configs: `/mnt/MX500/grimoire-ab/configs/models-bench-{60k,gemma-60k}.json`
- V3 design doc (deferred): `records/research/RSH-20260526-002-v3-adaptive-fa-scratch.md`
- V2 matrix validation: `records/research/RSH-20260526-001-v2-matrix-single-gpu.md`
- Original crash report: `records/research/RSH-20260523-001-cuda-vmm-pool-oom-agentic-crash.md`
