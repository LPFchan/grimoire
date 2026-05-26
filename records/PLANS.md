# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## Atomic CUDA FA V2 (active)

Source of truth: `records/research/RSH-20260523-001-cuda-vmm-pool-oom-agentic-crash.md` §Patch V2 Authoritative Plan. Target: `AtomicBot-ai/atomic-llama-cpp-turboquant` @ `0a635dcd92ba66c75fccfef91c3e106f4668f367`. Goal: remove K/V f16 FA scratch from the VMM pool in a way that is safe under CUDA graph capture/replay and under concurrent target/MTP host-thread submission, so `GGML_CUDA_GRAPHS=ON` can be re-enabled and MTP/NextN can stay active without VMM `cuMemCreate` OOM aborts.

Phases (dependency order; each closes with a `LOG-*` commit):

| Phase | Scope | Sections | Files |
| --- | --- | --- | --- |
| V2-1 | Foundational state + lock | §1, §8 | `common.cuh` |
| V2-2 | Sizing helper | §2 | `fattn.cuh`, `fattn.cu`, `fattn-common.cuh` |
| V2-3 | Destructor explicit teardown + retired-list drain | §6, §3 step 1 | `ggml-cuda.cu`, `common.cuh` |
| V2-4 | Compute path: reservation, borrower, recoverable failure, step 5 invalidation | §3, §4, §5 | `ggml-cuda.cu`, `fattn-common.cuh` |
| V2-5 | Stream-slot predictor + `graph_optimize` lock | §7, §8 | `ggml-cuda.cu` |
| V2-6 | Debug instrumentation: failure-injection hook, sizing-vs-actual drift assert, predictor drift assert | Validation Requirements | `common.cuh`, `fattn.cu`, `ggml-cuda.cu` |

**Phases V2-1 through V2-6 complete 2026-05-25.** Patch lives at `patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch`. Image `grimoire:ab-v2-graphs-on` built with `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON` and smoke-validated end-to-end at qwen3.6-mtp-27B ctx=180k with the original crash payload (three back-to-back agentic two-turn repros, 22,081-char tool result, all `ok=true`, NextN 67–91 % acceptance). Predictor-drift assert inside `evaluate_and_capture` deferred to follow-up.

**Open follow-ups** (none are gating; load-bearing case is green):

| Item | Notes |
| --- | --- |
| V2 graphs-OFF build + smoke | Confirms V2 also works when graphs are disabled. Single image build with `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=OFF`. |
| V0/V1 baseline rows in the matrix | The 8-row matrix in the RSH expects unpatched-on/off and V1-on baselines for throughput comparison. Today only V1-OFF (i.e. `grimoire:local`) is built. V0 (`APPLY_PATCHES=0`) and V1-ON would mainly reproduce the original crash. |
| 210k-ctx row of the matrix | Won't fit single-GPU on the dual-3090 host; defer or run with `--n-cpu-moe` offload. |
| Long-soak (≥ 1000 decode cycles), idle-after-growth, forced-failure-injection, F32 K/V VEC dispatch | Listed in the RSH §Validation Requirements as targeted cases. Not yet run. |
| Flip prod (`grimoire:local`) to V2 | Requires building `grimoire:local` with `GRIMOIRE_LLAMA_CPP_PATCH_FILE=0002-...` and graphs ON, then `compose up -d --force-recreate`. Pending operator approval. |

## Backlog (Future Interest)

| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| 1 | **DDTree tree-mode verify** — enable `--spec-branch-budget > 0` with larger `GGML_DFLASH_MAX_VERIFY_TOKENS` | Benchmarked (2026-05-18): tree-mode (budget=22) = 62.4 tok/s vs flat = 61.6 tok/s (+1%). The draft is already well-matched to the target for greedy decode (85% acceptance), so tree-mode adds negligible benefit. Deferred unless non-greedy sampling or a lower-quality draft changes the trade-off. | The 25-token cap fix is already deployed. Env var auto-derived by model_manager.py. |
| 2 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
- VMM park/unpark — measured (RSH-20260518-001): adds 2.5 GB VRAM overhead with no TTFT benefit for single-model. Unnecessary unless multi-model GPU sharing is needed later.
