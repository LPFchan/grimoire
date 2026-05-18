# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## Backlog (Future Interest)

| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| 1 | **DDTree tree-mode verify** — enable `--spec-branch-budget > 0` with larger `GGML_DFLASH_MAX_VERIFY_TOKENS` | Benchmarked (2026-05-18): tree-mode (budget=22) = 62.4 tok/s vs flat = 61.6 tok/s (+1%). The draft is already well-matched to the target for greedy decode (85% acceptance), so tree-mode adds negligible benefit. Deferred unless non-greedy sampling or a lower-quality draft changes the trade-off. | The 25-token cap fix is already deployed. Env var auto-derived by model_manager.py. |
| 2 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
- VMM park/unpark — measured (RSH-20260518-005): adds 2.5 GB VRAM overhead with no TTFT benefit for single-model. Unnecessary unless multi-model GPU sharing is needed later.
