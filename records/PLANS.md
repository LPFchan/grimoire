# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## CUDA Flash-Attention Scratch (deployed baseline)

The May Atomic V2 work was retired when Grimoire migrated to the current TheTom engine fork. The current build ports scoped CUDA flash-attention K/V scratch allocation to the pinned engine and keeps CUDA graphs off. Qwen3.8-27B reasoning aliases use a 190,000-token context, a physical prompt batch of 128, the BF16 vision projector on GPU, the MTP head on GPU, and symmetric turbo4 K/V. This configuration preserves fast GPU image preprocessing while retaining tested long-context headroom on one RTX 3090.

The incident audit and OpenCode soak are recorded in `records/research/RSH-20260828-001-qwen38-long-context-crash.md`. The GPU-projector boundary tests are recorded in `records/research/RSH-20260829-001-qwen38-gpu-mmproj-190k.md`.

A graph-safe reusable scratch reservation remains optional future work. Revisit it only if CUDA-graph throughput becomes more valuable than the currently verified long-agent-context headroom.

## Backlog (Future Interest)

| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| 1 | **DDTree tree-mode verify** — enable `--spec-branch-budget > 0` with larger `GGML_DFLASH_MAX_VERIFY_TOKENS` | Benchmarked (2026-05-18): tree-mode (budget=22) = 62.4 tok/s vs flat = 61.6 tok/s (+1%). The draft is already well-matched to the target for greedy decode (85% acceptance), so tree-mode adds negligible benefit. Deferred unless non-greedy sampling or a lower-quality draft changes the trade-off. | The 25-token cap fix is already deployed. Env var auto-derived by model_manager.py. |
| 2 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
- VMM park/unpark — measured (RSH-20260518-001): adds 2.5 GB VRAM overhead with no TTFT benefit for single-model. Unnecessary unless multi-model GPU sharing is needed later.
