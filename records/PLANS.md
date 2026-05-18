# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## Deferred Items

Ordered by expected impact vs effort:

| # | Item | Why Deferred | Prerequisite |
|---|------|-------------|--------------|
| 1 | **PINNED_SHA enforcement** — verify cloned SHA at Docker build time | Shell escaping in `RUN` didn't work; reverted to unblock build | None — just needs correct escaping (`grep`-based check instead of shell variable) |
| 2 | **VRAM drift soak** — `test_pflash_pipeline.py` against live server, measure VRAM across repeated compression cycles | PFlash models verified manually but no systematic soak | None |
| 3 | **Real 20K sysprompt canary test** — verify KV cache speedup at production scale | Mechanism proven at 18 tokens (1.93x); same at any scale | None |
| 4 | **Monitoring for 413 path** — Prometheus counter or structured log for pflash compression failures | PFlash 413 is new; no observability yet | None |
| 5 | **Multi-spec batched decode** — batch N slots through one speculative decode | Single-spec sufficient for current single-user workload | None |
| 6 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Item 5 (multi-spec provides the batch context for tree-mode) |
| 7 | **VMM park/unpark** — CUDA VMM for PFlash drafter slot | SIGTERM fallback works; VMM needs isolated measurement | None |
