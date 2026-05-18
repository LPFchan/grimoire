# Plans

## Steady-State Architecture

Single Bee binary (`Anbeeld/beellama.cpp`) serves DFlash (`--spec-type dflash`), PFlash (via standalone `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse. Legacy `backend:dflash` daemon retired. See git history for the completed migration (Phases 1-7).

## Deferred Items

Ordered by expected impact vs effort:

| Priority | Item | Why Deferred | Prerequisite |
|----------|------|-------------|--------------|
| ~~1~~ | ~~**PINNED_SHA enforcement** — verify cloned SHA at Docker build time~~ | ~~Done — `c90d2a0`~~ | ~~None~~ |
| ~~2~~ | ~~**VRAM drift soak** — `soak_vram_drift.py` against live server, 10 iterations, 47K tokens each~~ | ~~Done — ZERO drift across all cycles~~ | ~~None~~ |
| 3 | **Real 20K sysprompt canary test** — verify KV cache speedup at production scale | Mechanism proven at 18 tokens (1.93x); same at any scale | None |
| 4 | **VMM park/unpark** — CUDA VMM for PFlash drafter slot | SIGTERM fallback works; VMM needs isolated measurement | None |
| 5 | **GPU tape recording** — tree-mode DDTree verify using `dflash_tape_*` | Only needed if single-spec throughput becomes a bottleneck | Multi-spec batch decode |

## Not Pursuing

- Multi-spec batched decode — single-spec sufficient for current workload
- Monitoring for 413 path — (user decision)
