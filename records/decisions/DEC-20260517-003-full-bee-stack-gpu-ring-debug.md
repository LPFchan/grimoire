# Adopt Bee Stack, Debug GPU Ring + Turbo4 Bug

Opened: 2026-05-17 18-30-00 KST
Recorded by agent: opencode

## Decision

Adopt Bee's `llama-server` binary (`Anbeeld/beellama.cpp`) as the single runtime for both the native DFlash canary and the baseline production stack. TheTom's binary is retired as the served runtime. The single remaining blocker is: GPU cross ring buffer hangs when DFlash is used alongside `--cache-type-k turbo4`. The CPU fallback (`GGML_DFLASH_GPU_RING=0`) works but makes DFlash 2-3x slower, defeating its purpose.

## Context

Phase 1 ported Bee's DFlash pipeline onto TheTom (ring buffer, graph builder, flush_prefill). TheTom achieves 0% draft acceptance. Bee's binary achieves 100% with the same GGUF. Investigation revealed Bee already has turbo4 KV cache support built-in — no porting needed.

However, DFlash + turbo4 together hang during prompt processing when the GPU cross ring buffer is active. The GPU ring performs D2D copies of hidden states and runs interleave kernels on the default CUDA stream. With turbo4 KV cache, these operations conflict, causing a GPU-side deadlock. `GGML_DFLASH_GPU_RING=0` avoids the bug by using CPU copies, but adds 15-50ms per draft cycle (2-3x slower draft).

The purpose of DFlash is faster decode. A 2-3x slower draft path negates the speculative decoding benefit.

## Options Considered

| Option | Outcome |
| --- | --- |
| Full Bee stack, fix GPU ring bug (chosen) | DFlash at full speed (draft ~7ms). Single binary for everything. |
| Full Bee stack, accept CPU ring slowdown | DFlash 2-3x slower, marginal benefit over autoregressive. |
| Split deployment: TheTom for turbo4 models, Bee for DFlash | Two binaries, operational complexity, TheTom DFlash port is broken anyway. |

## Rationale

- Bee stack has everything: turbo4, DFlash (100% acceptance), TCQ, GPU ring
- The GPU ring bug is a CUDA stream conflict, not a fundamental architecture issue
- Fixing it unlocks full DFlash speed: draft ~7ms vs ~22-58ms
- Single binary simplifies deployment and testing
- See Dec issue #16 on `Anbeeld/beellama.cpp` — user reported similar GPU ring crash at long context; `GGML_DFLASH_GPU_RING=0` was their workaround too

## Consequences

- Production `grimoire` container switches from TheTom to Bee binary
- Bee binary at `/tmp/spec-analysis/bee-shallow/build/bin/llama-server`
- Models stay at `/home/yeowool/models/gguf/` — no file changes needed
- GPU ring bug tracked as remaining blocker before full-speed DFlash is live
