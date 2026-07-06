# Atomic llama.cpp Patches

Patches in this directory apply to `AtomicBot-ai/atomic-llama-cpp-turboquant` after the Docker build verifies `GRIMOIRE_LLAMA_CPP_PINNED_SHA`.

`GRIMOIRE_LLAMA_CPP_PATCH_FILE` is a comma-separated list of patch filenames in this directory, applied in order.

## Currently Served (`grimoire:local` default)
| File | Scope | Origin |
| --- | --- | --- |
| `0005-peft-trainable-token-replacements.patch` | Native PEFT TrainableTokens replacement rows for compact Gemma adapters. | private, B-prime ck9000 |
| `0006-mtmd-gemma4v-sequential-images.patch` | Gemma4V multi-image mtmd crash fix by avoiding unsafe projector batching and sequentially encoding image chunks. | local |

Dockerfile default: `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON`.

## Available But Not Shipped
| File | Status |
| --- | --- |
| `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | V1, kept for rollback / repro. Superseded by V2. |
| `0002-cuda-fa-v2-scratch-owner.patch` | V2 graph-safe FA scratch owner. Stale for the current pinned SHA; keep only for rebase/reference. |
| `0003-cuda-fa-view_src-sizing.patch` | Backport of closed upstream PR #23620. Tested and rejected for our boundary. |
| `0004-pool-flush-on-oom.patch` | Legacy pool flush-and-retry on CUDA OOM. Stale for the current pinned SHA; keep only for rebase/reference. |

## Build Examples
```bash
# Default (PEFT token replacements + Gemma4V mtmd multi-image fix, graphs ON)
docker build -t grimoire:local .

# Rollback to V1+OFF
docker build -t grimoire:ab-v1-graphs-off \
    --build-arg GRIMOIRE_LLAMA_CPP_PATCH_FILE=0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch \
    --build-arg GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=OFF .

# V0 (no patches) for crash repro
docker build -t grimoire:ab-unpatched-graphs-on \
    --build-arg GRIMOIRE_LLAMA_CPP_APPLY_PATCHES=0 \
    --build-arg GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON .
```
