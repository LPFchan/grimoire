# Atomic llama.cpp Patches

Patches in this directory apply to `AtomicBot-ai/atomic-llama-cpp-turboquant` after the Docker build verifies `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (`0a635dcd92ba66c75fccfef91c3e106f4668f367`).

`GRIMOIRE_LLAMA_CPP_PATCH_FILE` is a **comma-separated list** of patch filenames in this directory, applied in order. Patch hashes are part of the build cache key.

## Currently Served (`grimoire:local` default)

| File | Scope | Origin |
| --- | --- | --- |
| `0002-cuda-fa-v2-scratch-owner.patch` | V2 graph-safe FA scratch owner with recoverable failure | private, RSH-20260523-001 |
| `0004-pool-flush-on-oom.patch` | Legacy pool flush-and-retry on `cudaErrorMemoryAllocation` (backport of upstream PR #22155) | `ggml-org/llama.cpp` PR #22155, merge commit `97895129e5f2bde94d13dc01ca41ee79e9b629f2` |

Dockerfile default: `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON`.

The decision to ship this chain with graphs ON is documented in `records/research/RSH-20260526-003-v2-on-shipping-decision.md`. Validation evidence is `RSH-20260526-001`. V3 design (deferred) is `RSH-20260526-002`.

## Available But Not Shipped

| File | Status |
| --- | --- |
| `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | V1, kept for rollback / repro. Superseded by V2. |
| `0003-cuda-fa-view_src-sizing.patch` | Backport of closed upstream PR #23620. Tested and rejected for our boundary case — see RSH-20260526-003 §"The View_src False Positive". Kept for slow-VRAM-creep mitigation if that regime ever fires. |
| `0005-peft-trainable-token-replacements.patch` | Native PEFT TrainableTokens replacement rows for compact Gemma adapter serving. Tested with `eastself-v1-lora-ck6500-tokenrep.gguf`; not default-shipped yet. |

## Build Examples

```bash
# Default (V2+0004, graphs ON)
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
