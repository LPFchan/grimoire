# Atomic llama.cpp Patches

Patches in this directory apply to `AtomicBot-ai/atomic-llama-cpp-turboquant` after the Docker build verifies `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (`0a635dcd92ba66c75fccfef91c3e106f4668f367`).

Keep each patch scoped to one runtime issue and update the patch if the pinned SHA changes.

## Patches

| File | State | Scope | Notes |
| --- | --- | --- | --- |
| `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | **served** | V1 fix: replaces the CUDA FA K/V f16 dequant pool allocation with raw `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize` in `launch_fattn()`. Requires `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=OFF` because alloc/free/sync inside a captured graph is unsafe. | Currently wired into `Dockerfile` and applied on every build. |
| `0002-cuda-fa-v2-scratch-owner.patch` | **unwired** | V2 fix per `records/research/RSH-20260523-001-cuda-vmm-pool-oom-agentic-crash.md` §Patch V2 Authoritative Plan. Adds a per-context FA scratch owner with stable K/V pointers, `fattn_compute_mu` mutex, retired-pointer reclaim, `cuda_graphs` invalidation on growth, stream-slot predictor, recoverable allocation failure, and a debug failure-injection hook. Graph-safe — allows `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON`. | Conflicts with `0001` (both edit `launch_fattn`). Flip to V2 by editing `Dockerfile`: replace the `0001-...` filename with `0002-...`. Validation matrix lives in the RSH. |

## Flipping to V2

V2 supersedes V1. The two patches both edit `launch_fattn()` and cannot apply together — pick one.

1. Edit `Dockerfile`: change `patches/atomic-llama-cpp/0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` to `patches/atomic-llama-cpp/0002-cuda-fa-v2-scratch-owner.patch` in both the `sha256sum` and `git apply` commands.
2. Set `GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON` in the same file once V2 is validated.
3. Rebuild. The Dockerfile's `.atomic_build_config` cache key includes the patch hash, so swapping patches busts the build cache automatically.

Do not flip until the validation matrix in the RSH (`§Validation Requirements`, eight rows + targeted cases) is green.
