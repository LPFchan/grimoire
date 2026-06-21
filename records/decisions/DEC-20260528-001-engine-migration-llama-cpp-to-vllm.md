# DEC-20260528-001: Engine Migration — llama.cpp → vLLM

Opened: 2026-05-28 13-50-50 KST
Recorded by agent: claude-code
Updated: 2026-05-30 — prototype progress, drivers, llmcompressor, model quantization

## Metadata

- Status: in-progress
- Deciders: operator
- Related ids: RSH-20260529-006-awq-quantization-for-vllm
  (supporting evidence is upstream vLLM release notes and the failure analysis of conv `4278816a-fff3-4b4c-89bc-d2a15b198bf6`)

## Decision

Commit to replacing llama.cpp / `atomic-llama-cpp-turboquant` as grimoire's inference engine with vLLM. The grimoire gateway, webui, history layer, and model registry remain. Only the per-backend engine adapter (`model_manager.py`, `proxy/llama.py`) is rebuilt against vLLM. The Atomic CUDA FA V2 patch chain is retired with the engine swap.

Prototyping starts in `/tmp/grimoire-vllm-prototype/` (ephemeral, off-prod). Production stays on llama.cpp on one GPU during the prototype window; the other GPU is freed for vLLM evaluation.

## Context

The trigger was the crash on conv `4278816a` (gemma-4-mtp-mmproj-31B, 64,537-token multimodal prompt). Two compounding observations made it the last straw rather than a one-off:

1. **Request-time OOM is structural in llama.cpp.** The pool allocator grows on demand, and per-request VRAM peak is shaped by KV fill, mmproj expansion, and batch shape — none of which are bounded at model load. Patch V2 (`0002-cuda-fa-v2-scratch-owner.patch`) and the legacy pool flush-and-retry (`0004-pool-flush-on-oom.patch`) convert some OOMs into recoverable errors but cannot give a stable `--ctx-size` tuning baseline. The user's framing: "if that baseline shifts PER request, it makes it very hard to tune."
2. **Patch chain is now chasing a moving target.** SCALE-op aborts that V2 doesn't cover (this incident), V3 already scoped and deferred for FA adaptive sizing, and the per-op long tail of `GGML_CUDA_CHECK → ggml_abort` paths together imply the patch surface keeps growing rather than closing.

Upstream vLLM has, as of May 2026, gained the features that previously kept grimoire on the atomic fork:

- Qwen3.6 NextN / MTP — native via `speculative-config: method: qwen3_next_mtp`
- Gemma 4 MTP with the `google/gemma-4-31B-it-assistant` head — PR #41745, merged 2026-05-08
- TurboQuant KV-cache quantization — PR #38280; vLLM team benchmark recommends FP8 by default, 4-bit-nc as the practical pressure variant
- Sleep Mode (Oct 2025, mature in 2026) — 18-200× faster model switches than subprocess kill/restart; preserves allocator and CUDA graphs

The webui submodule's API surface to the gateway is small (`/v1/chat/completions`, `/v1/models`, `/history`, `/models{,/load,/unload}`, `/props`, `/tools`, `/registry/upload`). `/props` already has a synthetic router-mode branch in `src/grimoire/routes/models.py:329`. The webui therefore needs no commits for the engine swap.

## Options Considered

### Stay on llama.cpp, extend the V2/0004 patch chain to cover SCALE and other non-FA pool consumers

- Upside: smaller scope per round; keeps turbo4 weight + KV compression (lowest VRAM footprint at Q4_K_M)
- Upside: preserves existing MTP/NextN/PFlash work in `Anbeeld/beellama.cpp`
- Downside: does not fix the structural problem (peak VRAM is request-shaped); each round of patches only converts aborts into clean errors, never into actual headroom
- Downside: the patch surface is open-ended — every op that hits a direct `cudaMalloc` is a candidate
- Downside: tuning `--ctx-size` remains trial-and-error against a per-request-shifting baseline

### Migrate to vLLM

- Upside: preallocation contract — engine init either fits or refuses; no OOM mid-flight modulo bugs
- Upside: Sleep Mode replaces subprocess kill/restart, giving faster (and warmer) model switches than the current setup
- Upside: feature parity for the served model families is now upstream — no fork to maintain
- Upside: retires the V2/V3/0004 patch backlog
- Downside: turbo4 weight compression is not in vLLM (KV-only). Q4_K_M GGUF weights become BF16/FP8/AWQ on vLLM — meaningfully more VRAM per model
- Downside: one-model-per-GPU policy stays, but headroom budget shifts because weight format changes
- Downside: PFlash daemon and DFlash code paths are llama.cpp-specific and do not port; their workload-equivalent on vLLM is prefix caching + Sleep Mode + LMCache (production-stack tier)
- Downside: spec-decode acceptance behavior may differ in detail (vLLM's MTP integration is not bit-identical to the atomic-fork NextN/MTP path); needs a validation matrix

### Hybrid — keep llama.cpp for the Q4_K_M-served production models, run vLLM only for new families

- Upside: no immediate VRAM regression on existing served models
- Downside: two engines, two gateways, two patch surfaces, two telemetry shapes — multiplies operational complexity without solving the underlying llama.cpp tuning problem
- Downside: webui's engine-agnostic seam breaks down once two engines diverge on `/props` shape, multimodal handling, or tool-call format

## Rationale

The structural fix to "VRAM peak shifts per request" is preallocation; preallocation is vLLM's design contract and is not retrofittable to llama.cpp without rebuilding its allocator. Every additional patch in the V2 line is local — it converts aborts to recoverable errors but does not give the operator a stable tuning baseline. Sleep Mode further means the new engine's swap behavior is *better than* the current subprocess-swap baseline, not just equivalent. The remaining cost — losing turbo4 weight compression — is real but bounded and measurable; it is a one-time VRAM-budget recalculation, not an ongoing tuning trap.

## Migration Approach

Prototyping (off-prod):

1. Scaffold under `/tmp/grimoire-vllm-prototype/` — venv, install, smoke tests, benchmark scripts.
2. Free one GPU (operator picks which) and run a single vLLM engine there.
3. Validation matrix per served family (qwen3.6 NextN MTP, gemma-4 MTP with assistant head): correctness vs llama.cpp output on a fixed prompt set; spec-decode accept rate; TTFT; throughput; sleep/wake cycle stability over N iterations including a long-context multimodal prefill.
4. VRAM budget audit at AWQ-int4 / FP8 / BF16 weights for each production model, on a single 3090.

Cutover (when validation passes, separate decision):

1. Replace `model_manager.py` subprocess lifecycle with vLLM engine lifecycle + sleep/wake.
2. Collapse `/props` to synthetic-only.
3. Retire the patch chain and the `atomic-llama-cpp-turboquant` dependency; `webui/` submodule unchanged.
4. Update SPEC, STATUS, docker-compose, and the Dockerfile (vLLM image base instead of patched llama.cpp build).
5. Obsidian inference docs (`inference/00-overview.md`) updated to reflect the new engine and the Sleep Mode lifecycle.

Durable research artifacts (forthcoming, separate `RSH-*` files):

- vLLM AWQ-int4 vs llama.cpp Q4_K_M VRAM + quality comparison for qwopus3.6v2 and gemma-4-mtp-mmproj-31B
- Sleep Mode swap latency + correctness over a 24-hour soak
- MTP/NextN accept rate parity vs the atomic-fork baseline

## Consequences

- Prod remains on llama.cpp during prototyping; the change is reversible until the cutover DEC supersedes this one.
- The Atomic CUDA FA V2/V3 backlog (`RSH-20260523-001`, `RSH-20260526-001..003`) is paused, not retired, until cutover. Reopens only if the prototype fails.
- `webui/` submodule remains the SvelteKit fork; no upstream-rebase pressure from the engine swap.
- VRAM budget per model is expected to grow (no turbo4 weight quant). Each served model may require a smaller `max_model_len` than its current llama.cpp `--ctx-size`.
- Sleep Mode changes the operational model: model switches become near-instant; eviction becomes cheap; multi-model resident-on-CPU-RAM-with-one-woken becomes practical.
- PFlash daemon and the DFlash code paths are stranded on the llama.cpp engine and will be retired on cutover. Their workload contributions (cross-conversation prefix reuse, draft-model spec) are subsumed by vLLM's automatic prefix cache and native MTP/NextN.
- `inference/` Obsidian docs become stale on cutover and must be updated in the same change.

## Prototype Progress (2026-05-29/30)

### Environment

- Driver upgraded: `nvidia-driver-570` (570.211.01) → `nvidia-driver-580` (580.159.04), enabling CUDA 13 runtime
- vLLM 0.21.0 + PyTorch 2.11.0+cu130 installed in `/tmp/grimoire-vllm-prototype/.venv/`
- GPU 1 freed for vLLM; GPU 0 keeps production llama.cpp (gemma-4-mtp-mmproj-31B)
- DEC decision: one-model-per-GPU policy intact; no tensor-parallelism across GPUs

### vLLM Serving — Success

- Commercial AWQ model `cyankiwi/Qwen3.6-27B-AWQ-INT4` loads and serves correctly via vLLM on GPU 1
- Model uses 19.2 GiB VRAM; KV cache ~5,800 tokens at `max-model-len=2048` with `--enforce-eager`
- FP8 KV cache supported via FlashInfer software emulation (Ampere lacks native FP8)
- Chat completions confirmed working; model has CoT/reasoning output

### Model Quantization Pipeline

**Goal:** Quantize 27B/31B BF16 source models to AWQ for vLLM, targeting <17 GB to leave ≥7 GB for KV cache.

**Tooling:**

| Tool | Status | Reason |
|------|--------|--------|
| llmcompressor 0.10.0.2 | **Working** | Requires 3 patches: transformers `TORCH_INIT_FUNCTIONS` compat, `use_auth_token` removal, `_match_name` import. Separate venv `/tmp/quant-venv/` with transformers>=5.0 (for `qwen3_5` model type). See `RSH-20260529-006`. |
| autoawq 0.2.9 | **Unusable** | Deprecated; doesn't support `qwen3_5` architecture |
| Custom manual quant | **Not viable** | Weight packing incompatible with vLLM's Marlin/Exllama kernel expectations |

**llmcompressor Results — Model 5 (Qwen3.6-27B-heretic):**

- Source: 52 GB BF16 → Output: 19.17 GB (2.7× compression)
- Recipe: W4A16, gs=32, symmetric=false, `calibration_samples=128`
- Quantization took ~8 min (calibration + compression + write)
- Config format matches vLLM's compressed-tensors `pack-quantized` scheme

**vLLM Loading Issue:**

The quantized model loads weights (17.85 GiB) but vLLM's Marlin/Exllama kernel selection rejects layers with non-compliant shapes:

1. `linear_attn.in_proj_a/b` layers (outc=48) — Marlin requires outc%64==0
2. Visual encoder layers (outc=4304) — Marlin requires outc%64==0
3. llmcompressor strips multimodal weights from safetensors; vLLM recreates them and attempts to quantize

**Resolution:** Use regex ignore patterns (`re:.*visual.*`, `re:.*linear_attn.*`, `re:.*vision.*`) in quantization_config.ignore. This passes kernel selection but exposes a dimension mismatch (text-only `Qwen3_5ForCausalLM` vs multimodal `Qwen3_5ForConditionalGeneration`). Full end-to-end load pending final config patching.

### VRAM Assessment

| Model | BF16 Source | Expected AWQ | VRAM free on 24 GiB |
|-------|------------|--------------|---------------------|
| Qwen3.6-27B (stock) | 54 GB | ~17 GB | ~7 GB (~17K ctx at FP8 KV) |
| Qwen3.6-27B-heretic | 52 GB | ~17 GB | ~7 GB |
| Qwopus3.6-27B | 54 GB | ~17 GB | ~7 GB |
| gemma-4-31B-it | 62 GB | ~19 GB | ~5 GB (~12K ctx at FP8 KV) |
| gemma-4-31B-heretic | 62 GB | ~19 GB | ~5 GB |

### Next Steps

1. Fix llmcompressor recipe to include `LinearAttention` in ignore targets (prevent quantization of GDN linear attention layers)
2. Resolve text-only vs multimodal architecture mismatch in quantized model config
3. Complete quantization pipeline for remaining 4 models (download → quantize → delete BF16 source → next)
4. Validate vLLM serving with full-context benchmarks
5. Compare AWQ quality vs current GGUF Q4_K_M production baseline
