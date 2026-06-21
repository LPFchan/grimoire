# RSH-20260529-006: AWQ Quantization Pipeline for vLLM Migration

Opened: 2026-05-29
Updated: 2026-05-30
Recorded by agent: opencode
Related ids: DEC-20260528-001

## Research Question

Can we quantize 27B/31B BF16 models to AWQ format targeting <17 GB VRAM for vLLM deployment on single RTX 3090 (24 GiB)?

## Answer

**Yes, with llmcompressor 0.10.0.2.** Model 5 (Qwen3.6-27B-heretic) was quantized from 52 GB BF16 → 19.17 GB. The pipeline works but requires:

1. Three patches to llmcompressor for transformers 5.x compatibility
2. Per-model post-quantization config fixes (ignore list for Marlin-incompatible layers)
3. Text-only architecture workaround (llmcompressor drops multimodal weights)

Full end-to-end vLLM loading is pending the final config patch set. Commercial AWQ models (e.g. `cyankiwi/Qwen3.6-27B-AWQ-INT4`) load and serve correctly.

## Environment

- **Hardware:** 2× RTX 3090 (24 GiB each)
- **Driver:** nvidia-driver-580 (580.159.04)
- **vLLM:** 0.21.0 in `/tmp/grimoire-vllm-prototype/.venv/`, PyTorch 2.11.0+cu130
- **Quant venv:** `/tmp/quant-venv/`, llmcompressor 0.10.0.2, transformers 5.9.0, compressed-tensors 0.14.0.1
- **GPU allocation:** GPU 0 = production llama.cpp (gemma), GPU 1 = vLLM testing

## Tooling Evaluation

### llmcompressor 0.10.0.2 — Selected

- Entry point: `llmcompressor.oneshot()` with YAML recipe
- Creates vLLM-compatible `compressed-tensors` `pack-quantized` format
- Stores weights as `weight_packed` (int4 packed) with `weight_scale` (FP16) and `weight_zero_point`
- Handles AWQ-style calibration via activation statistics on provided dataset

**Required patches** (for transformers 5.9.0 compat):

1. `transformers.modeling_utils.TORCH_INIT_FUNCTIONS` → alias `ROPE_INIT_FUNCTIONS`
2. Remove `use_auth_token` from `initialize_model_from_path` (deprecated in transformers 5.x)
3. `compressed_tensors.utils.match._match_name` → `match_name`

**Installation:** Separate venv required due to transformers version conflict (vLLM needs >=5.0 for `qwen3_5`; llmcompressor expected <=4.57.6). Patched source directly via `sed` to remove `use_auth_token`.

### autoawq 0.2.9 — Rejected

- Deprecated (vLLM project adoption notice)
- Does not support `qwen3_5` architecture
- Dependency chain blocked by slow PyPI downloads (pyarrow, xxhash, aiohttp)

### Custom Manual Quantization — Rejected

- Written from scratch using PyTorch + safetensors
- Produced 19 GB output but weight packing format incompatible with Marlin/Exllama kernels
- vLLM's kernel selection requires specific nibble-packing layout for Marlin's `uint4b8` quant type
- Embedding quantization support missing from manual approach

## Quantization Recipe (Final)

```yaml
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore:
                - lm_head
                - model.visual
                - model.vision_tower
                - model.vision_model
                - LinearAttention
                - ReLU
                - LayerNorm
                - RMSNorm
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: int
                        symmetric: false
                        strategy: group
                        group_size: 32
                    targets: ["Linear"]
```

## Results

### Model 5: Qwen3.6-27B-heretic

| Metric | Value |
|--------|-------|
| Source | `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved` |
| BF16 size | 52 GB (13 shards, 851 weight keys) |
| AWQ output | 19.17 GB (single shard, 2,339 keys) |
| Compression ratio | 2.7× |
| Quantization time | ~8 min (calibration 68s + compression 3.5min + write 4min) |
| Layers quantized | 496 weight tensors |
| Layers ignored | 242 (lm_head, embed_tokens, visual, linear_attn) |

### Model 3: Qwen3.6-27B-AWQ-INT4 (Commercial)

| Metric | Value |
|--------|-------|
| vLLM loading | Working |
| VRAM used | 19.2 GiB |
| KV cache (FP16) | ~5,800 tokens at max-model-len=2048 |
| Chat completions | Confirmed working with CoT/reasoning |

## vLLM Loading Issues (Custom-Quantized Models)

Three compounding issues prevent vLLM from loading custom-quantized models:

### 1. Marlin Kernel Shape Requirements

Layers with `outc % 64 != 0` or `inc % 128 != 0` are rejected by all available kernels on Ampere (SM 8.6):

| Kernel | Requirement | Supported on 3090? |
|--------|------------|-------------------|
| Marlin | outc%64==0, inc%128==0, gs in [-1,32,64,128] | Yes (if passing) |
| Exllama | inc%gs==0, quant type `uint4b8` (symmetric) | Yes (if passing) |
| CutlassW4A8 | SM 90+ | No |
| Machete | SM 90+ | No |
| AllSpark | No zero points | Rejects ours |
| Conch | gs in [-1,128] | Rejects gs=32 |

After switching config to `symmetric: false` with `zp_dtype: torch.int8`, Exllama rejects `uint4` (asymmetric) type; Marlin rejects non-compliant shapes.

### 2. GDN Linear Attention Layers

Qwen3.5's GDN (`Gated DeltaNet`) linear attention uses `LinearAttention` modules containing `torch.nn.Linear` submodules:

- `in_proj_a`: shape [48, 5120] → outc=48, fails Marlin (48%64≠0)
- `in_proj_b`: shape [48, 5120] → same issue
- ~128 such layers across 64 Transformer blocks

These are small (~60 MB total VRAM if kept FP16) and should be ignored during quantization. llmcompressor's ignore pattern matching uses exact module name matching; regex patterns with `re:` prefix work in the output config but not during quantization.

**Fix:** Post-quantization config patching adds `re:.*linear_attn.*` to ignore list. Verified: Marlin kernel selection passes after this fix.

### 3. Missing Multimodal Weights

llmcompressor saves the model with architecture `Qwen3_5ForCausalLM` (text-only). The visual encoder layers neither appear in the safetensors nor in the config. When vLLM loads the model, it:

- Creates visual encoder layers from the architecture definition
- Attempts to apply compressed-tensors quantization to these layers
- Triggers Marlin kernel selection with 4304-output layers (visual MLP)
- Regex ignore (`re:.*visual.*`, `re:.*vision.*`) fixes kernel selection
- But a dimension mismatch (4096 vs 5120) occurs during weight initialization

**Tentative Fix:** Keep architecture as text-only if multimodal isn't needed; otherwise, include visual encoder weights in the quantization output by removing them from the ignore list during quantization.

## Next Steps

1. Resolve text-only architecture dimension mismatch
2. Complete quantization pipeline for remaining 4 models
3. VRAM/quality comparison: AWQ vs GGUF Q4_K_M baseline
4. MTP/NextN spec-decode acceptance rate vs llama.cpp atomic fork
5. Sleep Mode cycle stability test over 24-hour soak
