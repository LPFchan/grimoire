# RSH-20260519-011: DFlash ctx-size VRAM Sweep (dflash-qwen3.6-27B)

Opened: 2026-05-19 18-56-47 KST
Recorded by agent: mimo-v2.5-pro-precision

## Question

What is the maximum `ctx-size` the dflash model (`dflash-qwen3.6-27B`) can sustain on a single RTX 3090 (24 GB) before VRAM OOM?

## Setup

- **GPU**: RTX 3090 (24,576 MiB)
- **Model**: `gguf/Qwen3.6-27B-Q4_K_M.gguf` (~16 GB)
- **Draft**: `gguf/dflash-draft-3.6-q8_0.gguf` (~1.76 GB)
- **KV cache**: `q8_0` / `q8_0`
- **Parallel**: 1
- **Backend**: Bee llama-server (DFlash native spec decoding)
- **Env**: `GGML_DFLASH_MAX_VERIFY_TOKENS=20`

## Method

Binary search of `ctx-size` in `etc/models.json`, restarting the model via `POST /switch` and checking health endpoint + `nvidia-smi` VRAM reading. An OOM is detected when llama-server exits with code 1 before becoming ready.

## Results

| ctx-size | VRAM (MiB) | Free (MiB) | KV cache Δ | Status |
|----------|------------|------------|------------|--------|
| 60,000 | 20,234 | 4,342 | — | PASS (baseline) |
| 100,000 | 21,560 | 3,016 | +1,326 | PASS |
| 120,000 | 22,224 | 2,352 | +664 | PASS |
| 130,000 | 22,546 | 2,030 | +322 | PASS |
| 135,000 | 22,716 | 1,860 | +170 | PASS |
| 140,000 | 22,886 | 1,690 | +170 | PASS |
| 145,000 | 23,056 | 1,520 | +170 | PASS |
| 150,000 | 23,210 | 1,366 | +154 | PASS |
| 155,000 | 23,380 | 1,196 | +170 | PASS |
| 160,000 | 23,550 | 1,026 | +170 | PASS |
| 165,000 | 23,720 | 856 | +170 | PASS |
| 170,000 | 23,890 | 686 | +170 | PASS |
| 175,000 | 24,042 | 534 | +152 | PASS |
| 180,000 | 23,942 | 634 | -100 | PASS (variance) |
| 182,500 | 24,028 | 548 | +86 | PASS |
| 183,750 | 24,062 | 514 | +34 | PASS |
| 184,375 | 24,096 | 480 | +34 | PASS |
| 184,688 | 24,096 | 480 | 0 | PASS |
| 184,766 | 24,096 | 480 | 0 | PASS |
| 184,805 | 24,096 | 480 | 0 | PASS |
| 184,824 | 24,096 | 480 | 0 | PASS |
| 184,829 | 24,096 | 480 | 0 | PASS |
| 184,831 | 24,096 | 480 | 0 | PASS |
| **184,832** | **24,096** | **480** | **0** | **PASS (max)** |
| 184,833 | — | — | — | FAIL (exit code 1) |
| 184,844 | — | — | — | FAIL |
| 185,000 | — | — | — | FAIL |

## Findings

1. **Max ctx-size: 184,832** — a 3.08x increase from the original 60,000.
2. **VRAM ceiling**: 24,096 MiB at max ctx, leaving ~480 MiB headroom. Beyond this, `llama-server` exits with code 1 (CUDA OOM).
3. **KV cache growth rate**: ~170 MiB per 5,000 tokens = ~34 MiB per 1,000 tokens = ~0.034 MiB/token for `q8_0` KV cache on this architecture.
4. **Model + draft base overhead**: ~18,000 MiB (inferred: 20,234 - 0.034 * 60,000 ≈ 18,194 MiB).
5. **Measurement ceiling at 24,096 MiB**: VRAM readings plateau at 24,096 MiB for all ctx sizes above ~184,000 before OOM, suggesting the GPU memory allocator's last successful allocation hits this ceiling.

## Configuration Change

`etc/models.json` updated: `ctx-size` changed from 60,000 to 184,832 for `dflash-qwen3.6-27B`.

## Open Questions

- Would `cache-type-k f16` or a different quantization allow significantly more context at the cost of slower decode?
- Could the draft model be offloaded to CPU to free VRAM for main model context?
- What is the actual decode TPS at max ctx vs the original 60k — does the larger KV cache hurt throughput?
