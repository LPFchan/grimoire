# VMM Park/Unpark Overhead and VRAM Cost (2026-05-18)

Opened: 2026-05-18 18-30-00 KST
Recorded by agent: opencode

## Question

Does CUDA VMM park/unpark for the PFlash drafter slot provide measurable benefit over simple process coexistence, and what is the VRAM cost?

## Method

Loaded two model variants back-to-back on the same GPU (RTX 3090, 24 GB):

| Model | Description |
|-------|-------------|
| `pflash-park-qwen3.6-27B` | llama-server LD_PRELOAD'd with `pflash_shim.so` — all `cudaMalloc` redirected to VMM (`cuMemCreate`+`cuMemMap`). Park/unpark via FIFO commands to the shim's background thread. |
| `pflash-qwen3.6-27B` | Same GGUF/drafter, no LD_PRELOAD — llama-server and pflash_daemon coexist in VRAM normally. |

Each model received identical 4-message conversation (~18.5K tokens, above 36K PFlash threshold). Cold and warm TTFT measured via gateway API.

## Results

| Metric | With VMM | Without VMM | Delta |
|--------|----------|-------------|-------|
| Cold TTFT | 1.442s | 1.491s | -49ms |
| Warm TTFT | 1.462s | 1.435s | +27ms |
| VRAM after load | 22,837 MB | 20,347 MB | **+2,490 MB** |
| Cached tokens | 18,506 | 18,506 | 0 |

## Key Findings

1. **Negligible TTFT impact** (±50ms, within measurement noise). Park/unpark overhead (~0.34s round-trip copy of drafter data over PCIe) happens only on cold turns and is dwarfed by the 1.44s compression+prefill time.

2. **2.5 GB extra VRAM** with VMM. The shim's `vmm_init` reserves VA space (`cuMemAddressReserve` up to 22 GB) and VMM-allocated memory appears in `nvidia-smi` as used even when parked. On a 24 GB card, this leaves only ~1.2 GB free — dangerously tight.

3. **VMM only helps in multi-model sharing** scenarios where one model must yield VRAM to another. Park/unpark cycle takes ~2.6s (copy ~15 GB weights + KV at PCIe 3.0 speeds) vs SIGTERM+reload at ~24s. But for single-model operation (current deployment), simple coexistence works identically.

## Conclusion

VMM park/unpark is technically functional but unnecessary for the current single-model deployment. The 2.5 GB VRAM overhead is prohibitive on 24 GB cards. Recommend not pursuing unless multi-model GPU sharing is needed in the future.

## Related

- `pflash_shim.c`: VMM interception + park/unpark FIFO protocol
- `model_manager.py` lines 194-200: LD_PRELOAD setup
- `model_manager.py` lines 208-246: park/unpark FIFO client
- `proxy/llama.py` lines 149-175: cold turn park/unpark usage
