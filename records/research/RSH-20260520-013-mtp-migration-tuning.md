# RSH-20260520-013: MTP Migration — Deployment and Parameter Tuning

Opened: 2026-05-20 21-00-00 KST
Recorded by agent: mimo-v2.5-pro-precision

## Summary

Complete migration from Bee's DFlash/PFlash stack to `AtomicBot-ai/atomic-llama-cpp-turboquant`
with MTP/NextN speculative decoding. All three production models replaced with MTP variants.

## Architecture

### Previous Stack (Retired)
- **Bee llama.cpp** fork with DFlash speculative decoding
- 7 model entries: qwen, huihui, gemma + DFlash/PFlash variants
- ctx halving (n_seq_max = 2 × parallel) limiting DFlash to ~92k effective context

### Current Stack
- **atomic-llama-cpp-turboquant** (feature/turboquant-kv-cache branch)
- 3 model entries, all with MTP/NextN speculation
- No ctx halving — MTP heads share the target model's full context

### Models

| Alias | Base | Spec type | File |
|-------|------|-----------|------|
| qwen3.6-mtp-27B | Qwen3.6-27B | NextN | `Qwen3.6-27B-MTP-Q4_K_M.gguf` |
| huihui-qwen3.6-mtp-27B | Qwen3.6-27B (abliterated) | NextN | `Huihui-Qwen3.6-27B-abliterated-MTP-Q4_K.gguf` |
| gemma-4-mtp-31B | Gemma 4 31B | MTP (separate head) | `gemma-4-31B-it-Q4_K_M.gguf` + assistant head |

### Binary
- Source: `AtomicBot-ai/atomic-llama-cpp-turboquant`
- Built inside CUDA 12.8 devel container for GLIBC 2.35 compatibility
- Build tools added to CMakeLists.txt: `add_subdirectory(tools)` (not in upstream)
- Binary mounted at `/opt/atomic/llama-server`, libs at `/opt/atomic/`

### Config Changes
- `config.py`: `TURBOQUANT_LIB_DIR` / `TURBOQUANT_LIB64_DIR` now env-configurable
- `model_manager.py`: Added NextN/MTP speculative type handling
- `docker-compose.yml`: Mount atomic binary, set `GRIMOIRE_TURBOQUANT_LIB_DIR=/opt/atomic`

## Parameter Sweep Results

### Qwen3.6 NextN — draft-max sweep

10-run averaged, 512 gen tokens, short prompt:

| draft-max | Decode tok/s | Acceptance |
|-----------|-------------|------------|
| 1 | 57.3 | 81% |
| 2 | 57.2 | 81% |
| 3 | 57.3 | 82% |
| 4 | 56.3 | 78% |
| 6 | 56.8 | 80% |
| 8 | 56.0 | 78% |

**Verdict**: draft-max has negligible impact on the 27B dense model
(draft-compute-bound). Current setting `--draft-max 2 --draft-min 1` is optimal.

### Gemma 4 MTP — draft-block-size sweep

10-run averaged, 512 gen tokens, short prompt:

| Block size | Decode μ | σ | Acc μ | σ | n |
|-----------|---------|---|-------|---|--|
| **2** | **45.1** | ±0.6 | **79.3%** | ±1.8% | 10 |
| 3 | 44.1 | ±0.8 | 71.7% | ±2.3% | 10 |
| 4 | 40.5 | ±0.8 | 60.9% | ±2.0% | 10 |

**Verdict**: Block size 2 is significantly better than 3 or 4 across
all metrics. Current setting `--draft-block-size 2` is optimal.

## Production Configs

```json
"qwen3.6-mtp-27B": {
  "speculative-type": "nextn",
  "extra-args": ["--draft-max", "2", "--draft-min", "1"]
}
"gemma-4-mtp-31B": {
  "speculative-type": "mtp",
  "mtp-head": "gguf/gemma-4-31B-it-assistant.Q4_K_M.gguf",
  "extra-args": ["--draft-block-size", "2"]
}
```

## Disk Cleanup

- Old models purged: ~37 GB
- Build artifacts purged: ~2.5 GB
- Old Docker images pruned: ~15 GB
- Total reclaimed: ~54 GB (47G → 101G at peak)
- Final free: 69 GB
