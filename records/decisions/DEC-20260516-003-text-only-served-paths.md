# Served Paths Are Text-Only

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Decision

`pflash-qwen3.6-27B` and `dflash-pflash-qwen3.6-27B` are text-only served paths. Multimodal serving and `mmproj` wiring are retired for these models. The baseline `qwen-3.6-27B` remains multimodal.

## Context

The served PFlash and DFlash paths for the Qwen3.6-27B model never used multimodal in production. The `mmproj` entries in models.json were carry-over from the baseline llama config and added unnecessary VRAM overhead and complexity. The PFlash compressor operates on text tokens, not image embeddings — carrying multimodal support adds zero value and creates a vector for configuration errors.

## Options Considered

| Option | Consequence |
| --- | --- |
| Text-only (chosen) | Cleaner configs, less VRAM, simpler compression |
| Keep multimodal | Unused code path, but allows future multimodal + PFlash without config changes |

## Rationale

- No production usage of multimodal on the PFlash or DFlash served paths.
- The `mmproj` file is 600-1145 MiB — removing it frees VRAM for the actual model.
- Block-aware compression operates on token sequences, not image patches. Multimodal would require a separate compression strategy.
- MIGRATION_EXECUTION_CHECKLIST.md line 58 explicitly retired multimodal for this model.

## Consequences

- Registry configs for `pflash-qwen3.6-27B` and `pflash-park-qwen3.6-27B` have no `mmproj`.
- `qwen-3.6-27B` and other baseline models keep their multimodal configs unchanged.
- Future multimodal support on the DFlash path would require a new decision record.
