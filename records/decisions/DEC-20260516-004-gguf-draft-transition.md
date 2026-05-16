# GGUF Draft Transition Policy

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Decision

GGUF is the end-state artifact format for DFlash draft models, but the served config stays on the existing `.safetensors` draft until the converted GGUF is proven on hardware and versioned. The native canary (`dflash-native-qwen3.6-27B-canary`) uses the GGUF path as a testing ground.

## Context

The current DFlash draft lives at `dflash/Qwen3.6-27B-DFlash/model.safetensors`. The conversion pipeline (`lucebox/dflash/scripts/convert_dflash_to_gguf.py` / `quantize_draft_q8.py`) produces a GGUF with arch `qwen35-dflash-draft` or `dflash-draft`. The native TheTom path requires GGUF format with arch `dflash-draft`. Flipping the served config from safetensors to GGUF would change the draft loading path for all production requests — if the converted artifact is incorrect, every request fails.

## Options Considered

| Option | Consequence |
| --- | --- |
| Staged: safetensors stays served, GGUF on canary only (chosen) | Zero production risk. Canary proves the path. |
| Immediate flip to GGUF | Faster retirement of the old path, but production outage if GGUF is malformed. |

## Rationale

- The `.safetensors` draft is working in production today.
- The native canary exists specifically to de-risk the GGUF cutover.
- The conversion pipeline is new code that has not been tested on real hardware.
- MIGRATION_EXECUTION_CHECKLIST.md line 23: "do not flip the served config from .safetensors to GGUF until the converted artifact is proven and versioned."

## Consequences

- The served `dflash-pflash-qwen3.6-27B` config stays on safetensors.
- The canary `dflash-native-qwen3.6-27B-canary` uses GGUF draft.
- A future decision record will document the cutover after hardware validation completes.
