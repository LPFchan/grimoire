# DEC-20260703-001: Script Training Checkpoint Intake
Opened: 2026-07-03 17-39-14 KST
Recorded by agent: codex

## Metadata
- Status: accepted
- Deciders: operator, orchestrator
- Related ids: LOG-20260703-174035-codex

## Decision
Training checkpoints for Eastself B, B-prime, and later variants are taken into Grimoire through a scripted, manifest-backed path.

The intake procedure starts from a PEFT checkpoint directory, a base GGUF, and an optional local Hugging Face base config directory. It produces a compact LoRA GGUF, a tokenizer-aligned base GGUF, a provenance manifest, and an optional Grimoire registry entry.

Tokenizer rewrites must use exact `trainable_token_indices` from the checkpoint `adapter_config.json` or an explicit token-id JSON file. Numeric token-id ranges are retained only as a manual fallback.

## Context
The first Eastself B-prime Grimoire serving path proved the native runtime contract, including compact trainable-token replacements and tokenizer-aligned GGUF serving. The remaining weak point was procedural drift: artifact names, token ranges, converter execution context, and registry updates were still mostly reconstructed from session history.

The checkpoint-6500 adapter also carried a stale training-box `base_model_name_or_path` (`/workspace/models/gemma-4-31b`), while the converter dependencies live in the Grimoire container. The canonical path therefore needs to run LoRA conversion in the container and make the local base HF config explicit when the checkpoint path is not valid on Grimoire.

## Options Considered
### Keep Ad Hoc Shell Commands
- Upside: fastest for one checkpoint.
- Downside: future B/B-prime checkpoints can silently drift on token IDs, names, or registry shape.

### Preserve Numeric Token Ranges
- Upside: matched checkpoint-6500 because its trainable token IDs are contiguous.
- Downside: unsafe when the sticker/custom token map changes or becomes non-contiguous.

### Scripted Exact-ID Intake
- Upside: makes token-map changes explicit, validates checkpoint structure, records artifact provenance, and can reproduce current live artifact names through path overrides.
- Downside: the converter still needs a valid base HF config path when a checkpoint embeds a stale training-machine path.

## Rationale
The durable boundary is the checkpoint metadata, not the currently observed token range. Reading exact token IDs from `adapter_config.json` makes the tokenizer rewrite robust across future B/B-prime checkpoints and token-map changes.

Running the converter through the existing Grimoire container avoids adding a second dependency surface on the host. Writing a manifest makes each served adapter/base pairing auditable without relying on chat transcript recovery.

## Consequences
- `scripts/intake-peft-checkpoint.py` is the procedural entrypoint for future Eastself checkpoint intake.
- `scripts/write-gguf-tokenizer-from-hf.py --adapter-config ...` is the canonical tokenizer rewrite mode.
- Future checkpoint intake should provide `--base-hf` when `adapter_config.json` points at a training-machine path that is not mounted on Grimoire.
- Existing live ck6500 artifacts can be represented by explicit `--adapter-gguf` and `--tokenizer-gguf` overrides rather than renamed in place.
