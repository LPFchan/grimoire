# Project Spec

**Project:** Grimoire — Model serving gateway with DFlash/PFlash speculative decoding
**Canonical repo:** `git@github.com:LPFchan/grimoire.git` (refactor branch)
**Operator:** LPFchan
**Last updated:** 2026-05-16

## Thesis

Serve LLMs behind an OpenAI-compatible gateway with speculative decoding acceleration (DFlash tree-attention and PFlash compression), initially built on a Lucebox dflash base, migrating to a canonical TheTom/llama-cpp-turboquant base.

## Capabilities

- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints
- DFlash speculative decode via tree-attention with rollback cache
- PFlash prompt compression via standalone daemon
- Session persistence with `compact-full` snapshot mode (RAM-first, async disk mirror)
- Prefix caching, session KV slots, park/unpark for cold models
- Multi-model registry with GPU pinning
- Block-aware prompt compression with message boundary preservation

## Invariants

- `snapshot-mode=compact-full` is required for DFlash. No other snapshot mode is accepted.
- `pflash-qwen3.6-27B` is text-only. Multimodal is retired for this model.
- The non-PFlash llama path must never depend on `/opt/dflash` for symbol resolution.
- The final served runtime must not ship `/opt/dflash`.
- GGUF is the target end-state for DFlash draft artifacts. The served `.safetensors` draft stays until the converted GGUF is proven and versioned.

## Surfaces

| Surface | Path |
| --- | --- |
| Served config | `etc/models.json` |
| Control plane | `src/grimoire/` |
| Native C++ sources | `lucebox/dflash/src/` |
| TheTom canonical base | `tmp/spec-analysis/thetom-shallow/` |
| DFlash source refs | `tmp/spec-analysis/buun-shallow/`, `tmp/spec-analysis/bee-shallow/`, `tmp/spec-analysis/ggml-pr22105/` |
| PFlash upstream baseline | `tmp/spec-analysis/lucebox-hub/` |
| Tests | `tests/test_dflash.py`, `tests/test_e2e_smoke.py`, `tests/test_pflash_pipeline.py`, `tests/test_stress_dflash.py`, `tests/test_drop_in_blockers.py`, `tests/test_llama_proxy.py` |
