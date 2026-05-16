# Port Bee's DFlash Pipeline Onto TheTom

Opened: 2026-05-16 16-00-00 KST
Recorded by agent: opencode

## Decision

Port Bee's working DFlash speculative decoding pipeline from `Anbeeld/beellama.cpp` onto our TheTom base, rather than debugging our broken simplified patch. The scope is ~2000 lines concentrated in `common/speculative.cpp`, `tools/server/server-context.cpp`, and supporting `llama-context.cpp` changes. GPU ring buffer, K/V projection cache, and GPU tape recording are deferred — not required for basic decode.

## Context

Our `spec-dflash-contract.patch` added a simplified DFlash spec-decode to TheTom, but produces garbage draft tokens (uninitialized memory values like -1079354870). After confirming:

- Buun's `dflash-draft-3.6-q8_0.gguf` loads correctly (arch `dflash-draft`, 58 tensors, full 496K tokenizer)
- Bee's implementation works with this GGUF (confirmed by issue #16 reporter `@ppsx`)
- Our patch is a simplified version that omitted Bee's `prepare_batch_draft()`, ring buffer, and `flush_prefill()` — the cross-data path is broken

## Options Considered

| Option | Outcome |
| --- | --- |
| Port Bee's pipeline onto TheTom (chosen) | Working decode with turboquant. Bee's code is concentrated (~20 files, core in ~3 files). |
| Debug our broken patch piecemeal | Unknown time cost; root cause is the flat `vector<float> history` approach which is architecturally wrong |
| Rebuild from Buun's repo | Buun has no DFlash code at any commit depth — no advantage over TheTom |
| Rebuild from Bee's repo + port turboquant | Bee has no turboquant; porting turboquant is several times larger than porting the DFlash pipeline |

## Rationale

- Bee's implementation is proven working on the same GGUF (Buun's)
- The core pipeline lives in well-defined files (`common/speculative.cpp` ~1000 lines of DFlash-specific code, `tools/server/server-context.cpp` ~400 lines)
- Our TheTom base already has turboquant and DFlash model architecture (via existing patch) — only the spec-decode pipeline needs replacement
- Phases deferred (GPU ring, KV cache, tape) are not on the critical path for basic speculative decode

## Consequences

- Phase 1 (core pipeline) estimated 4 days
- Phase 2 (server integration) estimated 5 days
- Phase 3 (supporting infrastructure) estimated 5.5 days
- GPU ring buffer, K/V projection cache, and GPU tape recording are deferred
- Target layer IDs from Buun's GGUF ([9,5,5,1,16,31,46,61]) may differ from our default [0,1,2,3,4] — must verify against Qwen3.6-27B architecture
