# Canonical DFlash/PFlash Migration Checklist

## Mission
- Migrate the served DFlash/PFlash stack onto the canonical `TheTom/llama-cpp-turboquant` base.
- Preserve live runtime contracts first. Porting files is not enough.
- Lucebox retirement is blocked until decode, DFlash `compact-full` persistence, and the preserved llama-side PFlash path are all green.

## Inputs To Pin Before Coding
- Canonical target repo: working fork of `TheTom/llama-cpp-turboquant` using `tmp/spec-analysis/thetom-shallow/` as the local snapshot. Current local SHA used for patch generation and Docker build verification: `69d8e4be47243e83b3d0d71e932bc7aa61c644dc`.
- DFlash source refs: `tmp/spec-analysis/buun-shallow/`, `tmp/spec-analysis/bee-shallow/`, and `tmp/spec-analysis/ggml-pr22105/`. Current local SHAs: `buun-shallow=853eebdd02c2db4baf7bf781adadee6e7ce1d44e`, `bee-shallow=633cd34fb6df72ad88a74e9320dad03ddd788fb1`, and `ggml-pr22105=320a6a44a5b1de6a074ba781e65f5fd79fb4051a`.
- PFlash upstream baseline: `tmp/spec-analysis/lucebox-hub/` pinned to the `de31881` file baseline for `pflash_daemon`, `qwen3_drafter`, and `SPEC_PREFILL`. Current local SHA: `e5347801719ad7d45a3d7bd096e9e57778ce23ea`.
- Local native source of truth today: `lucebox/dflash/`.
- Local control-plane source of truth today: `src/grimoire/`.
- Required local native fixes to carry forward: all required `b4ed333` fixes plus `e4d4e32` for the `qwen3_5_0p8b_graph.cpp` leak fix.
- Do not rely on local folder names alone. Every upstream input must have an exact SHA recorded in this document before implementation starts.

## Current Served Contracts

### Contract A: DFlash Decode Contract
- Primary served model today: `dflash-pflash-qwen3.6-27B` in `etc/models.json`.
- Current target artifact: `gguf/Qwen3.6-27B-Q4_K_M.gguf`.
- Current draft artifact: `dflash/Qwen3.6-27B-DFlash/model.safetensors`.
- End-state target is GGUF for the DFlash draft path, but do not flip the served config from `.safetensors` to GGUF until the converted artifact is proven and versioned.
- Preserve text-only chat semantics, `parallel=1`, `ctx-size=60000`, `max-effective-context=60000`, `budget=18`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, `fa-window=2048`, and the `<|im_end|>` stop-string behavior unless a later deliberate product change approves otherwise.
- Prompt handling is block-aware. Compression boundaries, message metadata, and prompt layout semantics must survive `_prompt_layout_from_messages` -> compression -> reconstruction.
- Final served decode path must not require the Lucebox decode daemon.

### Contract B: DFlash `compact-full` Persistence Contract
- `snapshot-mode=compact-full` is required. It is not optional in this migration.
- Preserve session restore keyed by `conversation_id` and validated against the effective prompt prefix hash.
- Preserve the transient staging slot contract, currently `snapshot-staging-slot=7`, unless an explicitly equivalent mechanism replaces it.
- Preserve prompt-boundary inline snapshot behavior for reusable prefix work.
- Preserve RAM-first snapshot writes with asynchronous disk mirroring, manifest persistence, disk TTL, and RAM/disk budget enforcement.
- Preserve stale-session invalidation on prompt-prefix mismatch. Silent reuse of stale snapshots is a hard failure.
- Preserve support for prefix-cache semantics even if the default served config keeps `prefix-cache-slots=0`.
- Preserve all state needed for correct continuation, including KV plus any recurrent/native state and `target_feat` equivalents.

### Contract C: Preserved Llama-Side PFlash Contract
- Primary supported and served models today: `pflash-qwen3.6-27B` and `pflash-park-qwen3.6-27B` in `etc/models.json`.
- Preserve the standalone `pflash_daemon` path until an in-process replacement wins on measured TTFT, VRAM behavior, and startup cost without regressions.
- Preserve the current warm/cold split behavior unless retirement of that split is explicitly approved.
- Preserve FIFO-based park/unpark for park models, currently via `pflash_shim.so`.
- Preserve the `/slots/0` save/restore control-plane contract and the conversation-keyed `.kv` naming under `/dev/shm/grimoire-slots` unless an explicitly equivalent contract replaces it.
- Preserve the raw `int32` compressor protocol `compress <path> <keep_x1000>`.
- Preserve the `Qwen3.5-0.8B Q8_0` compressor path and all required native fixes.
- Preserve the current block-aware compression followed by token -> text -> message reconstruction. Direct prompt/token integration is out of scope for this migration.
- `pflash-qwen3.6-27B` is a text-only served path. Multimodal is retired for this model.

### Contract D: Runtime Isolation Contract
- End-state removes `/opt/dflash` entirely from the served runtime image, startup environment, and runtime search path.
- Any remaining `/opt/dflash` dependency before cutover is temporary migration debt and must be limited to pre-cutover preservation work.
- The canonical non-PFlash llama path must be anchored on `TheTom` libraries only throughout the migration.

## Required Decisions Before Phase 1
- Retirement target is `A + B + C`. Decode-only cutover is not allowed.
- `compact-full` parity is required.
- DFlash remains text-only for this migration.
- Served `pflash-qwen3.6-27B` is text-only. Do not carry multimodal behavior or `mmproj` wiring for this model into the migrated stack.
- GGUF is the target end-state for DFlash draft artifacts, but the served artifact flip happens only after the canonical path is proven.
- `Bee` is a selective helper source only. Do not adopt Bee wholesale unless the selective path fails and the failure is documented.
- Standalone PFlash packaging is the default first target.
- The llama-side PFlash path keeps the current token -> text -> message reconstruction flow after compression. Direct prompt/token integration is out of scope for this migration.
- VMM-based park/unpark is preferred only if isolated measurement proves the gain. Older SIGTERM/page-cache reload behavior is fallback, not default design.
- Full removal of `/opt/dflash` is required for the final served runtime, not just normal-llama decoupling.
- Smoke and stress harness defaults using the stale alias `dflash-pflash-qwen-27B` must be aligned with current served model IDs before they are treated as release gates.

## Phase 0 Status Artifacts
- The inventory and behavior matrix below close the remaining software-only Phase 0 documentation gap.
- Baseline metrics for decode, restore, compression, TTFT, decode TPS, and snapshot-store growth are still open and remain hardware-gated.

### Required-vs-Retired Behavior Matrix
| Surface | Must Preserve | Explicitly Retired / Not A Gate Yet |
| --- | --- | --- |
| `dflash-pflash-qwen3.6-27B` served decode | Text-only chat semantics, `parallel=1`, `ctx-size=60000`, `max-effective-context=60000`, `budget=18`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, `fa-window=2048`, `<|im_end|>` stop-string behavior, block-aware prompt compression/reconstruction, and `snapshot-mode=compact-full`. | Final served-path dependency on the Lucebox decode daemon. Early `.safetensors` -> GGUF draft cutover before the converted artifact is proven and versioned. |
| `pflash-qwen3.6-27B` preserved PFlash path | Standalone `pflash_daemon`, raw `compress <path> <keep_x1000>` protocol, `Qwen3.5-0.8B Q8_0` drafter path, and token -> text -> message reconstruction after compression. | Multimodal serving and `mmproj` wiring for this model. |
| `pflash-park-qwen3.6-27B` preserved park path | FIFO park/unpark via `pflash_shim.so` plus the same text-only compression semantics as `pflash-qwen3.6-27B`. | Global `/opt/dflash` library-path coupling outside preserved PFlash components. |
| `qwen-3.6-27B` normal llama path | Canonical non-PFlash startup and serve path, including the current multimodal config in `etc/models.json`, must resolve against `TheTom` libraries without `/opt/dflash`. | Any hidden fallback that only works because `/opt/dflash` is present in the runtime search path. |
| `dflash-native-qwen3.6-27B-canary` native canary | Dormant native-control-plane launch contract for compile/test coverage and later hardware validation. | Treating the canary as a served replacement or release gate before real-hardware decode verification is green. |

### Served Model Inventory And Harness Mapping
- `qwen-3.6-27B`: baseline llama.cpp served alias; multimodal; default normal-llama smoke target in `tests/test_e2e_smoke.py::LlamaCppSmokeTests`.
- `dflash-pflash-qwen3.6-27B`: primary served DFlash alias; text-only; current `.safetensors` draft artifact; default DFlash smoke target in `tests/test_e2e_smoke.py::DFlashSmokeTests` and default stress target via `STRESS_MODEL` in `tests/test_stress_dflash.py`.
- `dflash-native-qwen3.6-27B-canary`: dormant native DFlash canary alias; control-plane only; intentionally not the default target of smoke, stress, or PFlash harnesses.
- `pflash-qwen3.6-27B`: primary preserved standalone PFlash alias; text-only; default `MODEL` in `tests/test_pflash_pipeline.py`.
- `pflash-park-qwen3.6-27B`: preserved park/unpark PFlash alias; text-only; parameterized harness target only.
- Stale alias `dflash-pflash-qwen-27B` is not a valid current registry entry and must not reappear in harness defaults or release-gate docs.

## Required Verification Suite
- Semantic regression suite: `tests/test_dflash.py`.
- Live smoke suite: `tests/test_e2e_smoke.py`.
- Long-prompt compressor suite: `tests/test_pflash_pipeline.py`.
- Soak/leak/snapshot-growth suite: `tests/test_stress_dflash.py`.
- Manual runtime-isolation check: start normal `qwen-3.6-27B` without `/opt/dflash` in the runtime search path and verify it does not resolve symbols from `/opt/dflash`.
- Final image audit: the served runtime image no longer ships `/opt/dflash`, and startup env does not reference it via `LD_LIBRARY_PATH`, `LD_PRELOAD`, or equivalent wiring.
- Record five-run medians for TTFT and decode TPS on the same fixture before and after every cutover candidate.
- Hard floors: TTFT under 120s, decode TPS above 10, and restore speedup at or above 1.5x when the first turn TTFT is above 2s.
- Default regression budget unless a stricter baseline is recorded in Phase 0: median TTFT no worse than +20 percent, median decode TPS no worse than -10 percent, no unbounded RAM/disk growth across the stress run, and no repeated-call compressor VRAM drift above 256 MiB after steady state.

## Workstreams
- `A`: canonical DFlash decode on `TheTom`.
- `B`: DFlash `compact-full` persistence parity for `dflash-pflash-qwen3.6-27B`.
- `C`: preserved llama-side PFlash path, including its `.kv` slot contract, warm/cold behavior, and text-only reconstruction semantics.
- `A`, `B`, and `C` are not validation-independent. Changes to prompt layout, effective prompt semantics, or snapshot formats can invalidate multiple tracks at once.

## Phase 0
Scope Lock, Contract Capture, And Baseline

- Repos: `grimoire`, `tmp/spec-analysis/thetom-shallow/`, `tmp/spec-analysis/buun-shallow/`, `tmp/spec-analysis/bee-shallow/`, `tmp/spec-analysis/ggml-pr22105/`, `tmp/spec-analysis/lucebox-hub/`.
- Primary files: `src/grimoire/model_manager.py`, `src/grimoire/proxy/llama.py`, `src/grimoire/proxy/dflash.py`, `src/grimoire/dflash/daemon.py`, `src/grimoire/dflash/prefill.py`, `src/grimoire/dflash/prefix_cache.py`, `src/grimoire/dflash/session_kv.py`, `src/grimoire/dflash/snapshot_swap.py`, `src/grimoire/prompt/generic.py`, `etc/models.json`, `Dockerfile`, `lucebox/dflash/src/pflash_daemon.cpp`, `lucebox/dflash/src/qwen3_drafter.cpp`, `lucebox/dflash/src/qwen3_5_0p8b_drafter.h`, `lucebox/dflash/src/qwen3_5_0p8b_graph.cpp`, `lucebox/dflash/src/qwen3_5_0p8b_loader.cpp`.
- Deliverables: exact SHAs recorded for every upstream input, a required-vs-retired behavior matrix, a served model inventory, and baseline metrics for decode, restore, compression, TTFT, decode TPS, and snapshot-store growth.
- Deliverables: smoke/stress harness model IDs aligned with current registry entries or explicitly parameterized so they can be used as real gates.
- Exit criteria: `compact-full` is locked as required, the served `pflash-qwen3.6-27B` contract is locked as text-only, multimodal retirement for that model is reflected in live config, draft artifact transition policy is explicit, and the retirement target is confirmed as `A + B + C`.
- Decision gate: do not start implementation until every reference repo has an exact SHA recorded in this document.

## Phase 1
Canonical Base Bring-Up And `/opt/dflash` Isolation

- Repos: canonical working fork of `TheTom/llama-cpp-turboquant` plus `grimoire` for startup integration.
- Primary files: `CMakeLists.txt`, `include/llama.h`, `src/llama-context.cpp`, `src/llama-context.h`, `tools/server/server-context.cpp`, `tools/server/server.cpp`, `src/grimoire/model_manager.py`, `Dockerfile`.
- Deliverables: bootable canonical native branch, successful `llama-server` startup for `qwen-3.6-27B`, and a normal llama path that does not depend on `/opt/dflash` to resolve symbols.
- Current software-only status: Grimoire-side launch wiring and image/runtime-path isolation are covered locally by `tests/test_drop_in_blockers.py`, including normal llama startup env construction without `/opt/dflash` in `LD_LIBRARY_PATH`, park-model shim scoping via `LD_PRELOAD`, and runtime-image assertions that only preserved PFlash components land under `/opt/dflash`.
- Verification: startup and serve text on the target GPU, inspect runtime library resolution, and prove the non-PFlash llama path is anchored on canonical `TheTom` libraries only.
- Exit criteria: TurboQuant behavior remains intact, normal llama startup is decoupled from `/opt/dflash`, and any remaining `/opt/dflash` usage is explicitly limited to preserved PFlash components.
- Decision gate: stay on `TheTom` unless base bring-up exposes blocking incompatibilities that make a temporary selective `Bee` borrow materially faster overall.

## Phase 2
Canonical DFlash Decode MVP

- Repos: canonical `TheTom` fork as the target, with `tmp/spec-analysis/buun-shallow/`, `tmp/spec-analysis/bee-shallow/`, and `tmp/spec-analysis/ggml-pr22105/` as source refs.
- Primary native target files: `include/llama.h`, `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-model.cpp`, `src/llama-arch.cpp`, `src/llama-arch.h`, `src/models/dflash.cpp`, `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `common/speculative.cpp`, `tools/server/server-context.cpp`, `convert_hf_to_gguf.py`, `gguf-py/gguf/tensor_mapping.py`.
- Primary Grimoire files: `src/grimoire/proxy/dflash.py`, `src/grimoire/dflash/prefill.py`, `src/grimoire/prompt/generic.py`, `etc/models.json`.
- Current Grimoire checkpoint scope before native cutover: a dormant llama-server canary launch contract exists for `dflash-native-qwen3.6-27B-canary`, using Bee-style `--spec-type dflash`, `--spec-draft-model`, and `--spec-dflash-cross-ctx` flags. The local TheTom patch set now recognizes those launch flags, loads `dflash-draft` GGUF metadata/tensors, exposes guarded public DFlash model accessors plus `llama_model_share_tensors()`, auto-detects DFlash drafters via `LLM_ARCH_DFLASH_DRAFT`-guarded `llama_model_dflash_block_size()` so Bee's default `dflash_block_size=16` does not create false positives, adds Python-side GGUF conversion support for `DFlashDraftModel` plus the `dflash_fc`/`dflash_hidden_norm` tensor mapping, threads compile-safe DFlash runtime reservation parameters through `llama_context_params`, `llama_cparams`, context init, and `common_context_params_to_llama` (`dflash_n_slots`, `dflash_cross_ctx`), exposes the minimal `llama_set_dflash_n_slots()` / `llama_context::set_dflash_n_slots()` graph-reserve invalidation hook, adds compile-safe verifier gating fields/API/reuse-key plumbing (`dflash_verify_logits`, `dflash_verify_topk`, `llama_set_dflash_verify_logits()`), adds the `ggml_argmax_ext()` / `ggml_topk_ext()` front-end constructors needed for reduced-logit graph outputs, carries compile-safe reduced-logit result plumbing plus public argmax/top-K accessors (`t_logits_argmax`, `llama_get_logits_argmax*()`), wires the reduced-verifier consumer path across canonical `common/sampling.*`, `src/llama-context.*`, and `tools/server/server-context.cpp` with `llama_set_dflash_consume_reduced()`, per-ubatch raw-logit skipping, reduced-verify eligibility selection, and `dflash_sample_reduced_verify()` consumption for covered speculative-only verifier ubatches, and now adds a minimal native DFlash draft runtime on canonical TheTom: `LLM_ARCH_DFLASH_DRAFT` builder dispatch, `src/models/dflash_draft.cpp`, target hidden capture via `llama_set_dflash_capture()`, per-slot speculative sequence plumbing via `common_speculative_set_seq_id()`, drafter cross-data via `llama_set_cross_data_seq()`, server/speculative-simple tensor sharing before drafter-context creation, and acceptance-side hidden-history advancement in `common/speculative.cpp`. Focused canonical builds now succeed for `llama-speculative-simple`, `llama-server`, and `llama-speculative`, but this is still a minimal CPU-first slice and does not yet count as decode parity: no `compact-full` persistence, no Bee shared-drafter multi-slot batching/tree decode transplant, and no end-to-end model-backed semantic verification yet.
- Deliverables: buun DFlash core port, canonical decode path for the target Qwen models, GGUF DFlash draft load/convert path, and a Grimoire route that no longer depends on Lucebox for decode.
- Deliverables: prompt-layout and metadata fidelity across compression and reconstruction, including tool-call metadata, reasoning content, message boundaries, and protected blocks.
- Verification: `tests/test_dflash.py` now passes locally for prompt/block behavior, replay-style semantics, prefix/session snapshot helpers, protected-block preservation, and the DFlash Grimoire proxy path under fake-daemon/tokenizer harnesses. Remaining Phase 2 verification still required on real or equivalent GPU hardware: end-to-end text-only speculative decode with `parallel=1`, measured decode behavior, and confirmation that the served path no longer depends on the Lucebox decode daemon.
- Exit criteria: software-only verification is green, the native canary control-plane and minimal native DFlash runtime compile and pass local semantic tests, and the only remaining Phase 2 blockers are real-hardware decode verification and its measured correctness/performance checks. Full Phase 2 completion is still blocked until that hardware-backed decode pass is green.
- Decision gate: if buun core alone is insufficient, port only the minimum `Bee` runtime helpers needed for correctness. Do not adopt Bee wholesale unless the selective path fails.

## Phase 3
DFlash `compact-full` Persistence Implementation

- Repos: canonical `TheTom` fork for native save/load support and `grimoire` for persistence orchestration.
- Primary native files: `include/llama.h`, `src/llama-context.cpp`, `src/llama-context.h`, `common/speculative.cpp`, `tools/server/server-context.cpp`.
- Primary Grimoire files: `src/grimoire/proxy/dflash.py`, `src/grimoire/model_manager.py`, `src/grimoire/dflash/prefix_cache.py`, `src/grimoire/dflash/session_kv.py`, `src/grimoire/dflash/snapshot_swap.py`, `src/grimoire/dflash/daemon.py`, `etc/models.json`.
- Deliverables: native save/load for KV plus all recurrent/native state and `target_feat` equivalents, prompt-boundary inline snapshots, staging-slot semantics, Grimoire restore wiring, and restore/resume coverage matching the current `compact-full` contract.
- Deliverables: RAM-first snapshot writes, async disk mirroring, manifest persistence, TTL and budget enforcement, session hash invalidation, and prefix-cache compatibility even when the served config keeps it disabled by default.
- Verification: `tests/test_dflash.py`, `tests/test_e2e_smoke.py`, and `tests/test_stress_dflash.py`, plus explicit restart-resilience checks when disk-backed snapshots still exist.
- Exit criteria: save, stop, load, and continue works without semantic corruption, repeated-turn restore is stable, stale snapshots are invalidated instead of silently reused, and stress shows bounded RAM/disk growth.
- Decision gate: choose `slot-save + sidecar` only if it reproduces current semantics. Do not final-sign-off persistence until Phase 4 freezes the effective-prompt contract that produces the persisted `effective_ids`. If canonical native save/load cannot match the current contract, block Lucebox retirement and keep a split deployment.

## Phase 4
Preserved Llama-Side PFlash Parity And Effective-Prompt Freeze

- Repos: canonical `TheTom` fork as the runtime target, with `lucebox/dflash/` and `tmp/spec-analysis/lucebox-hub/` as source refs.
- Primary source files: `lucebox/dflash/src/pflash_daemon.cpp`, `lucebox/dflash/src/qwen3_drafter.h`, `lucebox/dflash/src/qwen3_drafter.cpp`, `lucebox/dflash/src/qwen3_5_0p8b_drafter.h`, `lucebox/dflash/src/qwen3_5_0p8b_graph.cpp`, `lucebox/dflash/src/qwen3_5_0p8b_loader.cpp`, `lucebox/dflash/src/flashprefill.h`, `lucebox/dflash/src/flashprefill.cpp`, `lucebox/dflash/src/flashprefill_q8.cpp`, `lucebox/dflash/src/internal.h`.
- Primary Grimoire files: `src/grimoire/proxy/llama.py`, `src/grimoire/model_manager.py`, `src/grimoire/dflash/daemon.py`, `src/grimoire/dflash/pflash_shim.c`, `etc/models.json`, `patches/slot-save-mtmd.patch`.
- Deliverables: preserved `Qwen3.5-0.8B Q8_0` compressor path, standalone `pflash_daemon`, raw `compress <path> <keep_x1000>` protocol, warm/cold split support, FIFO park/unpark for park models, `/slots/0` save/restore parity, the current token -> text -> message reconstruction path after compression, and all required `b4ed333` plus `e4d4e32` native fixes.
- Deliverables: explicit text-only served outcome for `pflash-qwen3.6-27B`, with no `mmproj` wiring carried into the migrated path.
- Verification: `tests/test_pflash_pipeline.py`, repeated cold and warm long-prompt runs, and repeated compressor calls to confirm no leak drift.
- Exit criteria: long-prompt compression works without Lucebox decode, repeated compress calls do not show unacceptable VRAM drift, warm/cold behavior is either reproduced or explicitly retired, reconstruction preserves the required message semantics, and the effective-prompt contract consumed by persistence is frozen for final Phase 3 sign-off.
- Decision gate: default to a standalone compressor service first. Switch to in-process only if standalone loses clearly on VRAM fragmentation, startup cost, or TTFT.

## Phase 5
Grimoire Integration And Artifact Cleanup

- Repos: `grimoire` plus the canonical native fork built in earlier phases.
- Primary files: `src/grimoire/dflash/prefill.py`, `src/grimoire/proxy/llama.py`, `src/grimoire/proxy/dflash.py`, `src/grimoire/model_manager.py`, `src/grimoire/prompt/generic.py`, `etc/models.json`, `Dockerfile`, `tests/test_dflash.py`, `tests/test_pflash_pipeline.py`, `tests/test_e2e_smoke.py`, `tests/test_stress_dflash.py`.
- Deliverables: block-aware long-prompt integration on the canonical stack, message reconstruction after compression, model registry cleanup, test-harness cleanup, and container/runtime wiring aligned with the final architecture.
- Deliverables: DFlash served config cleanup, explicit handling of the `.safetensors` -> GGUF draft transition, text-only PFlash config cleanup for `pflash-qwen3.6-27B`, and removal of hidden `/opt/dflash` dependencies from every served path.
- Verification: full required verification suite plus manual runtime-isolation checks.
- Exit criteria: Grimoire can run raw prompt -> optional PFlash compression -> optional restore -> generation on the canonical stack, test defaults map to real served model IDs, and no served path still relies on `/opt/dflash`.
- Decision gate: if real-session quality regresses, stop and retune keep ratio, tail protection, block policy, or prompt-layout handling before cutover.

## Phase 6
Optional Runtime Optimizations

- Repos: `grimoire`, canonical `TheTom` fork, and `tmp/spec-analysis/bee-shallow/` as the main optimization reference.
- Primary files: `src/grimoire/dflash/pflash_shim.c`, `src/grimoire/model_manager.py`, `src/grimoire/proxy/llama.py`, `tools/server/server-context.cpp`, `common/speculative.cpp`.
- Deliverables: only the optimizations that prove value, likely including VMM-based park/unpark, warm-turn detection plus KV slot reuse, or selective Bee runtime helpers.
- Verification: isolate each optimization, measure before and after on the same fixture, and keep only changes with proven correlation to the gain.
- Exit criteria: every retained optimization has a measurable win and passes isolated regression checks.
- Decision gate: if an optimization does not correlate cleanly to the observed gain, back it out instead of carrying it forward as migration debt.

## Phase 7
Cutover And Lucebox Retirement

- Repos: `grimoire` and the canonical native fork only for the served path.
- Primary files: `Dockerfile`, `etc/models.json`, `src/grimoire/model_manager.py`, `src/grimoire/proxy/llama.py`, `src/grimoire/proxy/dflash.py`, `src/grimoire/dflash/daemon.py`.
- Deliverables: collapsed runtime layout, final model configs, rollback instructions, production checklist, removal of `/opt/dflash` from the served runtime image, and quarantine or removal of the legacy Lucebox-specific served path.
- Verification: full required verification suite, explicit rollback test, explicit served-runtime `/opt/dflash` removal audit, and production-like runs for both the DFlash served path and the preserved llama-side PFlash served path.
- Exit criteria: required DFlash decode, required DFlash `compact-full` persistence, and required preserved PFlash all run on the canonical stack in a prod-like environment, rollback is tested, `/opt/dflash` is gone from the served runtime, and Lucebox is no longer needed for the served path.
- Decision gate: do not retire Lucebox until tracks `A`, `B`, and `C` are all green. If one track remains red, keep a temporary split deployment instead of forcing full retirement.

## Final Gates
- Gate 1: canonical base remains `TheTom + buun core + selective Bee`, not Bee-first, unless Phase 1 or 2 proves otherwise.
- Gate 2: DFlash decode parity is green for `dflash-pflash-qwen3.6-27B`.
- Gate 3: DFlash `compact-full` persistence parity is green, including restart resilience, staging-slot behavior, hash invalidation, and bounded snapshot-store growth.
- Gate 4: preserved llama-side PFlash parity is green, including the `.kv` slot contract, warm/cold policy, the reconstruction path, and all required native fixes.
- Gate 5: `pflash-qwen3.6-27B` is explicitly treated as a text-only served path, with no multimodal config or `mmproj` wiring in the served runtime.
- Gate 6: the served runtime is verified without `/opt/dflash` dependencies or `/opt/dflash` image content.
- Gate 7: every upstream input and served artifact version is pinned by exact SHA or artifact identifier in this document.
- Gate 8: Lucebox retirement is blocked until `A + B + C` are all green.
