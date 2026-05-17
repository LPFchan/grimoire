# Project Spec

**Project:** Grimoire — Model serving gateway with DFlash/PFlash speculative decoding
**Canonical repo:** `git@github.com:LPFchan/grimoire.git` (refactor branch)
**Operator:** LPFchan
**Last updated:** 2026-05-16

## Mission

Migrate the served DFlash/PFlash stack onto the canonical `TheTom/llama-cpp-turboquant` base. Preserve live runtime contracts first — porting files is not enough. Lucebox retirement is blocked until decode, DFlash `compact-full` persistence, and the preserved llama-side PFlash path are all green.

### Workstreams

- **A**: Canonical DFlash decode on TheTom.
- **B**: DFlash `compact-full` persistence parity for `dflash-pflash-qwen3.6-27B`.
- **C**: Preserved llama-side PFlash path, including its `.kv` slot contract, warm/cold behavior, and text-only reconstruction semantics.
- A, B, and C are not validation-independent. Changes to prompt layout, effective prompt semantics, or snapshot formats can invalidate multiple tracks at once.

## Required Decisions

These decisions were locked before Phase 1 and are not subject to renegotiation without a deliberate operator override:

- Retirement target is A + B + C. Decode-only cutover is not allowed.
- `compact-full` parity is required.
- DFlash remains text-only for this migration.
- Served `pflash-qwen3.6-27B` is text-only. Do not carry multimodal behavior or `mmproj` wiring for this model into the migrated stack.
- GGUF is the target end-state for DFlash draft artifacts, but the served artifact flip happens only after the canonical path is proven.
- Bee is a selective helper source only. Do not adopt Bee wholesale unless the selective path fails and the failure is documented.
- Standalone PFlash packaging is the default first target.
- The llama-side PFlash path keeps the current token -> text -> message reconstruction flow after compression. Direct prompt/token integration is out of scope.
- VMM-based park/unpark is preferred only if isolated measurement proves the gain. SIGTERM/page-cache reload behavior is fallback.
- Full removal of `/opt/dflash` is required for the final served runtime, not just normal-llama decoupling.

## Served Contracts

### Contract A: DFlash Decode
| Property | Value |
| --- | --- |
| Primary served model | `dflash-pflash-qwen3.6-27B` |
| Target artifact | `gguf/Qwen3.6-27B-Q4_K_M.gguf` |
| Draft artifact (current) | `dflash/Qwen3.6-27B-DFlash/model.safetensors` |
| Draft artifact (target) | GGUF (end-state) |
| Semantics | Text-only chat, `parallel=1`, `ctx-size=60000`, `max-effective-context=60000`, `budget=18`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, `fa-window=2048`, `<|im_end|>` stop |
| Prompt handling | Block-aware; compression boundaries/message metadata/prompt layout must survive `_prompt_layout_from_messages` -> compression -> reconstruction |
| Retirement | Final served decode path must not require the Lucebox decode daemon |

### Contract B: DFlash compact-full Persistence
- `snapshot-mode=compact-full` is required. Not optional.
- Session restore keyed by `conversation_id`, validated against effective prompt prefix hash.
- Transient staging slot contract, currently `snapshot-staging-slot=7`.
- Prompt-boundary inline snapshot behavior for reusable prefix work.
- RAM-first snapshot writes with asynchronous disk mirroring, manifest persistence, disk TTL, and RAM/disk budget enforcement.
- Stale-session invalidation on prompt-prefix mismatch. Silent reuse of stale snapshots is a hard failure.
- Support for prefix-cache semantics even when `prefix-cache-slots=0`.
- Preserve all state for correct continuation: KV, recurrent/native state, and `target_feat` equivalents.

### Contract C: Preserved Llama-Side PFlash
| Property | Value |
| --- | --- |
| Supported models | `pflash-qwen3.6-27B`, `pflash-park-qwen3.6-27B` |
| Daemon path | Standalone `pflash_daemon` until in-process replacement wins on measured TTFT/VRAM/startup |
| Warm/cold split | Preserved unless explicitly retired |
| Park/unpark | FIFO-based via `pflash_shim.so` |
| Slot contract | `/slots/0` save/restore; conversation-keyed `.kv` under `/dev/shm/grimoire-slots` |
| Compressor protocol | Raw `int32`: `compress <path> <keep_x1000>` |
| Drafter path | `Qwen3.5-0.8B Q8_0` |
| Reconstruction | Block-aware compression -> token -> text -> message. Direct prompt/token integration is out of scope. |
| Multimodal | Retired for this model |

### Contract D: Runtime Isolation
- End-state removes `/opt/dflash` entirely from served runtime image, startup environment, and runtime search path.
- Any remaining `/opt/dflash` dependency before cutover is temporary migration debt, limited to pre-cutover preservation work.
- The canonical non-PFlash llama path must be anchored on TheTom libraries only throughout the migration.

## Pinned Upstream Repos

| Repo | URL | Local path | SHA |
| --- | --- | --- | --- |
| TheTom/llama-cpp-turboquant | `https://github.com/TheTom/llama-cpp-turboquant.git` (branch `feature-turboquant-kv-cache-b9079-69d8e4b`) | `tmp/spec-analysis/thetom-shallow/` | `69d8e4be47243e83b3d0d71e932bc7aa61c644dc` |
| spiritbuun/buun-llama-cpp | `https://github.com/spiritbuun/buun-llama-cpp.git` | `tmp/spec-analysis/buun-shallow/` | `853eebdd02c2db4baf7bf781adadee6e7ce1d44e` |
| Anbeeld/beellama.cpp | `https://github.com/Anbeeld/beellama.cpp.git` | `tmp/spec-analysis/bee-shallow/` | `2b9aa77aa67ef0af7ee6eaa3d1f970215c7310fe` |
| ggml-org/llama.cpp PR 22105 | `https://github.com/ggml-org/llama.cpp.git` | `tmp/spec-analysis/ggml-pr22105/` | `320a6a44a5b1de6a074ba781e65f5fd79fb4051a` |
| Luce-Org/lucebox-hub | `https://github.com/Luce-Org/lucebox-hub.git` | `tmp/spec-analysis/lucebox-hub/` | `e5347801719ad7d45a3d7bd096e9e57778ce23ea` |

Local native source of truth: `lucebox/dflash/`. Local control-plane source of truth: `src/grimoire/`. Required native fixes to carry forward: all `b4ed333` fixes plus `e4d4e32` (qwen3_5_0p8b_graph.cpp leak fix).

## Served Model Inventory

| Alias | Backend | Draft/Drafter | Capabilities | Harness |
| --- | --- | --- | --- | --- |
| `qwen-3.6-27B` | llama | — | completion, multimodal | `test_e2e_smoke.py::LlamaCppSmokeTests` |
| `dflash-pflash-qwen3.6-27B` | dflash | `model.safetensors` | completion (text-only) | `test_e2e_smoke.py::DFlashSmokeTests`, `test_stress_dflash.py` |
| `dflash-native-qwen3.6-27B-canary` | llama (spec dflash) | `Qwen3.5-0.8B-Q8_0.gguf` | completion (text-only) | dormant — not a default harness target |
| `pflash-qwen3.6-27B` | llama (pflash) | `Qwen3.5-0.8B-Q8_0.gguf` | completion (text-only) | `test_pflash_pipeline.py` |
| `pflash-park-qwen3.6-27B` | llama (pflash+park) | `Qwen3.5-0.8B-Q8_0.gguf` | completion (text-only) | parameterized harness target only |

Stale alias `dflash-pflash-qwen-27B` is not a valid registry entry and must not reappear in harness defaults or release-gate docs.

## Required Verification Suite

| Suite | File | What it covers |
| --- | --- | --- |
| Semantic regression | `tests/test_dflash.py` | Prompt/block behavior, replay semantics, prefix/session helpers, protected blocks |
| Live smoke | `tests/test_e2e_smoke.py` | Basic chat completion on served models |
| Long-prompt compressor | `tests/test_pflash_pipeline.py` | PFlash compression pipeline |
| Soak/leak/snapshot-growth | `tests/test_stress_dflash.py` | Bounded RAM/disk growth |
| Runtime isolation | manual | Start `qwen-3.6-27B` without `/opt/dflash` in search path; verify no `/opt/dflash` symbols |
| Image audit | manual | Verify served runtime image does not ship `/opt/dflash` |

Hard floors: TTFT under 120s, decode TPS above 10, restore speedup >= 1.5x when first-turn TTFT > 2s.

Default regression budget: median TTFT no worse than +20%, median decode TPS no worse than -10%, no unbounded RAM/disk growth across stress run, no repeated-call compressor VRAM drift above 256 MiB after steady state.

## Required-vs-Retired Behavior Matrix

| Surface | Must Preserve | Explicitly Retired / Not A Gate Yet |
| --- | --- | --- |
| `dflash-pflash-qwen3.6-27B` served decode | Text-only chat semantics, `parallel=1`, `ctx-size=60000`, `max-effective-context=60000`, `budget=18`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, `fa-window=2048`, `<|im_end|>` stop-string behavior, block-aware prompt compression/reconstruction, `snapshot-mode=compact-full` | Lucebox decode daemon dependency. Early `.safetensors` -> GGUF draft cutover before proven. |
| `pflash-qwen3.6-27B` preserved PFlash path | Standalone `pflash_daemon`, raw `compress <path> <keep_x1000>` protocol, `Qwen3.5-0.8B Q8_0` drafter, token -> text -> message reconstruction | Multimodal serving and `mmproj` wiring |
| `pflash-park-qwen3.6-27B` preserved park path | FIFO park/unpark via `pflash_shim.so`, same text-only compression semantics as above | Global `/opt/dflash` library-path coupling |
| `qwen-3.6-27B` normal llama path | Canonical non-PFlash startup, multimodal config, resolves against TheTom libraries without `/opt/dflash` | Hidden fallback that only works because `/opt/dflash` is in runtime search path |
| `dflash-native-qwen3.6-27B-canary` native canary | Dormant native-control-plane launch contract | Treating canary as served replacement before real-hardware decode verification |

## Final Gates

1. Canonical base remains TheTom + buun core + selective Bee, not Bee-first.
2. DFlash decode parity is green for `dflash-pflash-qwen3.6-27B`.
3. DFlash `compact-full` persistence parity is green (restart resilience, staging-slot, hash invalidation, bounded snapshot-store growth).
4. Preserved llama-side PFlash parity is green (.kv slot contract, warm/cold, reconstruction, all required native fixes).
5. `pflash-qwen3.6-27B` is text-only, no multimodal config or `mmproj` in served runtime.
6. Served runtime is verified without `/opt/dflash` dependencies or image content.
7. Every upstream input and served artifact is pinned by exact SHA.
8. Lucebox retirement is blocked until A + B + C are all green.
