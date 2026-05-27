# RSH-20260528-005: Agentic Tool Lifecycle Polish Checkpoint
Opened: 2026-05-28 02-52-43 KST
Recorded by agent: gpt-5-codex

## Question

What should "tool lifecycle polish" add after the first lifecycle normalization checkpoint without changing llama-ui's persisted message model?

## Finding

The missing piece was terminal result fidelity.

The first lifecycle checkpoint normalized high-level states such as `tool_running`, `tool_success`, and `tool_error`. That is enough for orchestration, but too coarse for polished rendering: denied tools, interrupted tools, dismissed question prompts, and generic execution errors all looked like the same terminal error state.

## Implemented

This checkpoint adds a terminal result-kind layer:

- `AgenticToolResultKind`
  - `success`
  - `error`
  - `denied`
  - `interrupted`
  - `dismissed`
- `deriveToolResultKind()` maps model-visible tool result text into the normalized kind.
- `AgenticSection` and `AgenticLifecycleEvent` now carry `toolResultKind`.
- `deriveToolLifecycleState()` still maps all non-success terminal kinds to `TOOL_ERROR`, preserving the high-level lifecycle contract.
- Unit coverage now verifies generic errors plus denied, interrupted, and dismissed terminal results.

## Code-Level Shape

The implementation remains a view-model layer:

- no schema migration
- no new persisted message fields
- no new backend event store
- no change to OpenAI-compatible message payloads

This gives the next frontend rendering checkpoint enough state to render clearer labels and visual treatment without re-parsing arbitrary tool output in Svelte components.

## Classification

Adapted port.

opencode has more explicit event and part status surfaces around tool calls. llama-ui already stores assistant tool calls and tool result messages; the useful port is a normalized terminal-result vocabulary derived from those messages.

## Follow-Up Route

Next subsystem: frontend tool rendering.

Use `toolResultKind` to improve card subtitles, icon/color treatment, and result copy for denied/interrupted/dismissed tool calls while keeping the existing collapsible card structure.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-question.test.ts`
- Result: 26 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
