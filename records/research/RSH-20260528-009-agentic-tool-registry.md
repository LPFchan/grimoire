# RSH-20260528-009: Agentic Tool Registry Checkpoint
Opened: 2026-05-28 03-07-14 KST
Recorded by agent: gpt-5-codex

## Question

What registry layer is needed now that internal agentic tools are supplied by providers?

## Implemented

This checkpoint extracts provider registry behavior into pure helpers:

- `getAgenticProviderToolDefinitions(providers)`
- `findAgenticToolProvider(providers, toolName)`
- `mergeAgenticToolDefinitions(baseTools, providers)`

`AgenticStore` now delegates provider flattening, lookup, and duplicate-safe definition merging to those helpers.

Unit coverage verifies:

- provider definitions flatten in order
- providers can be found by tool name
- provider tools do not duplicate names already supplied by builtin/MCP/custom tools

## Code-Level Shape

The registry is deliberately small:

- no global singleton registry
- no separate backend service
- no plugin loading
- no persistence

It is just the normalized utility boundary needed by the client-side agentic loop. That keeps the code upstream-shaped while making downstream product providers easier to compose.

## Classification

Adapted port.

opencode's tool registry handles initialization, permissions, agents, and backend services. llama-ui already has multiple tool sources and a client-side execution loop. The useful port is a thin deterministic merge/lookup layer.

## Follow-Up Route

Next subsystem: `SKILL.md` support.

Skill support should plug into this registry as a provider only if it needs tool definitions. The first implementation should focus on discovery/loading and prompt-context injection rather than executable skill tooling.

## Opencode Source Provenance

This was an adapted registry-boundary port; no opencode registry implementation was copied.

- Registry interface shape referenced `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:73-80`.
- Builtin and custom tool collection referenced `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:117-134`, `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:137-220`, and `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:225-269`.
- Tool definition projection and dynamic description hooks referenced `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:282-360`.
- Execution context and result fields referenced `/home/yeowool/opencode/packages/opencode/src/tool/tool.ts:34-63`.
- Direct plugin/filesystem loading from `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:199-220` was explicitly not ported because the webui registry slice is browser-local.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-tool-registry.test.ts tests/unit/agentic-todo.test.ts tests/unit/agentic-question.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-sections.test.ts`
- Result: 33 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
