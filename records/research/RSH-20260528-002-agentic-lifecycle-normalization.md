# RSH-20260528-002: Agentic Lifecycle Normalization Checkpoint
Opened: 2026-05-28 02-33-06 KST
Recorded by agent: gpt-5-codex

## Question

How should the first opencode port candidate, agent event/lifecycle normalization, land in llama-ui without replacing the existing client-side agent loop?

## Implemented

The checkpoint adds a normalized lifecycle layer over the existing llama-ui agentic sections:

- `AgenticLifecycleState` enum for stable lifecycle states:
  - `text`
  - `reasoning`
  - `reasoning_pending`
  - `tool_input_streaming`
  - `tool_pending`
  - `tool_running`
  - `tool_success`
  - `tool_error`
- `AgenticSection` now carries normalized state, turn index, message id, and tool call id.
- `deriveAgenticLifecycleEvents()` exposes a compact event view for downstream product code without introducing a new persistence model.
- `ChatMessageAgenticContent.svelte` now groups turns by `section.turnIndex` instead of inferring turn boundaries from display order.
- Unit coverage was extended for lifecycle state mapping, error tool results, streaming tool input, and stable event ids.

## Code-Level Shape

The implementation stays upstream-shaped:

- Existing OpenAI-compatible message persistence remains unchanged.
- Existing `deriveAgenticSections()` callers continue to work; they receive additional fields.
- The new lifecycle API is a view-model utility, not a backend event store or opencode `MessageV2` import.
- The renderer consumes normalized turn indexes, which makes later `question` and permission-blocking UI less dependent on heuristic section grouping.

## Why This Is An Adapted Port

opencode has a richer event model in `packages/core/src/session-event.ts` and a processor-driven lifecycle in `packages/opencode/src/session/processor.ts`. A literal port would bring Effect services, bus events, storage assumptions, AI SDK event names, and backend runtime coupling.

llama-ui already has the important raw signals: reasoning chunks, text chunks, streaming tool-call deltas, persisted tool calls, tool result messages, permissions, and MCP execution. The useful port is the normalized state vocabulary and event shape, not opencode's runtime.

## Follow-Up Route

Next subsystem: `question` tool.

The lifecycle state/event layer should give the `question` tool a clean way to render as a tool call that blocks for structured user input, then returns a model-visible tool result.

## Verification

Baseline before edits:

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts`
- Result: 15 tests passed.

After edits:

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts`
- Result: 17 tests passed.
