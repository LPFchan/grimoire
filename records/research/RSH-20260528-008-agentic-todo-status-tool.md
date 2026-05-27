# RSH-20260528-008: Agentic Todo Status Tool Checkpoint
Opened: 2026-05-28 03-04-15 KST
Recorded by agent: gpt-5-codex

## Question

How should opencode's todo/status tool land in the llama-ui prototype without adding backend session tables or a product database?

## Implemented

This checkpoint adds an adapted `todowrite` provider tool:

- `AGENTIC_TODO_WRITE_TOOL_DEFINITION` exposes an OpenAI-compatible tool named `todowrite`.
- The schema mirrors opencode's todo shape:
  - `content`
  - `status`: `pending`, `in_progress`, `completed`, `cancelled`
  - `priority`: `high`, `medium`, `low`
- `parseTodoWriteArguments()` normalizes todo payloads.
- `formatTodoWriteResult()` returns pretty JSON, matching opencode's model-visible output shape.
- `AgenticStore` registers `todowrite` through the provider hook.
- Latest todos are kept as lightweight per-conversation reactive state via `agenticTodos(conversationId)`.
- Unit coverage verifies parsing, normalization, JSON output, and invalid payload rejection.

## Code-Level Shape

This is client-local state for the first prototype:

- no server table
- no database migration
- no cross-session persistence
- no separate todo panel yet
- no permission prompt yet, because provider tools do not have persisted permission keys in llama-ui's current permission store

The normal chat timeline still receives a tool result message, so todo updates remain inspectable in the existing tool-card path.

## Classification

Adapted port.

The tool contract and output are close to opencode. Persistence and permission handling are adapted to llama-ui's current architecture. A literal port would pull in opencode's `Todo.Service`, bus events, SQL table, and project/session ids.

## Follow-Up Route

Next subsystem: tool definitions/registry.

The new provider hook plus `question` and `todowrite` tools are enough pressure to consolidate internal tool definitions and registry behavior before adding `SKILL.md` support.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-todo.test.ts tests/unit/agentic-question.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-sections.test.ts`
- Result: 30 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
