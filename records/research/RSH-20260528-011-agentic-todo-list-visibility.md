# RSH-20260528-011: Agentic Todo List Visibility
Opened: 2026-05-28 10-29-47 KST
Recorded by agent: Codex

## Question

The `todowrite` internal tool could be called and completed, but users only saw the raw tool result if they expanded the tool card. The intended opencode-style behavior needs a visible status list in the chat flow.

## Findings

- `todowrite` already updated `agenticTodos(conversationId)` during live execution.
- No UI consumed `agenticTodos()`, so the visible chat surface did not show the current list.
- Persisted tool results already contain the todo array as stable JSON, which is a better source for rendering completed or reloaded conversations than ephemeral store state alone.

## Change

- Added a display parser for persisted `todowrite` result JSON.
- Added `ChatMessageAgenticTodoList.svelte` as a compact todo/status panel.
- Rendered the latest persisted `todowrite` result in `ChatMessageAgenticContent`, falling back to live store state while execution is active.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-todo.test.ts`
- `npm run build`

## Follow-Up

- Consider adding e2e coverage that forces a `todowrite` tool call and asserts the status panel appears.
