# RSH-20260528-007: Agentic Tool Provider Hook Checkpoint
Opened: 2026-05-28 03-00-48 KST
Recorded by agent: gpt-5-codex

## Question

How should downstream/internal tools attach to llama-ui's agentic loop without adding one-off branches for every opencode-style subsystem?

## Implemented

This checkpoint adds a small agentic tool-provider hook:

- `AgenticToolProvider`
  - `id`
  - `tools`
  - optional `requiresPermission(toolName)`
  - optional `executeTool(toolCall, context)`
- `AgenticToolExecutionContext`
  - `conversationId`
  - `toolCall`
  - `signal`
- `AgenticToolExecutionResult`
  - `content`
  - optional `isError`
- `agenticRegisterToolProvider(provider)` public registration function.

The existing internal `question` tool is now registered as the first provider instead of being hard-coded into the execution loop.

## Code-Level Shape

The provider hook sits between OpenAI tool-call detection and the existing builtin/MCP execution branches:

1. collect enabled builtin/MCP/custom tool definitions
2. merge provider tool definitions without duplicating names
3. find a provider by called tool name
4. ask permission only if the provider requires it
5. execute provider tools before falling back to builtin/MCP execution

This keeps question-tool behavior the same while making future product-only or upstreamable internal tools less invasive.

## Classification

Adapted port.

opencode has a registry/service-oriented tool harness. llama-ui already has builtin, MCP, and custom tool sources, so a literal registry port would be too much. The useful shape is the extension hook: tool definitions plus optional execution and permission policy.

## Follow-Up Route

Next subsystem: todo/status tool.

The todo/status prototype should be implemented as a provider tool instead of another bespoke branch in `AgenticStore`.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-question.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-sections.test.ts`
- Result: 26 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
