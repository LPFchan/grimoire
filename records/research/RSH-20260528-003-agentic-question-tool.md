# RSH-20260528-003: Agentic Question Tool Checkpoint
Opened: 2026-05-28 02-45-12 KST
Recorded by agent: gpt-5-codex

## Question

How should the second opencode port candidate, the `question` tool, land in llama-ui without introducing opencode's backend runtime or replacing llama-ui's existing MCP/tool lifecycle?

## Implemented

The checkpoint adds an internal `question` tool to the llama-ui agentic loop:

- `AGENTIC_QUESTION_TOOL_DEFINITION` exposes an OpenAI-compatible tool schema named `question`.
- `parseQuestionToolArguments()` accepts opencode-style arguments:
  - `questions[]`
  - `header`
  - `question`
  - `options[]`
  - `multiple`
  - `custom`
- `formatQuestionToolResult()` returns the model-facing answer text in the same shape as opencode:
  - `User has answered your questions: "...question..."="answer". You can now continue with the user's answers in mind.`
- `AgenticStore` now tracks pending question requests per conversation and resolves them into tool result messages.
- `ChatMessageActionCardQuestionRequest.svelte` renders the blocking user prompt with option selection, optional custom answers, submit, and dismiss.
- `ChatMessageAgenticContent.svelte` displays pending question prompts alongside the existing permission and continue action cards.

## Code-Level Shape

The tool is handled as an internal agentic tool, not as an MCP server tool:

- It is appended to the tool definitions sent to the model.
- It bypasses permission prompts because the tool call itself is a request for direct user input, not external execution.
- It still produces a normal tool result message and contributes to tool-call timing stats.
- It uses the same pending-request resolver pattern already used by llama-ui permissions and turn-limit continue prompts.

The implementation deliberately does not add:

- a backend session processor
- sandbox execution
- MCP account/auth handling
- persistent question history outside the normal message tree

## Why This Is An Adapted Port

The protocol and model-facing result are close to a literal opencode port, but the runtime is adapted.

opencode's `question` tool lives inside its server-side tool registry and session processor. llama-ui already has a client-side agentic loop, OpenAI-compatible tool messages, and Svelte action cards for blocking user decisions. Porting the backend processor would create framework and lifecycle mismatch. The clean upstream-shaped piece is the structured question schema plus a client-side pending-request protocol.

## Open Questions

- Whether the internal `question` tool should be shown in the tool enable/disable menu or remain always available while agentic mode is enabled.
- Whether a future upstream PR should expose a generic internal tool-provider hook so `question` does not need bespoke insertion into the agentic loop.
- Whether dismiss should be model-visible as a soft refusal or as an error-like tool result.

## Follow-Up Route

Next subsystem: permission rules refinement.

The current implementation should compose with llama-ui's existing allow once / always / always server / deny surface; the next checkpoint should tighten rule storage, internal-tool bypasses, and presentation language rather than replacing llama-ui's MCP permission model.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-question.test.ts tests/unit/agentic-sections.test.ts`
- Result: 21 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
