# RSH-20260529-004: Agentic Tool Event Normalization
Opened: 2026-05-29 01-03-25 KST
Recorded by agent: Codex

## Topic

Expose opencode-shaped normalized tool events over llama-ui's existing agentic message sections.

## Change

- Added `AGENTIC_TOOL_EVENT_TYPES` for `tool.input.started`, `tool.input.ended`, `tool.called`, `tool.success`, and `tool.failed`.
- Added `deriveAgenticToolEvents()` as a view-model utility over `deriveAgenticSections()`.
- Normalized tool event records now carry turn index, message id, call id, tool name, raw input, parsed input, output, and terminal result kind.
- Added unit coverage for completed and failed tool calls.

## Opencode Source Provenance

This slice is an adapted event-shape port; no opencode event bus or persistence code was copied.

- Referenced opencode's canonical tool event definitions from `/home/yeowool/opencode/packages/core/src/session-event.ts:213-307`.
- Referenced opencode's processor event publishing for tool input end and calls from `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:350-407`.
- Referenced opencode's processor event publishing for tool success and failure from `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:452-522`.
- Referenced opencode's TUI v2 sync reducer for tool events from `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/context/sync-v2.tsx:175-225`.

## Design Notes

- The webui implementation remains a derived view over existing OpenAI-shaped messages; it does not add event-sourced storage.
- Event ids are deterministic from message id, call id, and event type so downstream UI or product code can diff them reliably.
- Parsed input is best-effort JSON; invalid input is retained under a `raw` field.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts`
- `npm run build`
