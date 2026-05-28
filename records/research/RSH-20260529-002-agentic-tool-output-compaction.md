# RSH-20260529-002: Agentic Tool Output Compaction
Opened: 2026-05-29 00-54-55 KST
Recorded by agent: Codex

## Topic

Add an opencode-shaped tool-output compaction threshold so large tool results do not keep re-entering model context at full size.

## Change

- Added `compactToolOutputForContext()` with line and character thresholds.
- Applied tool-output compaction inside `compactApiMessagesForContext()` so persisted tool messages are bounded before normal requests are sent.
- Applied the same compaction to in-flight agentic loop tool results before the next model turn.
- Kept full tool results persisted in chat messages so the user-visible tool card still has the complete output.
- Added unit coverage for direct tool-output compaction and request-context tool-message compaction.

## Opencode Source Provenance

This slice is an adapted port; no opencode implementation code was copied.

- Referenced opencode's tool-output truncation service from `/home/yeowool/opencode/packages/opencode/src/tool/truncate.ts:16-27`, where output is bounded by max lines, max bytes, and direction.
- Referenced opencode's truncation result shape and full-output side channel from `/home/yeowool/opencode/packages/opencode/src/tool/truncate.ts:69-83` and `/home/yeowool/opencode/packages/opencode/src/tool/truncate.ts:86-142`. The webui does not write a filesystem output path, so full output stays in the persisted chat tool result instead.
- Referenced opencode's automatic tool wrapper truncation from `/home/yeowool/opencode/packages/opencode/src/tool/tool.ts:128-142`.
- Referenced opencode's MCP tool result truncation from `/home/yeowool/opencode/packages/opencode/src/session/tools.ts:153-194`.
- Referenced opencode's compaction-time tool-output cap from `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:37-38`, `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:406-409`, and `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:284`, `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:790-809`.

## Design Notes

- The compaction boundary is context-only, not UI-only. Users keep the full result in the chat surface while the model receives a concise tail preview.
- The existing `agenticMaxToolPreviewLines` setting now has runtime effect for agentic loop context.
- A fixed 8,000-character cap handles pathological single-line outputs where a line threshold alone would not help.

## Verification

- `npm run test:unit -- --run tests/unit/compaction.test.ts`
- `npm run build`
