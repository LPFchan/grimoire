# RSH-20260528-006: Agentic Tool Rendering Checkpoint
Opened: 2026-05-28 02-55-51 KST
Recorded by agent: gpt-5-codex

## Question

How should the frontend consume the normalized tool lifecycle data without redesigning llama-ui's chat renderer?

## Implemented

This checkpoint updates the existing agentic tool cards in `ChatMessageAgenticContent.svelte`:

- Completed tool calls render with a success icon and `completed` subtitle.
- Generic tool failures render with an alert icon and `failed` subtitle.
- Denied tool calls render with a terminal icon and `denied` subtitle.
- Interrupted tool calls render with a terminal icon and `interrupted` subtitle.
- Dismissed question prompts render with a terminal icon and `dismissed` subtitle.
- Non-success result panels use a destructive-tinted border/background.
- Pending and streaming states keep the existing spinner behavior.

## Code-Level Shape

The renderer uses `section.toolResultKind`, added by the previous lifecycle checkpoint, instead of parsing result text inside Svelte. The display remains upstream-shaped:

- no new card hierarchy
- no nested cards
- no persistence changes
- no new global state
- no replacement of `CollapsibleContentBlock`

## Classification

Adapted port.

opencode's frontend gives tool calls clearer status treatment. llama-ui already has a collapsible tool-call path, so the useful port is status-specific rendering on top of the existing component structure.

## Follow-Up Route

Next subsystem: external tool-provider hook.

The renderer now expects normalized lifecycle/result metadata. The next useful boundary is letting internal/downstream providers add tools without special-casing every tool in `AgenticStore`.

## Opencode Source Provenance

This was an adapted frontend behavior port; no Solid/opencode component code was copied.

- Generic tool selection and specialized renderer dispatch referenced `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx:1642-1719`.
- Generic expanded/collapsed tool rendering referenced `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx:1734-1768`.
- Inline/block tool treatment referenced `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx:1772-1876`.
- Status/error color decisions referenced tool-state access in `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx:1790-1810` and `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx:1870-1876`.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-question.test.ts`
- Result: 26 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
