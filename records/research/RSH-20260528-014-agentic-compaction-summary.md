# RSH-20260528-014: Agentic Compaction Summary
Opened: 2026-05-28 11-30-11 KST
Recorded by agent: Codex

## Question

The product needs an opencode-shaped compaction/summarization path without starting automatic overflow handling, sandbox orchestration, or backend persistence changes.

## Findings

- llama-ui had context telemetry and assistant timings, but no existing compaction boundary in chat history.
- opencode compaction keeps a structured Markdown summary and treats the latest compaction as a boundary for future context.
- llama-ui can model the same boundary client-side by storing a marked assistant summary message and rewriting outgoing request context around the latest marker.
- A manual first slice is safer than automatic compaction: it avoids surprising history changes and keeps overflow policy product-only for now.

## Change

- Added compaction utilities with an opencode-style Markdown summary template, stable hidden marker, and request-context collapse helper.
- Added manual compaction from the context ring: clicking it asks the current model for a summary and stores it as an assistant summary card.
- Added visible `Compacted context` cards for stored summary messages.
- Updated chat request construction so future requests preserve prior system prompts, inject the latest summary as system context, and keep only the tail after that summary.

## Upstream Shape

- Upstreamable: summary marker detection, visible summary cards, and request-context normalization behind a latest-summary boundary.
- Product-only for now: click-to-compact trigger, exact summary template, and policy for when automatic compaction should run.

## Verification

- `npm run test:unit -- --run tests/unit/compaction.test.ts`
- `npm run build`

## Follow-Up

- Add an explicit menu affordance or keyboard command if click-on-ring is too hidden.
- Consider automatic prompt-overflow recovery after the manual flow has been exercised.
- Add e2e coverage for compacting a long chat and confirming the next request contains summary plus tail.
