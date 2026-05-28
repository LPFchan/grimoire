# RSH-20260529-003: Agentic Auto Compaction
Opened: 2026-05-29 00-59-18 KST
Recorded by agent: Codex

## Topic

Extend the manual compaction path into a conservative automatic compaction trigger based on latest model context usage.

## Change

- Added timing helpers that compute used context tokens from `prompt_n + cache_n + predicted_n`.
- Added an automatic compaction threshold at 85% of known context.
- Triggered auto-compaction after completed normal and agentic responses.
- Reused the existing compaction summary card and summary prompt.
- Made auto-compaction failures non-modal: they are logged but do not interrupt the chat flow.

## Opencode Source Provenance

This slice is an adapted port; no opencode implementation code was copied.

- Referenced opencode's overflow threshold model from `/home/yeowool/opencode/packages/opencode/src/session/overflow.ts:6-31`, where usable context is derived from model limits and reserved output.
- Referenced opencode's processor overflow handoff from `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:610-615` and `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:751-758`.
- Referenced opencode's compaction creation and event reason split from `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:584-614` and `/home/yeowool/opencode/packages/core/src/session-event.ts:332-363`.
- Referenced opencode's auto-continue behavior from `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:477-558`. The webui slice does not synthesize a follow-up prompt; it only creates the summary boundary after a completed response.
- Referenced opencode's summary anchoring and tail-selection path from `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:123-140` and `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:245-294`.

## Design Notes

- The browser product does not have opencode's provider-side overflow exceptions or backend event bus, so the first auto trigger uses completed response timings rather than catching an overflow failure.
- The trigger is intentionally conservative at 85% and uses existing llama-server/model context telemetry.
- Auto summaries use the same visible card as manual compaction, so the history mutation remains user-visible.

## Verification

- `npm run test:unit -- --run tests/unit/compaction.test.ts`
- `npm run build`
