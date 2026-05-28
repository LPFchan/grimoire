# RSH-20260528-012: Agentic Context Budget Indicator
Opened: 2026-05-28 11-13-25 KST
Recorded by agent: Codex

## Question

Context budget was only visible while generation stats were shown. The desired agentic UI needs an opencode-style persistent circular context indicator beside the model selector, with a hover label like `Context: used / total`.

## Findings

- Live generation already exposes `activeProcessingState().contextUsed` and `contextTotal`.
- Completed assistant messages persist enough timing data to reconstruct last known context usage from `prompt_n + cache_n + predicted_n`.
- Model context size is available through router model props via `selectedModelContextSize()`, with server props `contextSize()` as fallback.
- The chat form action row is the right attachment point because it already owns the model selector and sits in the first-viewport composer controls.

## Change

- Added `ChatFormActionContextBudget.svelte`.
- Rendered it immediately left of `ChatFormActionModels`.
- The indicator uses live processing state while generating and falls back to the latest assistant timing data when idle.
- Tooltip shows `Context: <used> / <total> (<percent>%)`.
- Ring color changes at 70% and 90% thresholds.

## Opencode Source Provenance

This was an adapted visual/behavior reference; no opencode TUI code was copied.

- Persistent context usage display referenced opencode's sidebar context plugin at `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/feature-plugins/sidebar/context.tsx:16-35` and rendering at `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/feature-plugins/sidebar/context.tsx:38-44`.
- Subagent footer context/cost usage referenced `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/subagent-footer.tsx:35-53` and display at `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/routes/session/subagent-footer.tsx:83-93`.
- Manual compaction affordance was informed by opencode's compaction command binding at `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/config/keybind.ts:95` and command id at `/home/yeowool/opencode/packages/opencode/src/cli/cmd/tui/config/keybind.ts:292`; the webui later moved compaction into the popup button per operator direction.
- Overflow math for future thresholding referenced `/home/yeowool/opencode/packages/opencode/src/session/overflow.ts:6-31`, but current webui display uses llama-server timings/model context telemetry.

## Verification

- `npm run build`

## Follow-Up

- Once compaction/summarization exists, wire this indicator to compaction thresholds and warnings instead of only passive utilization display.
