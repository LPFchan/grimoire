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

## Verification

- `npm run build`

## Follow-Up

- Once compaction/summarization exists, wire this indicator to compaction thresholds and warnings instead of only passive utilization display.
