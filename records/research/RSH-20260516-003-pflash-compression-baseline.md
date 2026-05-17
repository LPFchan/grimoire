# Preserved PFlash Compression Baseline (2026-05-16)

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Method

Isolated GPU 1 container, `pflash-qwen3.6-27B` served model. Single-message and multi-turn probes via the gateway API with debug instrumentation.

## Results

| Probe type | Raw tokens | Compressed | Ratio | Wall-clock | VRAM |
| --- | --- | --- | --- | --- | --- |
| Single-message (20K chars) | 5,878 | — | — | ~14.8s | — |
| Single-message (above threshold) | 40,746 | 40,746 | 1.00x | ~46.5s | — |
| Multi-turn (13 msgs, ~150K chars) | 43,584 | 23,296 | 1.87x | ~21.0s | ~20.4 GiB |

The single-message probes did not fire compression because a one-message prompt leaves no compressible middle span after head/tail protection. The multi-turn probe produced the first explicit live ratio.

## Implications

- PFlash compression is functional on the isolated container.
- VRAM is stable at ~20.4 GiB during compression — no leak drift observed.
- The 1.87x ratio on a 13-message prompt is the baseline for regression testing.
- Harness updates added: `GRIMOIRE_LONG_PROMPT_MIN_CHARS` / `GRIMOIRE_LONG_PROMPT_MAX_CHARS`, `GRIMOIRE_SMOKE_CONTAINER`, `GPU_INDEX`.
