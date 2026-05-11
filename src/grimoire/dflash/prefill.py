"""PFlash speculative prefill for DFlash models.

For long-context prompts (>prefill_threshold tokens), a small drafter model
scores per-token importance and compresses the prompt before target prefill.
This reduces prefill time by ~10x for 64K-128K prompts.

The gateway detects long prompts, calls compress() to get the reduced token
list, then feeds that to the daemon's generate() instead of the full prompt.

Compression protects the system prompt (head) and the last N turns that fit
within tail_budget tokens (tail), compressing only the middle conversation
history. Boundaries are passed from the caller as token positions that mark
message boundaries.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PrefillConfig:
    """PFlash compression settings.

    Args:
        enabled: Whether PFlash compression is active
        threshold: Token count above which compression triggers
        keep_ratio: Fraction of source tokens to keep (0.01-1.0)
        drafter_path: Path to the drafter GGUF model
        tail_budget: Max tokens to protect at the tail (walks backwards from
            the end, protecting whole turns until the budget is consumed).
    """

    enabled: bool = False
    threshold: int = 32000
    keep_ratio: float = 0.05
    drafter_path: Optional[str] = None
    tail_budget: int = 16000


async def maybe_compress(
    prompt_ids: list,
    daemon,
    config: PrefillConfig,
    boundaries: Optional[list] = None,
    protected_ranges: Optional[list] = None,
) -> tuple:
    """Compress prompt if it exceeds the threshold.

    Protects the system prompt (head) and the last turns that fit within
    tail_budget (tail), compressing only the middle conversation history.
    Additional protected_ranges (e.g., obsidian_read-note tool outputs) are
    also exempted from compression.

    Args:
        prompt_ids: Original prompt token IDs
        daemon: DflashDaemon instance
        config: PrefillConfig
        boundaries: List of token positions marking message boundaries,
            derived from _dflash_prefix_boundaries(). Each element is an int
            (token offset) where a message ends.
        protected_ranges: List of (start, end) tuples marking token ranges
            that must not be compressed (e.g., tool outputs).

    Returns:
        (compressed_ids, compression_fired) tuple. If compression didn't
        fire, returns the original prompt_ids unchanged.
    """
    if not config.enabled or config.drafter_path is None:
        return prompt_ids, False

    if len(prompt_ids) < config.threshold:
        return prompt_ids, False

    n = len(prompt_ids)
    # Build list of protected intervals: [start, end) — never compressed.
    # Start with head = [0, n) meaning "protect everything" as a default.
    head_end = n

    if boundaries and len(boundaries) >= 2:
        # Head: system prompt + first user message (opencode compaction summary).
        head_end = boundaries[1]

        # Tail: walk backwards from the end, protecting whole turns until
        # tail_budget is consumed. Each turn = boundaries[i] - boundaries[i-1].
        # If a turn exceeds the budget, it's still protected (overshoot).
        tail_so_far = 0
        tail_start = n
        for i in range(len(boundaries) - 1, 1, -1):
            turn_len = boundaries[i] - boundaries[i - 1]
            if tail_so_far + turn_len > config.tail_budget:
                tail_start = boundaries[i - 1]
                break
            tail_so_far += turn_len
            tail_start = boundaries[i - 1]
        else:
            # All turns fit in tail_budget — nothing to compress in the middle.
            tail_start = boundaries[1]

        # Prevent head + tail from merging (which would leave nothing to compress).
        if tail_start <= head_end:
            tail_start = head_end

        protected: list[list[int]] = [[0, head_end], [tail_start, n]]

    elif boundaries and len(boundaries) >= 1:
        # Only system boundary (or system + 1 msg) — protect head.
        head_end = boundaries[0]
        protected = [[0, head_end]]

    else:
        # No boundaries — protect nothing (entire prompt is compressible).
        protected = []

    # Merge additional protected ranges (e.g., obsidian_read-note outputs).
    if protected_ranges:
        for ps, pe in protected_ranges:
            if 0 <= ps < pe <= n:
                protected.append([ps, pe])

    # Sort by start, then merge overlapping intervals.
    if protected:
        protected.sort()
        merged = [protected[0][:]]
        for s, e in protected[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
    else:
        merged = []

    # Find compressible gaps between protected intervals (and edges of prompt).
    gaps = []
    prev_end = 0
    for s, e in merged:
        if s - prev_end >= 256:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    # Gap after last protected interval to end of prompt.
    if n - prev_end >= 256:
        gaps.append((prev_end, n))

    total_compressible = sum(e - s for s, e in gaps)
    if total_compressible < 1024:
        return prompt_ids, False

    head_tail_len = n - total_compressible
    logger.info(
        f"pflash compressing {total_compressible} middle tokens "
        f"(head+tail={head_tail_len} keep={config.keep_ratio})"
    )

    t0 = time.monotonic()
    loop = asyncio.get_event_loop()

    # Compress each gap independently and rebuild the prompt.
    compressed = []
    for i, (gstart, gend) in enumerate(gaps):
        # Append the protected segment before this gap.
        if i == 0:
            compressed.extend(prompt_ids[:gstart])
        else:
            compressed.extend(prompt_ids[gaps[i - 1][1]:gstart])

        gap = prompt_ids[gstart:gend]
        compressed_gap = await loop.run_in_executor(
            None,
            lambda g=gap: daemon.compress(
                g,
                drafter_path=config.drafter_path,
                keep_ratio=config.keep_ratio,
            ),
        )
        compressed.extend(compressed_gap)

    # Append the final protected tail segment.
    if gaps:
        compressed.extend(prompt_ids[gaps[-1][1]:])
    else:
        compressed = prompt_ids[:]

    elapsed = time.monotonic() - t0
    logger.info(
        f"pflash middle {total_compressible} -> {len(compressed) - head_tail_len} tokens "
        f"({total_compressible / max(len(compressed) - head_tail_len, 1):.1f}x, "
        f"total {n} -> {len(compressed)}) in {elapsed:.1f}s"
    )

    return compressed, True
