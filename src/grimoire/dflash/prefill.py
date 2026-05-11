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
) -> tuple:
    """Compress prompt if it exceeds the threshold.

    Protects the system prompt (head) and the last turns that fit within
    tail_budget (tail), compressing only the middle conversation history.

    Args:
        prompt_ids: Original prompt token IDs
        daemon: DflashDaemon instance
        config: PrefillConfig
        boundaries: List of token positions marking message boundaries,
            derived from _dflash_prefix_boundaries(). Each element is an int
            (token offset) where a message ends.

    Returns:
        (compressed_ids, compression_fired) tuple. If compression didn't
        fire, returns the original prompt_ids unchanged.
    """
    if not config.enabled or config.drafter_path is None:
        return prompt_ids, False

    if len(prompt_ids) < config.threshold:
        return prompt_ids, False

    compress_start = 0
    compress_end = len(prompt_ids)

    if boundaries and len(boundaries) >= 2:
        # Head: system prompt ends at first boundary.
        compress_start = boundaries[0]

        # Tail: walk backwards from the end, protecting whole turns until
        # tail_budget is consumed. Each turn = boundaries[i] - boundaries[i-1].
        tail_so_far = 0
        for i in range(len(boundaries) - 1, 0, -1):
            turn_len = boundaries[i] - boundaries[i - 1]
            if tail_so_far + turn_len > config.tail_budget:
                compress_end = boundaries[i - 1]
                break
            tail_so_far += turn_len
        else:
            # All turns fit in tail_budget — nothing to compress in the middle.
            compress_end = boundaries[0]

    elif boundaries and len(boundaries) == 1:
        # Only system message boundary — protect head, compress everything after.
        compress_start = boundaries[0]

    # If the middle portion is too small to benefit from compression, skip.
    middle_len = compress_end - compress_start
    if middle_len < 1024:
        return prompt_ids, False

    head_len = compress_start
    tail_len = len(prompt_ids) - compress_end

    logger.info(
        f"pflash compressing {middle_len} middle tokens "
        f"(head={head_len} tail={tail_len} keep={config.keep_ratio})"
    )

    t0 = time.monotonic()

    # Compress only the middle portion.
    middle = prompt_ids[compress_start:compress_end]
    loop = asyncio.get_event_loop()
    compressed_middle = await loop.run_in_executor(
        None,
        lambda: daemon.compress(
            middle,
            drafter_path=config.drafter_path,
            keep_ratio=config.keep_ratio,
        ),
    )

    # Reattach protected head and tail.
    compressed = prompt_ids[:compress_start] + compressed_middle + prompt_ids[compress_end:]

    elapsed = time.monotonic() - t0
    logger.info(
        f"pflash middle {middle_len} -> {len(compressed_middle)} tokens "
        f"({middle_len / max(len(compressed_middle), 1):.1f}x, "
        f"total {len(prompt_ids)} -> {len(compressed)}) in {elapsed:.1f}s"
    )

    return compressed, True
