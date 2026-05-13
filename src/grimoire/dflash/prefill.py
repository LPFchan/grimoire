"""PFlash speculative prefill for DFlash models.

For long-context prompts (>prefill_threshold tokens), a small drafter model
scores per-token importance and compresses the prompt before target prefill.
This reduces prefill time by ~10x for 64K-128K prompts.

The gateway detects long prompts, calls compress() to get the reduced token
list, then feeds that to the daemon's generate() instead of the full prompt.
"""

import logging
import time
from dataclasses import dataclass, field
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
    """

    enabled: bool = False
    threshold: int = 32000
    keep_ratio: float = 0.05
    drafter_path: Optional[str] = None


async def maybe_compress(
    prompt_ids: list,
    daemon,
    config: PrefillConfig,
) -> tuple:
    """Compress prompt if it exceeds the threshold.

    Args:
        prompt_ids: Original prompt token IDs
        daemon: DflashDaemon instance
        config: PrefillConfig

    Returns:
        (compressed_ids, compression_fired) tuple. If compression didn't
        fire, returns the original prompt_ids unchanged.
    """
    if not config.enabled or config.drafter_path is None:
        return prompt_ids, False

    if len(prompt_ids) < config.threshold:
        return prompt_ids, False

    logger.info(
        f"pflash compressing {len(prompt_ids)} tokens "
        f"(threshold={config.threshold}, keep={config.keep_ratio})"
    )

    t0 = time.monotonic()

    # Run compression in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    compressed = await loop.run_in_executor(
        None,
        lambda: daemon.compress(
            prompt_ids,
            drafter_path=config.drafter_path,
            keep_ratio=config.keep_ratio,
        ),
    )

    elapsed = time.monotonic() - t0
    logger.info(
        f"pflash compressed {len(prompt_ids)} -> {len(compressed)} tokens "
        f"({len(prompt_ids) / max(len(compressed), 1):.1f}x) in {elapsed:.1f}s"
    )

    return compressed, True
