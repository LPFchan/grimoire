"""PFlash speculative prefill for DFlash models.

For long-context prompts (>prefill_threshold tokens), a small drafter model
scores per-token importance and compresses the prompt before target prefill.
This reduces prefill time by ~10x for 64K-128K prompts.

The gateway detects long prompts, calls compress() to get the reduced token
list, then feeds that to the daemon's generate() instead of the full prompt.

Compression is block-aware: the caller supplies a prompt manifest with stable
block ids, token spans, and metadata. PFlash preserves head/tail protection and
protected tool blocks, but only compresses whole blocks so metadata survives the
raw -> effective prompt transform.
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


@dataclass(frozen=True)
class PromptBlock:
    """One logical prompt block in raw token space."""

    block_id: str
    index: int
    start: int
    end: int
    role: str
    kind: str
    message_start: int
    message_end: int
    protected: bool = False
    metadata: Optional[dict] = None

    @property
    def token_count(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class EffectivePromptBlock:
    """One logical prompt block after PFlash transforms token spans."""

    block_id: str
    index: int
    start: int
    end: int
    raw_start: int
    raw_end: int
    role: str
    kind: str
    message_start: int
    message_end: int
    protected: bool = False
    compressed: bool = False
    metadata: Optional[dict] = None

    @property
    def token_count(self) -> int:
        return max(0, self.end - self.start)


def _default_prompt_blocks(prompt_ids: list[int]) -> list[PromptBlock]:
    if not prompt_ids:
        return []
    return [
        PromptBlock(
            block_id="prompt:0",
            index=0,
            start=0,
            end=len(prompt_ids),
            role="prompt",
            kind="prompt",
            message_start=0,
            message_end=0,
            protected=False,
            metadata={"default": True},
        )
    ]


def materialize_blocks(
    prompt_ids: list[int],
    blocks: Optional[list[PromptBlock]] = None,
) -> list[EffectivePromptBlock]:
    """Project raw prompt blocks into effective-token space unchanged."""
    raw_blocks = blocks or _default_prompt_blocks(prompt_ids)
    effective = []
    cursor = 0
    for block in raw_blocks:
        length = block.token_count
        effective.append(
            EffectivePromptBlock(
                block_id=block.block_id,
                index=block.index,
                start=cursor,
                end=cursor + length,
                raw_start=block.start,
                raw_end=block.end,
                role=block.role,
                kind=block.kind,
                message_start=block.message_start,
                message_end=block.message_end,
                protected=block.protected,
                compressed=False,
                metadata=block.metadata,
            )
        )
        cursor += length
    return effective


def _is_default_prompt_block(blocks: list[PromptBlock]) -> bool:
    return (
        len(blocks) == 1
        and blocks[0].kind == "prompt"
        and blocks[0].message_start == 0
        and blocks[0].message_end == 0
    )


def _protected_block_indexes(blocks: list[PromptBlock], tail_budget: int) -> set[int]:
    if not blocks:
        return set()
    if _is_default_prompt_block(blocks):
        return set()

    protected = {
        index
        for index, block in enumerate(blocks)
        if block.protected or block.kind == "generation_prompt" or block.message_start == block.message_end
    }

    content_indexes = [
        index
        for index, block in enumerate(blocks)
        if block.message_end > block.message_start and block.kind != "generation_prompt"
    ]
    if not content_indexes:
        return protected

    msg_groups: dict[int, list[int]] = {}
    for idx in content_indexes:
        key = blocks[idx].message_start
        msg_groups.setdefault(key, []).append(idx)

    msg_keys = sorted(msg_groups)

    head_count = 2 if len(msg_keys) >= 2 else 1
    for msg_key in msg_keys[:head_count]:
        protected.update(msg_groups[msg_key])

    tail_so_far = 0
    for msg_key in reversed(msg_keys[head_count:]):
        tail_so_far += sum(blocks[idx].token_count for idx in msg_groups[msg_key])
        protected.update(msg_groups[msg_key])
        if tail_so_far > tail_budget:
            break

    return protected


async def maybe_compress(
    prompt_ids: list,
    daemon,
    config: PrefillConfig,
    blocks: Optional[list[PromptBlock]] = None,
) -> tuple:
    """Compress prompt if it exceeds the threshold.

    Protects the head block set, the last blocks that fit within tail_budget,
    and any caller-marked protected blocks (for example obsidian note reads).
    Compression only happens on whole block spans so metadata can be preserved.

    Args:
        prompt_ids: Original prompt token IDs
        daemon: DflashDaemon instance
        config: PrefillConfig
        blocks: Prompt manifest in raw token space.

    Returns:
        (compressed_ids, compression_fired, effective_blocks). If compression
        doesn't fire, returns the original prompt_ids plus identity spans.
    """
    raw_blocks = blocks or _default_prompt_blocks(prompt_ids)

    if not config.enabled or config.drafter_path is None:
        return prompt_ids, False, materialize_blocks(prompt_ids, raw_blocks)

    if len(prompt_ids) < config.threshold:
        return prompt_ids, False, materialize_blocks(prompt_ids, raw_blocks)

    protected_indexes = _protected_block_indexes(raw_blocks, config.tail_budget)
    compressible_indexes = {
        index
        for index, block in enumerate(raw_blocks)
        if index not in protected_indexes and block.token_count >= 256
    }

    total_compressible = sum(raw_blocks[index].token_count for index in compressible_indexes)
    if total_compressible < 1024:
        return prompt_ids, False, materialize_blocks(prompt_ids, raw_blocks)

    logger.info(
        f"pflash compressing {total_compressible} tokens across {len(compressible_indexes)} blocks "
        f"(keep={config.keep_ratio})"
    )

    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    MAX_BLOCK = 14000  # keep under daemon MAX_S=16384

    compressed = []
    effective_blocks = []
    cursor = 0
    for index, block in enumerate(raw_blocks):
        raw_ids = prompt_ids[block.start:block.end]
        compressed_ids = raw_ids
        compressed_block = False
        if index in compressible_indexes:
            # Chunk large blocks so daemon doesn't OOM
            chunks = [raw_ids[i:i+MAX_BLOCK] for i in range(0, len(raw_ids), MAX_BLOCK)]
            chunked = []
            for ch in chunks:
                c = await loop.run_in_executor(
                    None,
                    lambda g=ch: daemon.compress(
                        g,
                        drafter_path=config.drafter_path,
                        keep_ratio=config.keep_ratio,
                    ),
                )
                chunked.extend(c)
            compressed_ids = chunked
            compressed_block = True
        compressed.extend(compressed_ids)
        effective_blocks.append(
            EffectivePromptBlock(
                block_id=block.block_id,
                index=block.index,
                start=cursor,
                end=cursor + len(compressed_ids),
                raw_start=block.start,
                raw_end=block.end,
                role=block.role,
                kind=block.kind,
                message_start=block.message_start,
                message_end=block.message_end,
                protected=index in protected_indexes,
                compressed=compressed_block,
                metadata=block.metadata,
            )
        )
        cursor += len(compressed_ids)

    elapsed = time.monotonic() - t0
    total_compressed = sum(block.token_count for block in effective_blocks if block.compressed)
    logger.info(
        f"pflash blocks {total_compressible} -> {total_compressed} tokens "
        f"({total_compressible / max(total_compressed, 1):.1f}x, "
        f"total {len(prompt_ids)} -> {len(compressed)}) in {elapsed:.1f}s"
    )

    return compressed, True, effective_blocks
