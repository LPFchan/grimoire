"""Prompt blocks in raw token space.

A prompt is rendered as an ordered set of blocks with stable ids and token
spans. Blocks are what make prefix caching addressable: the boundary between
one turn's tokens and the next is a block edge, so a cached prefix can be
matched and reused without re-tokenizing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
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
