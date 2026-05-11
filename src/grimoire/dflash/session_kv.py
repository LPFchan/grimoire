"""Session-aware KV management for DFlash models.

Maps conversation_id -> (slot, prefix_len, prefix_hash) so continuation turns
only prefill the delta (new messages since last turn) instead of the full
prompt. The hash is a truncated SHA-1 of the cached token prefix; on a
returning conversation the caller validates that the current prompt still
starts with that prefix before restoring. If content was edited/regenerated
in place, the hash mismatch causes eviction and a full prefill fallback.

After each generation, a snapshot is taken at len(effective_ids) so the next
turn can RESTORE from it.
"""

import hashlib
import logging
import struct
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# Reserve the last N daemon slots for session KV (not prefix cache).
# With cap=2 prefix slots and 2 session slots, slots 0-1 are prefix, 2-3 are
# session. This avoids the prefix cache evicting an active session snapshot.
SESSION_SLOT_OFFSET = 2
DEFAULT_SESSION_CAP = 2


def _prefix_hash(prompt_ids: list) -> bytes:
    """Compute a 12B truncated hash of a token prefix."""
    h = hashlib.sha1()
    h.update(struct.pack("<I", len(prompt_ids)))
    h.update(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
    return h.digest()[:12]


class SessionKV:
    """Manages per-conversation KV snapshot slots.

    Tracks which slot each conversation owns, the prefix length
    (position of the last snapshot), and a content hash of the cached token
    prefix. On a returning conversation, the caller validates the hash
    against the current prompt before restoring. Content edits, truncations,
    or regenerations with the same conversation_id will cause a hash mismatch
    and trigger a full prefill fallback instead of corrupting generation.

    Args:
        cap: Maximum concurrent session slots.
        prefix_cap: Number of slots reserved for the prefix cache before
            session slots start. Default SESSION_SLOT_OFFSET.
    """

    def __init__(self, cap: int = DEFAULT_SESSION_CAP, prefix_cap: int = SESSION_SLOT_OFFSET):
        self.cap = cap
        self.slot_offset = prefix_cap
        # LRU: conversation_id -> (slot, prefix_len, prefix_hash)
        self.sessions: OrderedDict[str, tuple] = OrderedDict()

    def _slot_for_index(self, index: int) -> int:
        """Return daemon slot ID for session index 0..cap-1."""
        return (self.slot_offset + index) % 8

    def get_session(self, conversation_id: str, prompt_ids: list) -> Optional[tuple]:
        """Get cached slot for a conversation if the prompt prefix matches.

        Returns (slot, prefix_len) if the conversation has an active snapshot
        slot AND the current prompt starts with the same prefix that was
        cached. Returns None on miss or hash mismatch (stale snapshot).

        Args:
            conversation_id: The conversation identifier.
            prompt_ids: Current full prompt token IDs (used for hash validation).
        """
        if conversation_id not in self.sessions:
            return None
        self.sessions.move_to_end(conversation_id)
        slot, prefix_len, prefix_hash = self.sessions[conversation_id]
        # Validate: current prompt must start with the cached prefix.
        if len(prompt_ids) < prefix_len:
            logger.debug(f"session kv: prompt shorter than cached prefix ({len(prompt_ids)} < {prefix_len}), evicting")
            self.sessions.pop(conversation_id)
            return None
        current_hash = _prefix_hash(prompt_ids[:prefix_len])
        if current_hash != prefix_hash:
            logger.debug(f"session kv: hash mismatch for {conversation_id[:8]}, evicting stale snapshot")
            self.sessions.pop(conversation_id)
            return None
        return (slot, prefix_len)

    def reserve_slot(self) -> Optional[int]:
        """Reserve a slot for a new session, evicting LRU if full.

        Returns the slot ID, or None if cap is 0.
        """
        if self.cap <= 0:
            return None

        if len(self.sessions) >= self.cap:
            old_id, _ = next(iter(self.sessions.items()))
            self.sessions.pop(old_id)

        # Pick the first slot in [offset, offset+cap) that isn't held by a
        # surviving session. After an LRU eviction this returns the evicted
        # slot; on a cold cache it returns the offset slot. Falling back to
        # len(sessions) is unsafe: after eviction it can collide with the
        # MRU survivor's slot.
        used = {entry[0] for entry in self.sessions.values()}
        for idx in range(self.cap):
            slot = self._slot_for_index(idx)
            if slot not in used:
                return slot
        return None

    def update(self, conversation_id: str, slot: int, prefix_len: int, prompt_ids: list) -> None:
        """Record or update the snapshot position and content hash.

        Args:
            conversation_id: The conversation identifier.
            slot: Daemon slot ID that holds the snapshot.
            prefix_len: Number of tokens captured in the snapshot.
            prompt_ids: Full prompt token IDs (used to compute content hash).
        """
        prefix_hash = _prefix_hash(prompt_ids[:prefix_len])
        self.sessions[conversation_id] = (slot, prefix_len, prefix_hash)
        self.sessions.move_to_end(conversation_id)

    def evict(self, conversation_id: str) -> None:
        """Remove a session (e.g., on new conversation or error)."""
        self.sessions.pop(conversation_id, None)

    def clear(self) -> None:
        """Clear all session state."""
        self.sessions.clear()
