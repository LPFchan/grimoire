"""Session-aware KV management for DFlash models.

Maps conversation_id -> (slot, prefix_len) so continuation turns only prefill
the delta (new messages since last turn) instead of the full prompt.

After each generation, a snapshot is taken at len(effective_ids) so the next
turn can RESTORE from it.
"""

import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# Reserve the last N daemon slots for session KV (not prefix cache).
# With cap=2 prefix slots and 2 session slots, slots 0-1 are prefix, 2-3 are
# session. This avoids the prefix cache evicting an active session snapshot.
SESSION_SLOT_OFFSET = 2
DEFAULT_SESSION_CAP = 2


class SessionKV:
    """Manages per-conversation KV snapshot slots.

    Tracks which slot each conversation owns and the prefix length
    (position of the last snapshot). On a returning conversation, the caller
    can RESTORE that slot and only prefill the new delta tokens.

    Args:
        cap: Maximum concurrent session slots.
        prefix_cap: Number of slots reserved for the prefix cache before
            session slots start. Default SESSION_SLOT_OFFSET.
    """

    def __init__(self, cap: int = DEFAULT_SESSION_CAP, prefix_cap: int = SESSION_SLOT_OFFSET):
        self.cap = cap
        self.slot_offset = prefix_cap
        # LRU: conversation_id -> (slot, prefix_len)
        self.sessions: OrderedDict[str, tuple] = OrderedDict()

    def _slot_for_index(self, index: int) -> int:
        """Return daemon slot ID for session index 0..cap-1."""
        return (self.slot_offset + index) % 8

    def get_session(self, conversation_id: str) -> Optional[tuple]:
        """Get cached slot for a conversation, or None.

        Returns (slot, prefix_len) if the conversation has an active
        snapshot slot. Moves the entry to the end (LRU).
        """
        if conversation_id not in self.sessions:
            return None
        self.sessions.move_to_end(conversation_id)
        return self.sessions[conversation_id]

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

    def update(self, conversation_id: str, slot: int, prefix_len: int) -> None:
        """Record or update the snapshot position for a conversation."""
        self.sessions[conversation_id] = (slot, prefix_len)
        self.sessions.move_to_end(conversation_id)

    def evict(self, conversation_id: str) -> None:
        """Remove a session (e.g., on new conversation or error)."""
        self.sessions.pop(conversation_id, None)

    def clear(self) -> None:
        """Clear all session state."""
        self.sessions.clear()
