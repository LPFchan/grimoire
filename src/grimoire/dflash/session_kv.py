"""Session-aware compact full snapshot metadata for DFlash models."""

import hashlib
import logging
import struct
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SESSION_CAP = 4


def _prefix_hash(prompt_ids: list) -> bytes:
    """Compute a 12B truncated hash of a token prefix."""
    h = hashlib.sha1()
    h.update(struct.pack("<I", len(prompt_ids)))
    h.update(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
    return h.digest()[:12]


def _session_key(conversation_id: str) -> bytes:
    """Compute a stable 16B key for persisted session snapshots."""
    h = hashlib.sha1()
    h.update(b"session-kv\x00")
    h.update(str(conversation_id).encode("utf-8", errors="surrogatepass"))
    return h.digest()[:16]


class SessionKV:
    """Manage per-conversation compact full snapshot metadata."""

    def __init__(self, cap: int = DEFAULT_SESSION_CAP):
        self.cap = max(0, int(cap))
        self.sessions: OrderedDict[str, tuple[bytes, int, bytes]] = OrderedDict()

    def swap_key(self, conversation_id: str) -> bytes:
        """Return the stable persisted snapshot key for a conversation."""
        return _session_key(conversation_id)

    def has_session(self, conversation_id: str) -> bool:
        """Return True if a conversation currently has session metadata."""
        return conversation_id in self.sessions

    def get_session(self, conversation_id: str, prompt_ids: list) -> Optional[tuple[bytes, int]]:
        """Return (snapshot_key, prefix_len) if the prompt prefix matches."""
        if conversation_id not in self.sessions:
            return None
        self.sessions.move_to_end(conversation_id)
        snapshot_key, prefix_len, prefix_hash = self.sessions[conversation_id]
        if len(prompt_ids) < prefix_len:
            logger.debug(
                "session kv: prompt shorter than cached prefix (%s < %s), evicting",
                len(prompt_ids),
                prefix_len,
            )
            self.sessions.pop(conversation_id, None)
            return None
        current_hash = _prefix_hash(prompt_ids[:prefix_len])
        if current_hash != prefix_hash:
            logger.debug("session kv: hash mismatch for %s, evicting stale snapshot", conversation_id[:8])
            self.sessions.pop(conversation_id, None)
            return None
        return (snapshot_key, prefix_len)

    def evict_lru_if_full(self, conversation_id: str) -> Optional[str]:
        """Evict the LRU session if adding conversation_id would exceed cap."""
        if self.cap <= 0 or conversation_id in self.sessions:
            return None
        if len(self.sessions) < self.cap:
            return None
        old_id, _ = next(iter(self.sessions.items()))
        self.sessions.pop(old_id)
        return old_id

    def update(self, conversation_id: str, prefix_len: int, prompt_ids: list) -> bytes:
        """Record or update the compact full snapshot for a conversation."""
        snapshot_key = self.swap_key(conversation_id)
        prefix_hash = _prefix_hash(prompt_ids[:prefix_len])
        self.sessions[conversation_id] = (snapshot_key, int(prefix_len), prefix_hash)
        self.sessions.move_to_end(conversation_id)
        while self.cap > 0 and len(self.sessions) > self.cap:
            self.sessions.popitem(last=False)
        return snapshot_key

    def all_keys(self, conversation_id: str) -> list[bytes]:
        """Return persisted snapshot keys for a conversation."""
        entry = self.sessions.get(conversation_id)
        if entry is None:
            return []
        return [entry[0]]

    def evict(self, conversation_id: str) -> None:
        """Remove a session (e.g., on new conversation or error)."""
        self.sessions.pop(conversation_id, None)

    def clear(self) -> None:
        """Clear all session state."""
        self.sessions.clear()
