"""Session-aware compact full snapshot metadata for DFlash models."""

import hashlib
import json
import logging
import os
import struct
from collections import OrderedDict
from pathlib import Path
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

    def __init__(self, cap: int = DEFAULT_SESSION_CAP, path: Optional[str] = None):
        self.cap = max(0, int(cap))
        self.path = Path(path) if path else None
        self.sessions: OrderedDict[str, tuple[bytes, int, bytes]] = OrderedDict()
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        if self.cap <= 0:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            entries = data.get("sessions", [])
            if not isinstance(entries, list):
                raise ValueError("sessions must be a list")
            for entry in entries:
                conversation_id = entry.get("conversation_id")
                snapshot_key_hex = entry.get("snapshot_key")
                prefix_len = entry.get("prefix_len")
                prefix_hash_hex = entry.get("prefix_hash")
                if not isinstance(conversation_id, str):
                    continue
                if not isinstance(snapshot_key_hex, str) or not isinstance(prefix_hash_hex, str):
                    continue
                if prefix_len is None:
                    continue
                self.sessions[conversation_id] = (
                    bytes.fromhex(snapshot_key_hex),
                    int(prefix_len),
                    bytes.fromhex(prefix_hash_hex),
                )
            while self.cap > 0 and len(self.sessions) > self.cap:
                self.sessions.popitem(last=False)
        except Exception as e:
            logger.warning("session kv: failed to load persisted state from %s: %s", self.path, e)
            self.sessions.clear()

    def _save(self) -> None:
        if self.path is None:
            return
        if self.cap <= 0:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        payload = {
            "sessions": [
                {
                    "conversation_id": conversation_id,
                    "snapshot_key": snapshot_key.hex(),
                    "prefix_len": prefix_len,
                    "prefix_hash": prefix_hash.hex(),
                }
                for conversation_id, (snapshot_key, prefix_len, prefix_hash) in self.sessions.items()
            ]
        }
        os.makedirs(self.path.parent, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
            f.write("\n")
        os.replace(tmp_path, self.path)

    def swap_key(self, conversation_id: str) -> bytes:
        """Return the stable persisted snapshot key for a conversation."""
        return _session_key(conversation_id)

    def has_session(self, conversation_id: str) -> bool:
        """Return True if a conversation currently has session metadata."""
        if self.cap <= 0:
            return False
        return conversation_id in self.sessions

    def get_session(self, conversation_id: str, prompt_ids: list) -> Optional[tuple[bytes, int]]:
        """Return (snapshot_key, prefix_len) if the prompt prefix matches."""
        if self.cap <= 0:
            return None
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
            self._save()
            return None
        current_hash = _prefix_hash(prompt_ids[:prefix_len])
        if current_hash != prefix_hash:
            logger.debug("session kv: hash mismatch for %s, evicting stale snapshot", conversation_id[:8])
            self.sessions.pop(conversation_id, None)
            self._save()
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
        self._save()
        return old_id

    def update(self, conversation_id: str, prefix_len: int, prompt_ids: list) -> Optional[bytes]:
        """Record or update the compact full snapshot for a conversation."""
        if self.cap <= 0:
            self.sessions.pop(conversation_id, None)
            self._save()
            return None
        snapshot_key = self.swap_key(conversation_id)
        prefix_hash = _prefix_hash(prompt_ids[:prefix_len])
        self.sessions[conversation_id] = (snapshot_key, int(prefix_len), prefix_hash)
        self.sessions.move_to_end(conversation_id)
        while self.cap > 0 and len(self.sessions) > self.cap:
            self.sessions.popitem(last=False)
        self._save()
        return snapshot_key

    def all_keys(self, conversation_id: str) -> list[bytes]:
        """Return persisted snapshot keys for a conversation."""
        if self.cap <= 0 or not conversation_id:
            return []
        entry = self.sessions.get(conversation_id)
        if entry is not None:
            return [entry[0]]
        return [self.swap_key(conversation_id)]

    def evict(self, conversation_id: str) -> None:
        """Remove a session (e.g., on new conversation or error)."""
        removed = self.sessions.pop(conversation_id, None)
        if removed is not None:
            self._save()

    def clear(self) -> None:
        """Clear all session state."""
        self.sessions.clear()
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
