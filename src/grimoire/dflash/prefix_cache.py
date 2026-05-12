"""Prefix cache metadata for DFlash compact snapshots.

Caches reusable prompt prefixes by hash and maps them to persisted compact full
snapshot keys in the snapshot store.
"""

import hashlib
import json
import logging
import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_DAEMON_SLOTS = 8


class PrefixCache:
    """LRU prefix cache backed by persisted compact snapshot keys."""

    def __init__(
        self,
        cap: int = 4,
        cache_dir: str = "/var/lib/grimoire/prefix_cache",
        kv_k_type: str = "q8_0",
        kv_v_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.kv_k_type = kv_k_type
        self.kv_v_type = kv_v_type
        self.fa_window = fa_window
        self.cache_dir = Path(cache_dir)

        if cap > MAX_DAEMON_SLOTS:
            logger.warning(
                "prefix cache cap=%s exceeds daemon limit (%s); clamping",
                cap,
                MAX_DAEMON_SLOTS,
            )
            cap = MAX_DAEMON_SLOTS
        self.cap = max(0, int(cap))
        self.disabled = self.cap <= 0

        if self.disabled:
            return

        self.entries: OrderedDict[bytes, bytes] = OrderedDict()
        self._pending_evict_key: Optional[bytes] = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("prefix cache enabled: cap=%s dir=%s", self.cap, self.cache_dir)

    def hash_prefix(self, prefix_ids: list) -> bytes:
        """Compute stable SHA-1 (16B) hash of a token prefix."""
        h = hashlib.sha1()
        h.update(struct.pack("<I", len(prefix_ids)))
        h.update(struct.pack(f"<{len(prefix_ids)}i", *prefix_ids))
        h.update(str(self.kv_k_type).encode())
        h.update(b"\x00")
        h.update(str(self.kv_v_type).encode())
        h.update(b"\x00")
        h.update(struct.pack("<I", self.fa_window or 0))
        return h.digest()[:16]

    def snapshot_key(self, prefix_hash: bytes) -> bytes:
        """Return the persisted snapshot key for a prefix hash."""
        h = hashlib.sha1()
        h.update(b"prefix-cache\x00")
        h.update(prefix_hash)
        return h.digest()[:16]

    def lookup(self, prompt_ids: list, boundaries: Optional[list] = None) -> Optional[tuple[bytes, int]]:
        """Find the longest cached prefix of prompt_ids at a known boundary."""
        if self.disabled:
            return None

        candidates = list(boundaries) if boundaries else []
        candidates.append(len(prompt_ids))
        seen = set()
        for boundary in sorted({b for b in candidates if 0 < b <= len(prompt_ids)}, reverse=True):
            if boundary in seen:
                continue
            seen.add(boundary)
            prefix_hash = self.hash_prefix(prompt_ids[:boundary])
            snapshot_key = self.entries.get(prefix_hash)
            if snapshot_key is None:
                continue
            self.entries.move_to_end(prefix_hash)
            logger.debug("prefix cache hit len=%s", boundary)
            return (snapshot_key, boundary)
        return None

    def prepare_inline_snap(self, prompt_ids: list, boundary: int) -> Optional[tuple[bytes, int]]:
        """Prepare a new persisted snapshot entry for a prompt boundary."""
        if self.disabled or boundary <= 0 or boundary > len(prompt_ids):
            return None

        prefix_hash = self.hash_prefix(prompt_ids[:boundary])
        if prefix_hash in self.entries:
            self.entries.move_to_end(prefix_hash)
            return None

        if len(self.entries) >= self.cap:
            self._pending_evict_key = next(iter(self.entries))
        else:
            self._pending_evict_key = None
        return (self.snapshot_key(prefix_hash), boundary)

    def confirm_inline_snap(self, snapshot_key: bytes, boundary: int, prompt_ids: list) -> Optional[bytes]:
        """Commit a prepared prefix snapshot and return any evicted snapshot key."""
        if self.disabled:
            return None

        evicted_snapshot_key = None
        if self._pending_evict_key is not None:
            evicted_snapshot_key = self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None

        prefix_hash = self.hash_prefix(prompt_ids[:boundary])
        self.entries[prefix_hash] = snapshot_key
        self.entries.move_to_end(prefix_hash)
        return evicted_snapshot_key

    def abort_inline_snap(self, snapshot_key: bytes) -> None:
        """Cancel a pending inline snapshot reservation."""
        if self.disabled:
            return
        self._pending_evict_key = None

    def discard(self, snapshot_key: bytes) -> None:
        """Drop any cache entry pointing at snapshot_key."""
        if self.disabled:
            return
        doomed = [prefix_hash for prefix_hash, key in self.entries.items() if key == snapshot_key]
        for prefix_hash in doomed:
            self.entries.pop(prefix_hash, None)

    def _meta_path(self) -> Path:
        return self.cache_dir / "index.json"

    def save(self) -> None:
        """Persist the prefix-hash -> snapshot-key index."""
        if self.disabled:
            return

        meta = {
            "entries": [
                {
                    "key_hex": key.hex(),
                    "snapshot_key_hex": snapshot_key.hex(),
                }
                for key, snapshot_key in self.entries.items()
            ]
        }

        meta_path = self._meta_path()
        try:
            tmp = meta_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(str(tmp), str(meta_path))
            logger.info("prefix cache saved: %s entries -> %s", len(self.entries), meta_path)
        except Exception as e:
            logger.error("prefix cache save failed: %s", e)

    def load(self) -> None:
        """Restore the persisted prefix cache index."""
        if self.disabled:
            return

        meta_path = self._meta_path()
        if not meta_path.exists():
            return

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self.entries.clear()
            for entry in meta.get("entries", []):
                key = bytes.fromhex(entry["key_hex"])
                snapshot_key = bytes.fromhex(entry["snapshot_key_hex"])
                self.entries[key] = snapshot_key
            logger.info("prefix cache loaded: %s entries from %s", len(self.entries), meta_path)
        except Exception as e:
            logger.error("prefix cache load failed: %s", e)
            self.entries.clear()

    def clear(self) -> None:
        """Clear all cache metadata."""
        self.entries.clear()
        self._pending_evict_key = None
        meta_path = self._meta_path()
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)
        logger.info("prefix cache cleared")

    def cleanup(self, daemon=None) -> None:
        """Clean up on model unload.

        Prefix snapshots are persisted out-of-band in the snapshot store, so
        there is nothing to free here.
        """
        return
