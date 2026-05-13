"""Prefix cache for DFlash models.

LRU-managed KV cache with disk persistence. Caches prompt prefixes so repeated
system prompts / conversation starts skip the expensive prefill phase.

On model load: snapshots are restored from disk.
On model unload: active snapshots are saved to disk.
During requests: inline snapshots are taken during prefill for future hits.
"""

import asyncio
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
    """LRU prefix cache with disk persistence.

    The daemon owns the actual KV snapshots (in VRAM). This class maintains
    the Python-side index: hash(prefix_ids) -> slot_id.

    On save/restore, snapshot metadata is persisted to disk so cache state
    survives daemon reloads. The actual KV data lives in the daemon's VRAM
    and is restored via the daemon's SNAPSHOT/RESTORE protocol.

    Args:
        cap: Maximum number of snapshot slots (0-8). 0 disables caching.
        cache_dir: Directory for persistent metadata. Snapshots save/restore
                   via the daemon's SNAPSHOT/RESTORE commands.
        kv_k_type: KV cache type (part of cache key hash).
        fa_window: Flash attention window (part of cache key hash).
    """

    def __init__(
        self,
        cap: int = 4,
        cache_dir: str = "/var/lib/grimoire/prefix_cache",
        kv_k_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window
        self.cache_dir = Path(cache_dir)

        if cap > MAX_DAEMON_SLOTS:
            logger.warning(
                f"prefix cache cap={cap} exceeds daemon limit ({MAX_DAEMON_SLOTS}); clamping"
            )
            cap = MAX_DAEMON_SLOTS
        self.cap = cap
        self.disabled = cap <= 0

        if self.disabled:
            return

        # LRU index: hash -> slot_id
        self.entries: OrderedDict[bytes, int] = OrderedDict()
        self.next_slot = 0
        self._pending_evict_key: Optional[bytes] = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"prefix cache enabled: cap={cap} dir={self.cache_dir}"
        )

    def hash_prefix(self, prefix_ids: list) -> bytes:
        """Compute stable SHA-1 (16B) hash of a token prefix."""
        h = hashlib.sha1()
        h.update(struct.pack("<I", len(prefix_ids)))
        h.update(struct.pack(f"<{len(prefix_ids)}i", *prefix_ids))
        h.update(str(self.kv_k_type).encode())
        h.update(b"\x00")
        h.update(struct.pack("<I", self.fa_window or 0))
        return h.digest()[:16]

    def lookup(self, prompt_ids: list, boundaries: Optional[list] = None) -> Optional[tuple]:
        """Find the longest cached prefix of prompt_ids at a known boundary.

        Args:
            prompt_ids: full prompt token sequence
            boundaries: candidate token positions to probe (e.g., end of system
                message, end of each conversation turn). Each must be an integer
                in (0, len(prompt_ids)]. If None, only the full prompt is probed.

        Returns:
            (slot_id, prefix_len) for the deepest cached match, or None.
        """
        if self.disabled:
            return None

        candidates = list(boundaries) if boundaries else []
        candidates.append(len(prompt_ids))
        # Probe deepest first so we win on the longest cached prefix.
        seen = set()
        for boundary in sorted({b for b in candidates if 0 < b <= len(prompt_ids)}, reverse=True):
            if boundary in seen:
                continue
            seen.add(boundary)
            key = self.hash_prefix(prompt_ids[:boundary])
            if key in self.entries:
                slot = self.entries[key]
                self.entries.move_to_end(key)
                logger.debug(f"prefix cache hit slot={slot} len={boundary}")
                return (slot, boundary)
        return None

    def prepare_inline_snap(
        self, prompt_ids: list, boundary: int
    ) -> Optional[tuple]:
        """Prepare an inline snapshot at the given boundary position.

        Picks a slot and defers eviction until confirm. Returns
        (slot_id, boundary) or None if the boundary is unusable or already cached.

        Args:
            prompt_ids: Full prompt token IDs
            boundary: Token position to snapshot at (e.g., end of system prompt).
                Must be > 0 and <= len(prompt_ids); otherwise no snapshot is taken.
        """
        if self.disabled:
            return None
        if boundary <= 0 or boundary > len(prompt_ids):
            return None

        prefix = prompt_ids[:boundary]
        key = self.hash_prefix(prefix)

        if key in self.entries:
            self.entries.move_to_end(key)
            return None  # Already cached

        # Pick slot — defer eviction until confirm
        if len(self.entries) >= self.cap:
            old_key = next(iter(self.entries))
            slot = self.entries[old_key]
            self._pending_evict_key = old_key
        else:
            slot = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.cap
            self._pending_evict_key = None

        return (slot, boundary)

    def confirm_inline_snap(
        self, slot: int, boundary: int, prompt_ids: list
    ) -> None:
        """Confirm an inline snapshot after successful daemon execution.

        Atomically evicts the pending old entry and registers the new one.
        """
        if self.disabled:
            return

        if self._pending_evict_key is not None:
            self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None

        prefix = prompt_ids[:boundary]
        key = self.hash_prefix(prefix)
        self.entries[key] = slot
        logger.debug(f"prefix cache committed slot={slot} len={boundary}")

    def abort_inline_snap(self, slot: int) -> None:
        """Cancel a pending snapshot reservation.

        If we were at-cap and reserved a slot, we conservatively drop
        the pending eviction key since we can't know if the daemon
        already wrote to that slot.
        """
        if self.disabled:
            return
        if self._pending_evict_key is not None:
            self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None

    def _meta_path(self) -> Path:
        """Path to the cache metadata file."""
        return self.cache_dir / "index.json"

    def save(self) -> None:
        """Save cache index to disk.

        Persists the hash->slot mapping so it can be restored on daemon reload.
        Note: the actual KV data in VRAM must be saved via daemon SNAPSHOT
        commands before daemon shutdown.
        """
        if self.disabled:
            return

        meta = {
            "entries": [
                {"key_hex": key.hex(), "slot": slot}
                for key, slot in self.entries.items()
            ],
            "next_slot": self.next_slot,
        }

        meta_path = self._meta_path()
        try:
            tmp = meta_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(str(tmp), str(meta_path))
            logger.info(
                f"prefix cache saved: {len(self.entries)} entries -> {meta_path}"
            )
        except Exception as e:
            logger.error(f"prefix cache save failed: {e}")

    def load(self) -> None:
        """Restore cache index from disk.

        Called when the model is loaded. Rebuilds the hash->slot index
        from the persisted metadata. The daemon will have restored the
        actual KV data via its SNAPSHOT/RESTORE protocol.
        """
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
                slot = entry["slot"]
                if 0 <= slot < self.cap:
                    self.entries[key] = slot
            self.next_slot = meta.get("next_slot", 0) % max(self.cap, 1)
            logger.info(
                f"prefix cache loaded: {len(self.entries)} entries from {meta_path}"
            )
        except Exception as e:
            logger.error(f"prefix cache load failed: {e}")
            self.entries.clear()

    def clear(self) -> None:
        """Clear all cache entries and disk state."""
        self.entries.clear()
        self.next_slot = 0
        self._pending_evict_key = None
        meta_path = self._meta_path()
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)
        logger.info("prefix cache cleared")

    def cleanup(self, daemon=None) -> None:
        """Clean up on model unload.

        Tells the daemon to free all snapshot slots, then clears the index.
        """
        if self.disabled:
            return

        # Free daemon slots
        if daemon and daemon.is_running():
            for _, slot in list(self.entries.items()):
                try:
                    daemon.free_snapshot(slot)
                except Exception:
                    pass

        self.clear()
