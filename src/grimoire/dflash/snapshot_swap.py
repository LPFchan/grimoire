"""SSD-backed KV snapshot swap for DFlash models.

When VRAM slots are exhausted, the LRU snapshot is serialized to SSD and
its slot is freed. When the evicted slot is needed again, it's loaded back
from disk. This effectively gives unlimited prefix cache capacity at the
cost of SSD I/O latency (~100-300ms per swap on NVMe).

The swap directory holds .dfsn files named by slot (e.g., slot-2.dfsn).
A manifest.json tracks which slots are on disk and their hash keys.
"""

import json
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Swap metadata file in the swap directory.
MANIFEST_NAME = "swap-manifest.json"


class SnapshotSwap:
    """LRU-managed SSD swap for KV snapshot slots.

    Wraps the PrefixCache and daemon to transparently swap cold snapshots
    between VRAM and SSD when the slot count exceeds the VRAM budget.

    Args:
        swap_dir: Directory for .dfsn files and manifest.
        max_vram_slots: Maximum slots to keep in VRAM. When exceeded,
            the LRU snapshot is saved to SSD and freed from VRAM.
    """

    def __init__(self, swap_dir: str, max_vram_slots: int = 4):
        self.swap_dir = Path(swap_dir)
        self.max_vram_slots = max_vram_slots
        # VRAM slots: hash_key -> slot_id (in-use, in VRAM)
        self.vram: OrderedDict[bytes, int] = OrderedDict()
        # SSD slots: hash_key -> path (on disk, not in VRAM)
        self.disk: OrderedDict[bytes, str] = OrderedDict()
        # Manifest: maps slot_id -> {key_hex, path, hash_key}
        self._load_manifest()

        self.swap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"snapshot swap enabled: dir={self.swap_dir} "
            f"max_vram={max_vram_slots}"
        )

    def _manifest_path(self) -> Path:
        return self.swap_dir / MANIFEST_NAME

    def _load_manifest(self) -> None:
        """Restore swap state from disk manifest."""
        mp = self._manifest_path()
        if not mp.exists():
            return
        try:
            with open(mp) as f:
                data = json.load(f)
            for entry in data.get("disk", []):
                key = bytes.fromhex(entry["key_hex"])
                path = entry["path"]
                if Path(path).exists():
                    self.disk[key] = path
            logger.info(f"swap manifest loaded: {len(self.disk)} disk entries")
        except Exception as e:
            logger.error(f"swap manifest load failed: {e}")

    def _save_manifest(self) -> None:
        """Persist swap state to disk."""
        mp = self._manifest_path()
        data = {
            "disk": [
                {"key_hex": key.hex(), "path": path}
                for key, path in self.disk.items()
            ]
        }
        tmp = mp.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(str(tmp), str(mp))
        except Exception as e:
            logger.error(f"swap manifest save failed: {e}")

    def _evict_one(self, daemon) -> int:
        """Evict LRU VRAM slot to SSD, returning the freed slot ID."""
        if not self.vram:
            raise RuntimeError("No VRAM slots to evict")

        key, slot = next(iter(self.vram.items()))
        self.vram.pop(key)

        path = str(self.swap_dir / f"slot-{slot}.dfsn")
        try:
            daemon.save_snapshot(slot, path)
            self.disk[key] = path
            self._save_manifest()
            logger.info(f"swap: evicted slot={slot} to {path}")
            return slot
        except Exception as e:
            # If save fails, keep the slot in VRAM (don't lose the snapshot).
            self.vram[key] = slot
            logger.error(f"swap: eviction failed for slot={slot}: {e}")
            raise

    def _load_to_vram(self, daemon, slot: int, key: bytes) -> None:
        """Load a snapshot from SSD into a VRAM slot."""
        if key not in self.disk:
            raise KeyError(f"Key not in disk cache: {key.hex()}")

        path = self.disk[key]
        if not Path(path).exists():
            logger.warning(f"swap: disk file missing for key={key.hex()}, evicting")
            del self.disk[key]
            self._save_manifest()
            raise KeyError(f"Disk file missing: {path}")

        try:
            daemon.load_snapshot(slot, path)
            self.vram[key] = slot
            del self.disk[key]
            self._save_manifest()
            logger.info(f"swap: loaded slot={slot} from {path}")
        except Exception as e:
            logger.error(f"swap: load failed for slot={slot}: {e}")
            raise

    def reserve_slot(self, daemon, key: bytes) -> Optional[int]:
        """Reserve a VRAM slot, evicting to SSD if necessary.

        Returns slot ID or None if swap is disabled (max_vram_slots == 0).
        """
        if self.max_vram_slots <= 0:
            return None

        if len(self.vram) >= self.max_vram_slots:
            # Need to evict LRU to make room.
            self._evict_one(daemon)

        # Find a free slot (not in vram, not in disk).
        used = set(self.vram.values())
        for slot in range(8):
            if slot not in used:
                self.vram[key] = slot
                return slot

        # All 8 daemon slots are taken (only reachable when max_vram_slots > 8,
        # since otherwise the loop above would have found a free ID). Evict
        # again and reuse the slot _evict_one returns directly — looking it up
        # from self.vram after eviction picks up the MRU survivor instead of
        # the freed slot.
        freed_slot = self._evict_one(daemon)
        self.vram[key] = freed_slot
        return freed_slot

    def get(self, key: bytes) -> Optional[tuple]:
        """Look up a slot by hash key.

        Returns (slot, in_vram) where in_vram is True if the snapshot
        is currently in VRAM, or False if it's on disk.
        """
        if key in self.vram:
            self.vram.move_to_end(key)
            return (self.vram[key], True)
        if key in self.disk:
            self.disk.move_to_end(key)
            # We don't know the slot ID for disk entries. Return None for now.
            # The caller should use reserve_slot to get a slot, then load.
            return None
        return None

    def release(self, daemon, key: bytes) -> None:
        """Release a slot (e.g., on conversation end).

        If the slot is in VRAM, it may be evicted to SSD. If on disk,
        it's deleted.
        """
        if key in self.vram:
            slot = self.vram[key]
            self.vram.pop(key)
            # Save to disk instead of freeing, so it can be restored later.
            path = str(self.swap_dir / f"slot-{slot}.dfsn")
            try:
                daemon.save_snapshot(slot, path)
                self.disk[key] = path
                self._save_manifest()
            except Exception:
                pass
        elif key in self.disk:
            path = self.disk.pop(key)
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
            self._save_manifest()

    def clear(self) -> None:
        """Clear all swap state."""
        self.vram.clear()
        self.disk.clear()
        if self.swap_dir.exists():
            shutil.rmtree(self.swap_dir, ignore_errors=True)
        self.swap_dir.mkdir(parents=True, exist_ok=True)
