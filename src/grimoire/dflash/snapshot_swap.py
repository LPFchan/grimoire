"""SSD-backed KV snapshot swap for DFlash models.

When VRAM slots are exhausted, the LRU snapshot is serialized to SSD and
its slot is freed. When the evicted slot is needed again, it's loaded back
from disk. This effectively gives unlimited prefix cache capacity at the
cost of SSD I/O latency (~100-300ms per swap on NVMe).

Disk files are named by content hash (e.g., swap-a3f2c91b.dfsn) so two
different snapshots evicted from the same slot don't clobber each other.
A swap-manifest.json maps hash keys → disk paths.
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


def _disk_path(swap_dir: Path, key: bytes) -> Path:
    """Return a unique disk file path derived from the hash key."""
    return swap_dir / f"swap-{key.hex()[:16]}.dfsn"


class SnapshotSwap:
    """LRU-managed SSD swap for KV snapshot slots.

    Wraps the PrefixCache and daemon to transparently swap cold snapshots
    between VRAM and SSD when the slot count exceeds the VRAM budget.

    Args:
        swap_dir: Directory for .dfsn files and manifest.
        max_vram_slots: Maximum slots to keep in VRAM. When exceeded,
            the LRU snapshot is saved to SSD and freed from VRAM.
    """

    def __init__(
        self,
        swap_dir: str,
        max_vram_slots: int = 4,
        slot_offset: int = 0,
        slot_count: int = 8,
    ):
        self.swap_dir = Path(swap_dir)
        slot_count = max(0, min(8 - slot_offset, slot_count))
        self.allowed_slots = list(range(slot_offset, slot_offset + slot_count))
        if max_vram_slots > len(self.allowed_slots):
            logger.warning(
                f"snapshot swap max_vram_slots={max_vram_slots} exceeds "
                f"allowed slots ({len(self.allowed_slots)}); clamping"
            )
            max_vram_slots = len(self.allowed_slots)
        self.max_vram_slots = max_vram_slots
        # VRAM slots: hash_key -> slot_id (in-use, in VRAM)
        self.vram: OrderedDict[bytes, int] = OrderedDict()
        # SSD slots: hash_key -> path (on disk, not in VRAM)
        self.disk: OrderedDict[bytes, str] = OrderedDict()
        self._load_manifest()

        self.swap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"snapshot swap enabled: dir={self.swap_dir} "
            f"max_vram={self.max_vram_slots} slots={self.allowed_slots}"
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

    def _evict_to_disk(self, daemon, key: bytes, slot: int) -> str:
        """Save a VRAM snapshot to disk, returning the disk path."""
        path = str(_disk_path(self.swap_dir, key))
        try:
            daemon.save_snapshot(slot, path)
            logger.info(f"swap: evicted slot={slot} to {path}")
            return path
        except Exception as e:
            logger.error(f"swap: eviction failed for slot={slot}: {e}")
            raise

    def _load_from_disk(self, daemon, key: bytes, target_slot: int) -> None:
        """Load a disk snapshot into a VRAM slot."""
        path = self.disk[key]
        if not Path(path).exists():
            logger.warning(f"swap: disk file missing for key={key.hex()[:8]}, evicting")
            del self.disk[key]
            self._save_manifest()
            raise KeyError(f"Disk file missing: {path}")

        try:
            daemon.load_snapshot(target_slot, path)
            logger.info(f"swap: loaded slot={target_slot} from {path}")
        except Exception as e:
            logger.error(f"swap: load failed for key={key.hex()[:8]}: {e}")
            raise

    def reserve_slot(self, daemon, key: bytes) -> Optional[int]:
        """Reserve a VRAM slot, evicting to SSD if necessary.

        Returns slot ID or None if swap is disabled (max_vram_slots == 0).
        """
        if self.max_vram_slots <= 0:
            return None

        if key in self.vram:
            self.vram.move_to_end(key)
            return self.vram[key]

        # Make room if VRAM is full — applies whether the key is on disk
        # (and we're about to load it back) or new.
        if len(self.vram) >= self.max_vram_slots:
            lru_key, lru_slot = next(iter(self.vram.items()))
            disk_path = self._evict_to_disk(daemon, lru_key, lru_slot)
            self.vram.pop(lru_key)
            self.disk[lru_key] = disk_path
            self._save_manifest()

        # Find a free daemon slot inside the reserved slot range.
        used = set(self.vram.values())
        free_slot = next((s for s in self.allowed_slots if s not in used), None)
        if free_slot is None:
            raise RuntimeError("All allowed daemon snapshot slots are in use")

        # If the key is on disk, restore its contents into the slot.
        if key in self.disk:
            self._load_from_disk(daemon, key, free_slot)
            self.disk.pop(key)

        self.vram[key] = free_slot
        self._save_manifest()
        return free_slot

    def get(self, key: bytes) -> Optional[tuple]:
        """Look up a snapshot by hash key.

        Returns:
            (slot, True)  if in VRAM (ready for RESTORE)
            (None, False) if on disk (needs LOAD_SNAPSHOT before RESTORE)
            None           if not found at all
        """
        if key in self.vram:
            self.vram.move_to_end(key)
            return (self.vram[key], True)
        if key in self.disk:
            self.disk.move_to_end(key)
            return (None, False)
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
            path = self._evict_to_disk(daemon, key, slot)
            self.disk[key] = path
            self._save_manifest()
        elif key in self.disk:
            path = self.disk.pop(key)
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
            self._save_manifest()

    def discard(self, daemon, key: bytes) -> None:
        """Drop a snapshot from VRAM/disk without saving it first."""
        if key in self.vram:
            slot = self.vram.pop(key)
            if daemon is not None:
                try:
                    daemon.free_snapshot(slot)
                except Exception:
                    pass
            self._save_manifest()
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
