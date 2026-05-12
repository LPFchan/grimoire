"""Dual-persistence snapshot store for DFlash models.

Snapshots are saved to tmpfs synchronously for the hot path, then mirrored to
disk asynchronously for restart resilience. No snapshot remains resident in a
daemon slot beyond the load/restore/save operation itself.
"""

import asyncio
import json
import logging
import os
import shutil
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "swap-manifest.json"


class SnapshotStore:
    """RAM-backed snapshot store with asynchronous disk mirroring."""

    def __init__(
        self,
        ram_dir: str,
        disk_dir: str,
        ram_budget_gb: float = 20.0,
        disk_budget_gb: float = 100.0,
        disk_ttl_hours: float = 24.0,
    ):
        self.ram_dir = Path(ram_dir)
        self.disk_dir = Path(disk_dir)
        self.ram_budget = max(0, int(float(ram_budget_gb) * 1024**3))
        self.disk_budget = max(0, int(float(disk_budget_gb) * 1024**3))
        self.disk_ttl = None
        try:
            ttl_hours = float(disk_ttl_hours)
        except (TypeError, ValueError):
            ttl_hours = 24.0
        if ttl_hours > 0:
            self.disk_ttl = timedelta(hours=ttl_hours)
        self.ram: OrderedDict[bytes, str] = OrderedDict()
        self.disk: OrderedDict[bytes, str] = OrderedDict()
        self._pending_mirrors: set[asyncio.Task] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread_id: Optional[int] = None

        self.ram_dir.mkdir(parents=True, exist_ok=True)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifest()
        self._cleanup_disk()
        logger.info(
            "snapshot store enabled: ram_dir=%s disk_dir=%s ram_budget=%s",
            self.ram_dir,
            self.disk_dir,
            self.ram_budget,
        )

    def _manifest_path(self) -> Path:
        return self.disk_dir / MANIFEST_NAME

    def _path_for(self, root: Path, key: bytes) -> Path:
        return root / f"swap-{key.hex()[:16]}.dfsn"

    def ram_path(self, key: bytes) -> Path:
        return self._path_for(self.ram_dir, key)

    def disk_path(self, key: bytes) -> Path:
        return self._path_for(self.disk_dir, key)

    def _load_manifest(self) -> None:
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
            logger.info("snapshot manifest loaded: %s disk entries", len(self.disk))
        except Exception as e:
            logger.error("snapshot manifest load failed: %s", e)

    def _save_manifest(self) -> None:
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
            logger.error("snapshot manifest save failed: %s", e)

    def bind_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Remember the main asyncio loop for thread-safe mirror scheduling."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        self._loop = loop
        self._loop_thread_id = threading.get_ident()

    def _schedule_mirror(self, loop: asyncio.AbstractEventLoop, src: Path, dst: Path, key: bytes, target: str) -> None:
        def _spawn() -> None:
            task = loop.create_task(self._mirror_async(src, dst, key, target))
            self._pending_mirrors.add(task)
            task.add_done_callback(self._pending_mirrors.discard)

        loop.call_soon_threadsafe(_spawn)

    async def _mirror_async(self, src: Path, dst: Path, key: bytes, target: str) -> None:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, src, dst)
            if target == "disk":
                self.disk[key] = str(dst)
                self.disk.move_to_end(key)
                self._cleanup_disk()
            else:
                self.put_ram(key, dst, mirror_to_disk=False)
        except Exception as e:
            logger.warning("snapshot mirror failed for %s: %s", key.hex()[:8], e)

    def _queue_mirror(self, src: Path, dst: Path, key: bytes, target: str = "disk") -> None:
        try:
            loop = asyncio.get_running_loop()
            self.bind_loop(loop)
        except RuntimeError:
            loop = self._loop
            if loop is None or not loop.is_running():
                logger.warning(
                    "snapshot mirror skipped: no running event loop (key=%s target=%s)",
                    key.hex()[:8],
                    target,
                )
                return

        if loop is self._loop and self._loop_thread_id != threading.get_ident():
            self._schedule_mirror(loop, src, dst, key, target)
            return

        task = loop.create_task(self._mirror_async(src, dst, key, target))
        self._pending_mirrors.add(task)
        task.add_done_callback(self._pending_mirrors.discard)

    def _ram_usage(self) -> int:
        total = 0
        for path in self.ram.values():
            p = Path(path)
            if p.exists():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def _evict_ram_if_over_budget(self) -> None:
        total = self._ram_usage()
        while self.ram_budget and total > self.ram_budget and self.ram:
            lru_key, lru_path = self.ram.popitem(last=False)
            p = Path(lru_path)
            try:
                if p.exists():
                    total -= p.stat().st_size
                    p.unlink(missing_ok=True)
            except OSError:
                pass
            logger.info("snapshot store evicted RAM entry %s", lru_key.hex()[:8])

    def _cleanup_disk(self) -> None:
        now = datetime.now(timezone.utc)

        for key, path in list(self.disk.items()):
            p = Path(path)
            if not p.exists():
                self.disk.pop(key, None)
                continue
            if self.disk_ttl is not None:
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    self.disk.pop(key, None)
                    continue
                if now - mtime > self.disk_ttl:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.disk.pop(key, None)
                    logger.info("snapshot store expired disk entry %s", key.hex()[:8])

        if not self.disk_budget:
            return

        total = 0
        sizes: dict[bytes, int] = {}
        for key, path in list(self.disk.items()):
            p = Path(path)
            if not p.exists():
                self.disk.pop(key, None)
                continue
            try:
                size = p.stat().st_size
            except OSError:
                self.disk.pop(key, None)
                continue
            sizes[key] = size
            total += size

        while total > self.disk_budget and self.disk:
            lru_key, lru_path = self.disk.popitem(last=False)
            p = Path(lru_path)
            total -= sizes.get(lru_key, 0)
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
            logger.info("snapshot store evicted disk entry %s", lru_key.hex()[:8])

        self._save_manifest()

    def has(self, key: bytes) -> bool:
        return key in self.ram or key in self.disk

    def put_ram(self, key: bytes, path: Path, mirror_to_disk: bool = True) -> Path:
        """Register a RAM snapshot path and optionally mirror it to disk."""
        path = Path(path)
        self.ram[key] = str(path)
        self.ram.move_to_end(key)
        if mirror_to_disk:
            self._queue_mirror(path, self.disk_path(key), key, target="disk")
        self._evict_ram_if_over_budget()
        return path

    def save(self, daemon, key: bytes, slot: int) -> Path:
        """Save a daemon snapshot slot to RAM, then mirror it to disk."""
        ram_path = self.ram_path(key)
        ram_path.parent.mkdir(parents=True, exist_ok=True)
        daemon.save_snapshot(slot, str(ram_path))
        return self.put_ram(key, ram_path, mirror_to_disk=True)

    def load(self, daemon, key: bytes, slot: int) -> bool:
        """Load a snapshot into the given transient daemon slot."""
        if key in self.ram:
            path = Path(self.ram[key])
            if path.exists():
                daemon.load_snapshot(slot, str(path))
                self.ram.move_to_end(key)
                return True
            self.ram.pop(key, None)

        if key in self.disk:
            self._cleanup_disk()

        if key in self.disk:
            path = Path(self.disk[key])
            if not path.exists():
                self.disk.pop(key, None)
                self._save_manifest()
                return False
            self.disk.move_to_end(key)
            daemon.load_snapshot(slot, str(path))
            ram_path = self.ram_path(key)
            self._queue_mirror(path, ram_path, key, target="ram")
            return True
        return False

    def discard(self, key: bytes) -> None:
        """Delete a snapshot from both RAM and disk stores."""
        ram_path = self.ram.pop(key, None)
        if ram_path:
            try:
                Path(ram_path).unlink(missing_ok=True)
            except OSError:
                pass
        disk_path = self.disk.pop(key, None)
        if disk_path:
            try:
                Path(disk_path).unlink(missing_ok=True)
            except OSError:
                pass
        self._save_manifest()

    def clear(self) -> None:
        """Clear all snapshot state and files."""
        self.ram.clear()
        self.disk.clear()
        if self.ram_dir.exists():
            shutil.rmtree(self.ram_dir, ignore_errors=True)
        if self.disk_dir.exists():
            shutil.rmtree(self.disk_dir, ignore_errors=True)
        self.ram_dir.mkdir(parents=True, exist_ok=True)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._save_manifest()


DualSwap = SnapshotStore
SnapshotSwap = SnapshotStore
