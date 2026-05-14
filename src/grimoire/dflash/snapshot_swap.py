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
import uuid
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
        self._pending_mirror_keys: dict[bytes, set[asyncio.Task]] = {}
        self._pending_spawns = 0
        self._state_lock = threading.Lock()
        self._clear_epoch = 0
        self._key_epochs: dict[bytes, int] = {}
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
        with self._state_lock:
            disk_items = list(self.disk.items())
        data = {
            "disk": [
                {"key_hex": key.hex(), "path": path}
                for key, path in disk_items
            ]
        }
        tmp = mp.with_suffix(".tmp")
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
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

    def _mirror_token(self, key: bytes) -> tuple[int, int]:
        with self._state_lock:
            return (self._clear_epoch, self._key_epochs.get(key, 0))

    def _token_is_current(self, key: bytes, token: tuple[int, int]) -> bool:
        with self._state_lock:
            return token == (self._clear_epoch, self._key_epochs.get(key, 0))

    def _schedule_mirror(
        self,
        loop: asyncio.AbstractEventLoop,
        src: Path,
        dst: Path,
        key: bytes,
        target: str,
        token: tuple[int, int],
    ) -> None:
        with self._state_lock:
            self._pending_spawns += 1

        def _spawn() -> None:
            try:
                self._track_mirror_task(loop.create_task(self._mirror_async(src, dst, key, target, token)), key)
            finally:
                with self._state_lock:
                    self._pending_spawns = max(0, self._pending_spawns - 1)

        try:
            loop.call_soon_threadsafe(_spawn)
        except RuntimeError:
            with self._state_lock:
                self._pending_spawns = max(0, self._pending_spawns - 1)
            logger.warning(
                "snapshot mirror skipped: loop unavailable during schedule (key=%s target=%s)",
                key.hex()[:8],
                target,
            )

    def _track_mirror_task(self, task: asyncio.Task, key: bytes) -> None:
        with self._state_lock:
            self._pending_mirrors.add(task)
            self._pending_mirror_keys.setdefault(key, set()).add(task)

        def _done(done_task: asyncio.Task) -> None:
            with self._state_lock:
                self._pending_mirrors.discard(done_task)
                tasks = self._pending_mirror_keys.get(key)
                if tasks is not None:
                    tasks.discard(done_task)
                    if not tasks:
                        self._pending_mirror_keys.pop(key, None)

        task.add_done_callback(_done)

    def _cancel_pending_for_key(self, key: bytes) -> None:
        with self._state_lock:
            self._key_epochs[key] = self._key_epochs.get(key, 0) + 1
            tasks = tuple(self._pending_mirror_keys.get(key, ()))
        for task in tasks:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _cancel_all_pending(self) -> None:
        with self._state_lock:
            self._clear_epoch += 1
            self._key_epochs.clear()
            tasks = tuple(self._pending_mirrors)
        for task in tasks:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _publish_mirror(self, key: bytes, dst: Path, target: str, token: tuple[int, int]) -> bool:
        with self._state_lock:
            if token != (self._clear_epoch, self._key_epochs.get(key, 0)):
                return False
            if target == "disk":
                self.disk[key] = str(dst)
                self.disk.move_to_end(key)
            else:
                self.ram[key] = str(dst)
                self.ram.move_to_end(key)
            return True

    async def _mirror_async(self, src: Path, dst: Path, key: bytes, target: str, token: tuple[int, int]) -> None:
        tmp_dst = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, src, tmp_dst)
            if not self._token_is_current(key, token):
                return
            await asyncio.to_thread(os.replace, str(tmp_dst), str(dst))
            if not self._publish_mirror(key, dst, target, token):
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            if target == "disk":
                self._cleanup_disk()
            else:
                self._evict_ram_if_over_budget()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("snapshot mirror failed for %s: %s", key.hex()[:8], e)
        finally:
            try:
                tmp_dst.unlink(missing_ok=True)
            except OSError:
                pass

    def _queue_mirror(self, src: Path, dst: Path, key: bytes, target: str = "disk") -> None:
        token = self._mirror_token(key)
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
            self._schedule_mirror(loop, src, dst, key, target, token)
            return

        self._track_mirror_task(loop.create_task(self._mirror_async(src, dst, key, target, token)), key)

    def _ram_usage(self) -> int:
        with self._state_lock:
            ram_paths = list(self.ram.values())
        total = 0
        for path in ram_paths:
            p = Path(path)
            if p.exists():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def _evict_ram_if_over_budget(self) -> None:
        total = self._ram_usage()
        while self.ram_budget and total > self.ram_budget:
            with self._state_lock:
                if not self.ram:
                    break
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
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)

        with self._state_lock:
            tracked_paths = {Path(path) for path in self.disk.values()}
            disk_items = list(self.disk.items())
        for stray in self.disk_dir.glob("swap-*.dfsn"):
            if stray in tracked_paths:
                continue
            try:
                stray.unlink(missing_ok=True)
            except OSError:
                pass

        for key, path in disk_items:
            p = Path(path)
            if not p.exists():
                with self._state_lock:
                    self.disk.pop(key, None)
                continue
            if self.disk_ttl is not None:
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    with self._state_lock:
                        self.disk.pop(key, None)
                    continue
                if now - mtime > self.disk_ttl:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                    with self._state_lock:
                        self.disk.pop(key, None)
                    logger.info("snapshot store expired disk entry %s", key.hex()[:8])

        if not self.disk_budget:
            return

        total = 0
        sizes: dict[bytes, int] = {}
        with self._state_lock:
            disk_items = list(self.disk.items())
        for key, path in disk_items:
            p = Path(path)
            if not p.exists():
                with self._state_lock:
                    self.disk.pop(key, None)
                continue
            try:
                size = p.stat().st_size
            except OSError:
                with self._state_lock:
                    self.disk.pop(key, None)
                continue
            sizes[key] = size
            total += size

        while total > self.disk_budget:
            with self._state_lock:
                if not self.disk:
                    break
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
        with self._state_lock:
            return key in self.ram or key in self.disk

    def put_ram(self, key: bytes, path: Path, mirror_to_disk: bool = True) -> Path:
        """Register a RAM snapshot path and optionally mirror it to disk."""
        path = Path(path)
        with self._state_lock:
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
        with self._state_lock:
            ram_path = self.ram.get(key)
        if ram_path is not None:
            path = Path(ram_path)
            if path.exists():
                daemon.load_snapshot(slot, str(path))
                with self._state_lock:
                    if key in self.ram:
                        self.ram.move_to_end(key)
                return True
            with self._state_lock:
                self.ram.pop(key, None)

        with self._state_lock:
            in_disk = key in self.disk
        if in_disk:
            self._cleanup_disk()

        with self._state_lock:
            disk_path = self.disk.get(key)
        if disk_path is not None:
            path = Path(disk_path)
            if not path.exists():
                with self._state_lock:
                    self.disk.pop(key, None)
                self._save_manifest()
                return False
            with self._state_lock:
                if key in self.disk:
                    self.disk.move_to_end(key)
            daemon.load_snapshot(slot, str(path))
            ram_path = self.ram_path(key)
            self._queue_mirror(path, ram_path, key, target="ram")
            return True
        return False

    def discard(self, key: bytes) -> None:
        """Delete a snapshot from both RAM and disk stores."""
        self._cancel_pending_for_key(key)
        with self._state_lock:
            ram_path = self.ram.pop(key, None)
            disk_path = self.disk.pop(key, None)
        if ram_path:
            try:
                Path(ram_path).unlink(missing_ok=True)
            except OSError:
                pass
        if disk_path:
            try:
                Path(disk_path).unlink(missing_ok=True)
            except OSError:
                pass
        self._save_manifest()

    def clear(self) -> None:
        """Clear all snapshot state and files."""
        self._cancel_all_pending()
        with self._state_lock:
            self.ram.clear()
            self.disk.clear()
        if self.ram_dir.exists():
            shutil.rmtree(self.ram_dir, ignore_errors=True)
        if self.disk_dir.exists():
            shutil.rmtree(self.disk_dir, ignore_errors=True)
        self.ram_dir.mkdir(parents=True, exist_ok=True)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._save_manifest()

    async def flush_pending(self) -> None:
        """Wait for any in-flight mirror tasks to finish or cancel."""
        while True:
            with self._state_lock:
                pending = tuple(self._pending_mirrors)
                pending_spawns = self._pending_spawns
            if not pending and pending_spawns == 0:
                return
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            else:
                await asyncio.sleep(0)

    def flush_pending_sync(self, timeout: Optional[float] = None) -> None:
        """Synchronously wait for pending mirrors from a non-loop thread."""
        with self._state_lock:
            if not self._pending_mirrors and self._pending_spawns == 0:
                return
            loop = self._loop
        if loop is None or not loop.is_running():
            return
        if self._loop_thread_id == threading.get_ident():
            logger.warning("snapshot flush skipped from bound event-loop thread")
            return
        future = asyncio.run_coroutine_threadsafe(self.flush_pending(), loop)
        future.result(timeout=timeout)


DualSwap = SnapshotStore
SnapshotSwap = SnapshotStore
