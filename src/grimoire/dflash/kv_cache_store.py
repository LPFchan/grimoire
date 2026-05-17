"""Content-hash KV cache store with RAM→disk tiering.

Manages llama-server `.kv` files by content hash of the prompt prefix.
RAM (tmpfs) is the primary tier; disk (SSD) is the backup for restart
resilience. Used by proxy/llama.py for content-hash-based slot save/restore.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import struct
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "kv-cache-manifest.json"
KV_PREFIX = "kv-"
KV_SUFFIX = ".kv"


class KVCacheStore:
    def __init__(
        self,
        ram_dir: str = "/dev/shm/grimoire-slots",
        disk_dir: str = "",
        disk_budget_gb: float = 30.0,
        disk_ttl_hours: float = 24.0,
        cap: int = 8,
        kv_k_type: str = "q8_0",
        kv_v_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.ram_dir = Path(ram_dir)
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.disk_budget = int(disk_budget_gb * 1024**3) if disk_budget_gb > 0 else 0
        self.cap = cap
        self.kv_k_type = kv_k_type
        self.kv_v_type = kv_v_type
        self.fa_window = fa_window

        ttl_hours = float(disk_ttl_hours) if disk_ttl_hours > 0 else 0
        self.disk_ttl = timedelta(hours=ttl_hours) if ttl_hours > 0 else None

        # RAM index: OrderedDict[bytes, Path] — hash → file path
        self.ram: OrderedDict[bytes, Path] = OrderedDict()
        # Disk index: OrderedDict[bytes, Path] — hash → file path
        self.disk: OrderedDict[bytes, Path] = OrderedDict()

        self._pending_mirrors: set[asyncio.Task] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.ram_dir.mkdir(parents=True, exist_ok=True)
        if self.disk_dir:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            self._load_manifest()
            self._cleanup_disk()

    def hash_prefix(self, prompt_ids: list) -> bytes:
        h = hashlib.sha1()
        h.update(struct.pack("<I", len(prompt_ids)))
        h.update(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
        h.update(str(self.kv_k_type).encode())
        h.update(b"\x00")
        h.update(str(self.kv_v_type).encode())
        h.update(b"\x00")
        h.update(struct.pack("<I", self.fa_window))
        return h.digest()[:16]

    def kv_filename(self, hash_bytes: bytes) -> str:
        return f"{KV_PREFIX}{hash_bytes.hex()[:16]}{KV_SUFFIX}"

    def ram_path(self, hash_bytes: bytes) -> Path:
        return self.ram_dir / self.kv_filename(hash_bytes)

    def disk_path(self, hash_bytes: bytes) -> Path:
        if not self.disk_dir:
            return self.ram_path(hash_bytes)
        return self.disk_dir / self.kv_filename(hash_bytes)

    def _manifest_path(self) -> Path:
        if not self.disk_dir:
            return self.ram_dir / MANIFEST_NAME
        return self.disk_dir / MANIFEST_NAME

    def _load_manifest(self) -> None:
        mp = self._manifest_path()
        if not mp.exists():
            return
        try:
            with open(mp) as f:
                data = json.load(f)
            for entry in data.get("disk", []):
                hash_bytes = bytes.fromhex(entry["hash"])
                path = Path(entry["path"])
                if path.exists():
                    self.disk[hash_bytes] = path
            logger.info("kv cache manifest loaded: %s disk entries", len(self.disk))
        except Exception as e:
            logger.warning("kv cache manifest load failed: %s", e)

    def _save_manifest(self) -> None:
        if not self.disk_dir:
            return
        mp = self._manifest_path()
        data = {
            "disk": [
                {"hash": key.hex(), "path": str(path)}
                for key, path in self.disk.items()
            ]
        }
        tmp = mp.with_suffix(".tmp")
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(str(tmp), str(mp))
        except Exception as e:
            logger.warning("kv cache manifest save failed: %s", e)

    def lookup(self, hash_bytes: bytes) -> Optional[Path]:
        if hash_bytes in self.ram:
            self.ram.move_to_end(hash_bytes)
            return self.ram[hash_bytes]

        if hash_bytes in self.disk:
            self.disk.move_to_end(hash_bytes)
            return self.disk[hash_bytes]

        if not self.disk_dir:
            return None
        path = self.disk_path(hash_bytes)
        if path.exists():
            self.disk[hash_bytes] = path
            return path

        return None

    def register(self, hash_bytes: bytes) -> None:
        rp = self.ram_path(hash_bytes)
        if rp.exists():
            self.ram[hash_bytes] = rp
            self.ram.move_to_end(hash_bytes)
            if self.disk_dir:
                self._schedule_mirror(hash_bytes)

        while len(self.ram) > self.cap:
            self.ram.popitem(last=False)

        if self.disk_dir:
            self._cleanup_disk()

    def promote_to_ram(self, hash_bytes: bytes, disk_path: Path) -> Optional[Path]:
        rp = self.ram_path(hash_bytes)
        try:
            shutil.copy2(str(disk_path), str(rp))
            self.ram[hash_bytes] = rp
            self.ram.move_to_end(hash_bytes)
            return rp
        except OSError as e:
            logger.warning("kv cache promote to RAM failed: %s", e)
            return None

    def _schedule_mirror(self, hash_bytes: bytes) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
        except RuntimeError:
            loop = self._loop
        if loop is None or not loop.is_running():
            return

        task = loop.create_task(self._mirror_to_disk(hash_bytes))
        self._pending_mirrors.add(task)
        task.add_done_callback(self._pending_mirrors.discard)

    async def _mirror_to_disk(self, hash_bytes: bytes) -> None:
        if not self.disk_dir:
            return
        src = self.ram_path(hash_bytes)
        if not src.exists():
            return
        dst = self.disk_path(hash_bytes)
        tmp = dst.with_name(f".{dst.name}.{os.urandom(4).hex()}.tmp")
        try:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, str(src), str(tmp))
            os.replace(str(tmp), str(dst))
            self.disk[hash_bytes] = dst
            self.disk.move_to_end(hash_bytes)
            self._save_manifest()
        except (OSError, asyncio.CancelledError):
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_disk(self) -> None:
        if not self.disk_dir:
            return
        now = datetime.now(timezone.utc)
        tracked = set(self.disk.values())

        for f in self.disk_dir.glob(f"{KV_PREFIX}*{KV_SUFFIX}"):
            if f not in tracked:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass

        expired = []
        for key, path in list(self.disk.items()):
            if not path.exists():
                expired.append(key)
                continue
            if self.disk_ttl is not None:
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    if now - mtime > self.disk_ttl:
                        expired.append(key)
                except OSError:
                    expired.append(key)
        for key in expired:
            path = self.disk.pop(key, None)
            if path:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        if self.disk_budget <= 0:
            self._save_manifest()
            return

        disk_items = list(self.disk.items())
        total = 0
        sizes = {}
        for key, path in disk_items:
            try:
                sz = path.stat().st_size
                sizes[key] = sz
                total += sz
            except OSError:
                self.disk.pop(key, None)

        for key, _ in disk_items:
            if total <= self.disk_budget:
                break
            if key not in self.disk:
                continue
            path = self.disk.pop(key, None)
            if path:
                total -= sizes.get(key, 0)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

        self._save_manifest()

    def discard(self, hash_bytes: bytes) -> None:
        if hash_bytes in self.ram:
            path = self.ram.pop(hash_bytes)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if hash_bytes in self.disk:
            path = self.disk.pop(hash_bytes)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self._save_manifest()

    def clear(self) -> None:
        self.ram.clear()
        self.disk.clear()
        if self.ram_dir.exists():
            shutil.rmtree(self.ram_dir, ignore_errors=True)
        if self.disk_dir and self.disk_dir.exists():
            shutil.rmtree(self.disk_dir, ignore_errors=True)
        self.ram_dir.mkdir(parents=True, exist_ok=True)

    def flush_pending(self) -> None:
        for task in list(self._pending_mirrors):
            try:
                task.cancel()
            except Exception:
                pass
        self._pending_mirrors.clear()
