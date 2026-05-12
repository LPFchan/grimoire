"""System monitoring utilities for end-to-end tests.

Provides periodic logging of RAM, disk, GPU, and snapshot-store state so that
if a test crashes (OOM, disk full, GPU OOM) the last log line before death
pinpoints the cause.

Usage in a test:

    from tests._monitor import SystemMonitor

    monitor = SystemMonitor(period_sec=5.0)
    monitor.start()
    try:
        # ... long running test ...
    finally:
        monitor.stop()

Or use the context manager:

    with SystemMonitor(period_sec=5.0):
        # ... long running test ...
"""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SystemSnapshot:
    """A single point-in-time measurement."""
    timestamp: float
    host_ram_used_mb: int = 0
    host_ram_total_mb: int = 0
    host_ram_percent: float = 0.0
    container_ram_used_mb: int = 0
    container_ram_limit_mb: int = 0
    container_ram_percent: float = 0.0
    gpu_vram_used_mb: int = 0
    gpu_vram_total_mb: int = 0
    gpu_vram_percent: float = 0.0
    snapshot_ram_mb: int = 0
    snapshot_disk_mb: int = 0
    root_disk_used_gb: float = 0.0
    root_disk_free_gb: float = 0.0
    root_disk_percent: float = 0.0
    loadavg_1m: float = 0.0


class SystemMonitor:
    """Periodically log system state to stdout while a test runs."""

    def __init__(
        self,
        period_sec: float = 5.0,
        container_name: str = "grimoire",
        gpu_id: int = 0,
    ):
        self.period_sec = period_sec
        self.container_name = container_name
        self.gpu_id = gpu_id
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.history: list[SystemSnapshot] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
        except Exception:
            return ""

    def _host_ram(self) -> tuple[int, int, float]:
        """Return (used_mb, total_mb, percent)."""
        out = self._run(["free", "-m"])
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total = int(parts[1])
                used = int(parts[2])
                return used, total, (used / total * 100) if total else 0.0
        return 0, 0, 0.0

    def _container_ram(self) -> tuple[int, int, float]:
        """Return (used_mb, limit_mb, percent) for the Docker container."""
        # cgroup v2 path inside the container
        out = self._run([
            "docker", "exec", self.container_name,
            "bash", "-c",
            "cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0"
        ])
        try:
            used_bytes = int(out.strip().split()[0])
        except (ValueError, IndexError):
            used_bytes = 0

        out_limit = self._run([
            "docker", "exec", self.container_name,
            "bash", "-c",
            "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0"
        ])
        try:
            limit_bytes = int(out_limit.strip().split()[0])
        except (ValueError, IndexError):
            limit_bytes = 0

        used_mb = used_bytes // (1024 * 1024)
        limit_mb = limit_bytes // (1024 * 1024) if limit_bytes < (1 << 60) else 0  # max = no limit
        pct = (used_mb / limit_mb * 100) if limit_mb else 0.0
        return used_mb, limit_mb, pct

    def _gpu_vram(self) -> tuple[int, int, float]:
        """Return (used_mb, total_mb, percent) for the target GPU."""
        out = self._run([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i", str(self.gpu_id),
        ])
        try:
            used, total = out.strip().split(",")
            used_mb = int(used.strip())
            total_mb = int(total.strip())
            return used_mb, total_mb, (used_mb / total_mb * 100) if total_mb else 0.0
        except Exception:
            return 0, 0, 0.0

    def _snapshot_store(self) -> tuple[int, int]:
        """Return (ram_mb, disk_mb) for snapshot stores inside the container."""
        ram_mb = 0
        disk_mb = 0
        try:
            out = self._run(["docker", "exec", self.container_name, "du", "-sm", "/dev/shm/grimoire-snapshots/"])
            ram_mb = int(out.split()[0])
        except Exception:
            pass
        try:
            out = self._run(["docker", "exec", self.container_name, "du", "-sm", "/var/lib/grimoire/snapshot_swap/"])
            disk_mb = int(out.split()[0])
        except Exception:
            pass
        return ram_mb, disk_mb

    def _root_disk(self) -> tuple[float, float, float]:
        """Return (used_gb, free_gb, percent)."""
        out = self._run(["df", "-BG", "/"])
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 6:
                used = int(parts[2].replace("G", ""))
                avail = int(parts[3].replace("G", ""))
                pct = float(parts[4].replace("%", ""))
                return used, avail, pct
        return 0.0, 0.0, 0.0

    def _loadavg(self) -> float:
        try:
            with open("/proc/loadavg") as f:
                return float(f.read().split()[0])
        except Exception:
            return 0.0

    def _sample(self) -> SystemSnapshot:
        host_used, host_total, host_pct = self._host_ram()
        container_used, container_limit, container_pct = self._container_ram()
        gpu_used, gpu_total, gpu_pct = self._gpu_vram()
        snapshot_ram, snapshot_disk = self._snapshot_store()
        root_used, root_free, root_pct = self._root_disk()
        return SystemSnapshot(
            timestamp=time.monotonic(),
            host_ram_used_mb=host_used,
            host_ram_total_mb=host_total,
            host_ram_percent=host_pct,
            container_ram_used_mb=container_used,
            container_ram_limit_mb=container_limit,
            container_ram_percent=container_pct,
            gpu_vram_used_mb=gpu_used,
            gpu_vram_total_mb=gpu_total,
            gpu_vram_percent=gpu_pct,
            snapshot_ram_mb=snapshot_ram,
            snapshot_disk_mb=snapshot_disk,
            root_disk_used_gb=root_used,
            root_disk_free_gb=root_free,
            root_disk_percent=root_pct,
            loadavg_1m=self._loadavg(),
        )

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def _loop(self):
        while not self._stop_event.is_set():
            snap = self._sample()
            with self._lock:
                self.history.append(snap)
            # Compact single-line log
            print(
                f"[MONITOR] "
                f"HostRAM={snap.host_ram_used_mb}MB/{snap.host_ram_percent:.0f}% "
                f"ContainerRAM={snap.container_ram_used_mb}MB/{snap.container_ram_percent:.0f}% "
                f"GPU={snap.gpu_vram_used_mb}MB/{snap.gpu_vram_percent:.0f}% "
                f"SnapRAM={snap.snapshot_ram_mb}MB SnapDisk={snap.snapshot_disk_mb}MB "
                f"RootDisk={snap.root_disk_used_gb}GB/{snap.root_disk_percent:.0f}% "
                f"Load={snap.loadavg_1m:.1f}",
                file=sys.stdout,
                flush=True,
            )
            # Warn on pressure
            if snap.container_ram_percent > 85:
                print(f"[WARN] Container RAM at {snap.container_ram_percent:.0f}% — near OOM", file=sys.stdout, flush=True)
            if snap.gpu_vram_percent > 90:
                print(f"[WARN] GPU VRAM at {snap.gpu_vram_percent:.0f}% — near OOM", file=sys.stdout, flush=True)
            if snap.root_disk_percent > 90:
                print(f"[WARN] Root disk at {snap.root_disk_percent:.0f}% — near full", file=sys.stdout, flush=True)

            self._stop_event.wait(self.period_sec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.period_sec + 2)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_history(self) -> list[SystemSnapshot]:
        with self._lock:
            return list(self.history)

    def write_report(self, path: str | Path):
        """Dump full history as JSON for post-mortem analysis."""
        data = [
            {
                "timestamp": s.timestamp,
                "host_ram_used_mb": s.host_ram_used_mb,
                "host_ram_percent": s.host_ram_percent,
                "container_ram_used_mb": s.container_ram_used_mb,
                "container_ram_percent": s.container_ram_percent,
                "gpu_vram_used_mb": s.gpu_vram_used_mb,
                "gpu_vram_percent": s.gpu_vram_percent,
                "snapshot_ram_mb": s.snapshot_ram_mb,
                "snapshot_disk_mb": s.snapshot_disk_mb,
                "root_disk_used_gb": s.root_disk_used_gb,
                "root_disk_percent": s.root_disk_percent,
                "loadavg_1m": s.loadavg_1m,
            }
            for s in self.get_history()
        ]
        Path(path).write_text(json.dumps(data, indent=2))
