"""System telemetry sampling (GPU + CPU) for the dashboard endpoint.

Runs a 5-second background asyncio task that records GPU temperature and
power draw from nvidia-smi, plus CPU temperature from /sys/class/hwmon and
CPU power from RAPL /sys/class/powercap, into a SQLite ring. The dashboard
endpoint queries this ring binned to N points for the selected window.
"""

import asyncio
import glob
import logging
import os
import subprocess
import sqlite3
import time
from threading import RLock

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_PATH = os.environ.get(
    "GRIMOIRE_TELEMETRY_PATH", "/var/lib/grimoire/telemetry.sqlite3"
)
FALLBACK_TELEMETRY_PATH = os.path.expanduser("~/.local/share/grimoire/telemetry.sqlite3")

SAMPLE_INTERVAL_S = float(os.environ.get("GRIMOIRE_TELEMETRY_INTERVAL_S", "5"))
# Retention in days. 0 = keep forever (default) so lifetime graphs stay accurate.
# At 5s sampling for 5 metrics that's ~1 GB/year — set a positive value if disk
# pressure matters and you'd rather lose old GPU/CPU history past N days.
RETENTION_DAYS = int(os.environ.get("GRIMOIRE_TELEMETRY_RETENTION_DAYS", "0"))
DEFAULT_BINS = 60

CPU_HWMON_NAMES = ("k10temp", "coretemp", "zenpower")
CPU_HWMON_LABELS = ("Tctl", "Tdie", "Package id 0", "Tccd1")
FAN_HWMON_NAMES = ("nct6798", "nct6775", "nct6779", "it87", "w83627ehf")

_RAPL_ENERGY_PATH = "/host-powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_STATE = {"last_ts": None, "last_energy_uj": None}


class TelemetryStore:
    """SQLite-backed ring of system samples."""

    def __init__(self, path=DEFAULT_TELEMETRY_PATH):
        self.path = path
        self._lock = RLock()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except PermissionError:
            self.path = FALLBACK_TELEMETRY_PATH
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS system_samples (
                    ts INTEGER NOT NULL,
                    gpu_index INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_ts
                    ON system_samples(ts);
                CREATE INDEX IF NOT EXISTS idx_samples_metric_ts
                    ON system_samples(metric, gpu_index, ts);
                """
            )

    def record(self, ts, samples):
        """Insert a batch of (gpu_index, metric, value) tuples at one timestamp."""
        if not samples:
            return
        rows = [(int(ts), gpu, metric, float(value)) for gpu, metric, value in samples]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO system_samples (ts, gpu_index, metric, value) VALUES (?,?,?,?)",
                rows,
            )

    def prune(self, older_than_ts):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM system_samples WHERE ts < ?", (int(older_than_ts),))

    def latest(self, metric, gpu_index):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_samples WHERE metric=? AND gpu_index=? ORDER BY ts DESC LIMIT 1",
                (metric, gpu_index),
            ).fetchone()
        return float(row["value"]) if row else None

    def binned_avg(self, metric, gpu_index, ts_from, ts_to, bins):
        """Average per bin over [ts_from, ts_to). Returns list of length `bins`,
        with None for empty bins."""
        if ts_to <= ts_from or bins <= 0:
            return []
        width = (ts_to - ts_from) / bins
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    CAST((ts - ?) / ? AS INTEGER) AS bin,
                    AVG(value) AS avg_value
                FROM system_samples
                WHERE metric=? AND gpu_index=? AND ts >= ? AND ts < ?
                GROUP BY bin
                """,
                (ts_from, width, metric, gpu_index, ts_from, ts_to),
            ).fetchall()
        out = [None] * bins
        for r in rows:
            b = int(r["bin"])
            if 0 <= b < bins:
                out[b] = float(r["avg_value"])
        return out

    def earliest_ts(self):
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT MIN(ts) AS t FROM system_samples").fetchone()
        return int(row["t"]) if row and row["t"] is not None else None


telemetry_store = TelemetryStore()


def _read_gpu_samples():
    """Run nvidia-smi once, return list of (gpu_index, metric, value) tuples."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"nvidia-smi unavailable: {e}")
        return []
    if result.returncode != 0:
        return []
    out = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        try:
            out.append((idx, "gpu_temp", float(parts[1])))
        except ValueError:
            pass
        try:
            out.append((idx, "gpu_power", float(parts[2])))
        except ValueError:
            pass
        if len(parts) >= 4:
            try:
                out.append((idx, "gpu_vram", float(parts[3])))
            except ValueError:
                pass
    return out


def _read_cpu_temp():
    """Probe /sys/class/hwmon for a CPU temp sensor. Returns float °C or None.

    Prefers k10temp/coretemp/zenpower hwmon entries, then prefers Tctl/Tdie
    style temp_label inputs over the first available temp*_input.
    """
    for hw_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hw_dir, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name not in CPU_HWMON_NAMES:
            continue
        labelled = []
        for label_path in sorted(glob.glob(os.path.join(hw_dir, "temp*_label"))):
            input_path = label_path[: -len("_label")] + "_input"
            try:
                with open(label_path) as f:
                    label = f.read().strip()
                with open(input_path) as f:
                    value_milli = int(f.read().strip())
            except (OSError, ValueError):
                continue
            labelled.append((label, value_milli / 1000.0))
        for preferred in CPU_HWMON_LABELS:
            for label, value in labelled:
                if label == preferred:
                    return value
        if labelled:
            return labelled[0][1]
        for input_path in sorted(glob.glob(os.path.join(hw_dir, "temp*_input"))):
            try:
                with open(input_path) as f:
                    return int(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    return None


def _read_cpu_power():
    """Read RAPL package energy counter, return instantaneous power in watts.

    Reads /sys/class/powercap/intel-rapl:0/energy_uj (cumulative microjoules),
    computes delta over time.  Returns None on first call or if the interface is
    unavailable / unreadable.
    """
    try:
        with open(_RAPL_ENERGY_PATH) as f:
            energy_uj = int(f.read().strip())
    except (OSError, ValueError):
        return None
    now = time.time()
    power = None
    if _RAPL_STATE["last_ts"] is not None and _RAPL_STATE["last_energy_uj"] is not None:
        dt = now - _RAPL_STATE["last_ts"]
        if dt > 0:
            delta_uj = energy_uj - _RAPL_STATE["last_energy_uj"]
            if delta_uj < 0:
                delta_uj += 65532610987
            power = delta_uj / dt / 1e6
    _RAPL_STATE["last_ts"] = now
    _RAPL_STATE["last_energy_uj"] = energy_uj
    return power


def _read_system_ram_mb():
    """Read total system RAM usage from /proc/meminfo (MB)."""
    try:
        with open("/proc/meminfo") as f:
            lines = dict(
                line.split(":", 1) for line in f.read().strip().splitlines() if ":" in line
            )
        total = float(lines.get("MemTotal", "0 kB").strip().split()[0])
        available = float(lines.get("MemAvailable", "0 kB").strip().split()[0])
        return (total - available) / 1024.0
    except (OSError, ValueError, KeyError):
        return None


def _read_container_ram_mb():
	"""Read container anonymous memory from cgroup v2 memory.stat (MB)."""
	try:
		with open("/sys/fs/cgroup/memory.stat") as f:
			for line in f:
				if line.startswith("anon "):
					return int(line.split()[1]) / (1024 * 1024)
	except (OSError, ValueError):
		pass
	try:
		with open("/sys/fs/cgroup/memory.current") as f:
			return int(f.read().strip()) / (1024 * 1024)
	except (OSError, ValueError):
		return None


def _read_fan_rpm():
    """Read fan1_input and fan2_input from the main fan-controller hwmon chip.

    Returns a list of (fan_index, metric, rpm) tuples, or empty list.
    """
    for hw_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hw_dir, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name not in FAN_HWMON_NAMES:
            continue
        fans = []
        for fan_idx in (1, 2):
            fan_path = os.path.join(hw_dir, f"fan{fan_idx}_input")
            try:
                with open(fan_path) as f:
                    rpm = int(f.read().strip())
                fans.append((fan_idx, f"fan{fan_idx}_rpm", rpm))
            except (OSError, ValueError):
                pass
        if fans:
            return fans
    return []


def collect_one_sample():
    """Synchronous one-shot sample. Returns the rows that were recorded."""
    rows = _read_gpu_samples()
    cpu_temp = _read_cpu_temp()
    if cpu_temp is not None:
        rows.append((0, "cpu_temp", cpu_temp))
    cpu_power = _read_cpu_power()
    if cpu_power is not None and cpu_power >= 0:
        rows.append((0, "cpu_power", cpu_power))
    for fan_idx, metric, rpm in _read_fan_rpm():
        rows.append((0, metric, rpm))
    sys_ram = _read_system_ram_mb()
    if sys_ram is not None:
        rows.append((0, "system_ram_mb", sys_ram))
    ctr_ram = _read_container_ram_mb()
    if ctr_ram is not None:
        rows.append((0, "container_ram_mb", ctr_ram))
    if rows:
        telemetry_store.record(time.time(), rows)
    return rows


async def telemetry_sampler():
    """Background task: sample every SAMPLE_INTERVAL_S, prune only if retention is set."""
    last_prune = 0.0
    while True:
        try:
            await asyncio.to_thread(collect_one_sample)
            if RETENTION_DAYS > 0:
                now = time.time()
                if now - last_prune > 3600:
                    cutoff = now - RETENTION_DAYS * 86400
                    await asyncio.to_thread(telemetry_store.prune, cutoff)
                    last_prune = now
        except Exception:
            logger.exception("Telemetry sample failed")
        await asyncio.sleep(SAMPLE_INTERVAL_S)
