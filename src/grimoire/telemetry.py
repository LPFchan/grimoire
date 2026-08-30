"""System telemetry sampling (GPU + CPU) for the dashboard endpoint.

Runs a 5-second background asyncio task that records GPU temperature and
power draw from nvidia-smi, plus CPU temperature from /sys/class/hwmon and
CPU power from RAPL /sys/class/powercap, into a SQLite ring. The dashboard
endpoint queries this ring binned to N points for the selected window.

Two-tier storage
----------------
Raw 5-second samples answer short windows. Every raw sample older than one
bucket is also folded into `system_rollup`, a per-minute (sum, count) table that
answers long windows. The dashboard only ever draws 60 points, so anything wider
than an hour gains nothing from 5-second resolution and pays a full table scan
for it: at 18M raw rows a single "all" window cost ~15s of scanning per metric
series. Reading the rollup instead keeps every window in the millisecond range.

Storing sum and count rather than a pre-divided average lets arbitrary bin
widths be re-aggregated without skew from unevenly populated buckets.
"""

import asyncio
import glob
import logging
import os
import shutil
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
# Retention in days for RAW samples only. 0 = keep forever (default). Rollups are
# never pruned, so lifetime graphs stay accurate even with retention on: setting
# this only costs 5-second resolution on windows older than N days.
RETENTION_DAYS = int(os.environ.get("GRIMOIRE_TELEMETRY_RETENTION_DAYS", "0"))
DEFAULT_BINS = 60

# Rollup bucket width. 60s over 60 bins means any window of an hour or more is
# served from the rollup.
ROLLUP_BUCKET_S = 60
# Rows of raw samples folded per backfill pass, so the first run against a large
# existing database makes progress without holding the write lock for minutes.
ROLLUP_BACKFILL_CHUNK_S = 6 * 3600

CPU_HWMON_NAMES = ("k10temp", "coretemp", "zenpower")
CPU_HWMON_LABELS = ("Tctl", "Tdie", "Package id 0", "Tccd1")
FAN_HWMON_NAMES = ("nct6798", "nct6775", "nct6779", "it87", "w83627ehf")

_RAPL_ENERGY_PATH = "/host-powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_MAX_ENERGY_PATH = "/host-powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
# Fallback for hardware that doesn't expose max_energy_range_uj. Matches the
# value seen on common Intel package counters; off-by-this-much per wrap on
# anything else, which happens about once per several weeks at steady draw.
_RAPL_WRAP_FALLBACK_UJ = 65532610987
_RAPL_STATE = {"last_ts": None, "last_energy_uj": None, "max_energy_uj": None}


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
        # WAL lets the dashboard read while the sampler writes. Under the default
        # rollback journal a long read holds a shared lock that stalls the
        # 5-second sampler, and vice versa.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
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

                -- Per-minute aggregate serving every window wider than an hour.
                -- Sum and count rather than an average so bins of any width can
                -- be recombined without weighting error.
                CREATE TABLE IF NOT EXISTS system_rollup (
                    metric TEXT NOT NULL,
                    gpu_index INTEGER NOT NULL,
                    bucket_ts INTEGER NOT NULL,
                    sum_value REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    PRIMARY KEY (metric, gpu_index, bucket_ts)
                ) WITHOUT ROWID;

                -- Single-row watermark: raw samples strictly below this have
                -- already been folded into system_rollup.
                CREATE TABLE IF NOT EXISTS rollup_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    rolled_through_ts INTEGER NOT NULL
                );
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
        """Drop raw samples older than a cutoff, never past the rollup watermark.

        Clamping to the watermark means retention can never discard a sample
        that has not yet been folded into the aggregate.
        """
        with self._lock, self._connect() as conn:
            watermark = self._rolled_through(conn)
            cutoff = int(older_than_ts)
            if watermark is not None:
                cutoff = min(cutoff, watermark)
            conn.execute("DELETE FROM system_samples WHERE ts < ?", (cutoff,))

    @staticmethod
    def _rolled_through(conn):
        """Bucket-aligned timestamp below which every raw sample is rolled up."""
        row = conn.execute("SELECT rolled_through_ts FROM rollup_state WHERE id = 1").fetchone()
        return int(row["rolled_through_ts"]) if row else None

    def roll_up(self, now_ts=None):
        """Fold one chunk of raw samples into the per-minute rollup.

        Returns True while there is more history left to fold, so a backfill can
        drive this in a loop without any single pass holding the write lock for
        long. Chunk boundaries are bucket-aligned, which makes re-running a chunk
        after a crash idempotent rather than double-counting.
        """
        now = time.time() if now_ts is None else now_ts
        # Only fold buckets that can no longer receive samples.
        complete_through = int(now // ROLLUP_BUCKET_S) * ROLLUP_BUCKET_S

        with self._lock, self._connect() as conn:
            start = self._rolled_through(conn)
            if start is None:
                row = conn.execute("SELECT MIN(ts) AS t FROM system_samples").fetchone()
                if not row or row["t"] is None:
                    return False
                start = int(row["t"]) // ROLLUP_BUCKET_S * ROLLUP_BUCKET_S

            end = min(start + ROLLUP_BACKFILL_CHUNK_S, complete_through)
            if end <= start:
                return False

            conn.execute(
                """
                INSERT INTO system_rollup (metric, gpu_index, bucket_ts, sum_value, sample_count)
                SELECT metric, gpu_index, (ts / ?) * ?, SUM(value), COUNT(*)
                FROM system_samples
                WHERE ts >= ? AND ts < ?
                GROUP BY metric, gpu_index, (ts / ?) * ?
                ON CONFLICT (metric, gpu_index, bucket_ts) DO UPDATE SET
                    sum_value = excluded.sum_value,
                    sample_count = excluded.sample_count
                """,
                (ROLLUP_BUCKET_S, ROLLUP_BUCKET_S, start, end, ROLLUP_BUCKET_S, ROLLUP_BUCKET_S),
            )
            conn.execute(
                """
                INSERT INTO rollup_state (id, rolled_through_ts) VALUES (1, ?)
                ON CONFLICT (id) DO UPDATE SET rolled_through_ts = excluded.rolled_through_ts
                """,
                (end,),
            )

        return end < complete_through

    def latest(self, metric, gpu_index):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_samples WHERE metric=? AND gpu_index=? ORDER BY ts DESC LIMIT 1",
                (metric, gpu_index),
            ).fetchone()
        return float(row["value"]) if row else None

    @staticmethod
    def _raw_bins(conn, metric, gpu_index, ts_from, ts_to, origin, width):
        return conn.execute(
            """
            SELECT
                CAST((ts - ?) / ? AS INTEGER) AS bin,
                SUM(value) AS sum_value,
                COUNT(*) AS sample_count
            FROM system_samples
            WHERE metric=? AND gpu_index=? AND ts >= ? AND ts < ?
            GROUP BY bin
            """,
            (origin, width, metric, gpu_index, ts_from, ts_to),
        ).fetchall()

    @staticmethod
    def _rollup_bins(conn, metric, gpu_index, ts_from, ts_to, origin, width):
        return conn.execute(
            """
            SELECT
                CAST((bucket_ts - ?) / ? AS INTEGER) AS bin,
                SUM(sum_value) AS sum_value,
                SUM(sample_count) AS sample_count
            FROM system_rollup
            WHERE metric=? AND gpu_index=? AND bucket_ts >= ? AND bucket_ts < ?
            GROUP BY bin
            """,
            (origin, width, metric, gpu_index, ts_from, ts_to),
        ).fetchall()

    def binned_avg(self, metric, gpu_index, ts_from, ts_to, bins):
        """Average per bin over [ts_from, ts_to). Returns list of length `bins`,
        with None for empty bins.

        Bins at least one rollup bucket wide are served from the aggregate, which
        is what keeps long windows cheap. The window is split at the rollup
        watermark so a partially backfilled database still reads correctly: older
        than the watermark comes from the aggregate, newer from raw samples.

        On the aggregate path the bin origin is snapped down to a bucket
        boundary. Without that, a bucket straddling a bin edge is credited
        wholesale to one side, which at the 1h window (bins exactly one bucket
        wide) moved values by as much as 17%. Snapped, every standard window has
        a bin width that is a whole number of buckets, so bins and buckets nest
        exactly and the averages are the same ones the raw rows would give. The
        cost is that the plotted range can start up to one bucket early.
        """
        if ts_to <= ts_from or bins <= 0:
            return []

        width = (ts_to - ts_from) / bins
        sums = [0.0] * bins
        counts = [0] * bins

        def accumulate(rows):
            for r in rows:
                b = int(r["bin"])
                if 0 <= b < bins and r["sample_count"]:
                    sums[b] += float(r["sum_value"])
                    counts[b] += int(r["sample_count"])

        with self._lock, self._connect() as conn:
            if width >= ROLLUP_BUCKET_S:
                # Align bins to bucket boundaries; the watermark already is one.
                origin = int(ts_from // ROLLUP_BUCKET_S) * ROLLUP_BUCKET_S
                watermark = self._rolled_through(conn)
                split = origin if watermark is None else max(origin, min(ts_to, watermark))
                if split > origin:
                    accumulate(
                        self._rollup_bins(conn, metric, gpu_index, origin, split, origin, width)
                    )
                if split < ts_to:
                    accumulate(
                        self._raw_bins(conn, metric, gpu_index, split, ts_to, origin, width)
                    )
            else:
                accumulate(
                    self._raw_bins(conn, metric, gpu_index, ts_from, ts_to, ts_from, width)
                )

        return [sums[i] / counts[i] if counts[i] else None for i in range(bins)]

    def earliest_ts(self):
        """Oldest timestamp still represented, in raw samples or in the rollup."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(t) AS t FROM (
                    SELECT MIN(ts) AS t FROM system_samples
                    UNION ALL
                    SELECT MIN(bucket_ts) AS t FROM system_rollup
                )
                """
            ).fetchone()
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
    if _RAPL_STATE["max_energy_uj"] is None:
        try:
            with open(_RAPL_MAX_ENERGY_PATH) as f:
                _RAPL_STATE["max_energy_uj"] = int(f.read().strip())
        except (OSError, ValueError):
            _RAPL_STATE["max_energy_uj"] = _RAPL_WRAP_FALLBACK_UJ
    now = time.time()
    power = None
    if _RAPL_STATE["last_ts"] is not None and _RAPL_STATE["last_energy_uj"] is not None:
        dt = now - _RAPL_STATE["last_ts"]
        if dt > 0:
            delta_uj = energy_uj - _RAPL_STATE["last_energy_uj"]
            if delta_uj < 0:
                delta_uj += _RAPL_STATE["max_energy_uj"]
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


def _read_disk_usage_pct():
    """Read root filesystem usage percentage via shutil.disk_usage."""
    try:
        usage = shutil.disk_usage("/")
        return usage.used / usage.total * 100.0
    except OSError:
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
    disk_pct = _read_disk_usage_pct()
    if disk_pct is not None:
        rows.append((0, "disk_used_pct", disk_pct))
    if rows:
        telemetry_store.record(time.time(), rows)
    return rows


def backfill_rollup():
    """Fold all existing history into the rollup, one chunk at a time.

    Runs once at startup. On a database that predates the rollup this walks
    months of samples, so it is deliberately chunked: each pass takes the write
    lock briefly and releases it, leaving the sampler free to keep recording.
    """
    passes = 0
    started = time.monotonic()
    while telemetry_store.roll_up():
        passes += 1
        if passes % 100 == 0:
            logger.info("Telemetry rollup backfill: %d chunks folded", passes)
    if passes:
        logger.info(
            "Telemetry rollup backfill complete: %d chunks in %.1fs",
            passes, time.monotonic() - started,
        )


async def telemetry_sampler():
    """Background task: sample every SAMPLE_INTERVAL_S, keep the rollup current,
    prune raw samples only if retention is set."""
    last_prune = 0.0

    try:
        await asyncio.to_thread(backfill_rollup)
    except Exception:
        logger.exception("Telemetry rollup backfill failed")

    while True:
        try:
            await asyncio.to_thread(collect_one_sample)
            await asyncio.to_thread(telemetry_store.roll_up)
            if RETENTION_DAYS > 0:
                now = time.time()
                if now - last_prune > 3600:
                    cutoff = now - RETENTION_DAYS * 86400
                    await asyncio.to_thread(telemetry_store.prune, cutoff)
                    last_prune = now
        except Exception:
            logger.exception("Telemetry sample failed")
        await asyncio.sleep(SAMPLE_INTERVAL_S)
