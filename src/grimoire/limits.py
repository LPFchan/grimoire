"""Absolute chart scales for the dashboard.

A sparkline scaled to its own min and max is unreadable across cards: a CPU
temperature drifting between 40 and 42 degrees fills the card exactly like a GPU
swinging from idle to its power limit. Worse, an all-zero series has no range at
all and used to be drawn as a filled block at half height, which looks like data.

So each series gets a fixed floor and, where a real ceiling exists, a fixed
ceiling. Nothing here is a hard-coded number for one machine: ceilings are read
from the hardware when it will say (nvidia-smi, /proc/meminfo, the cgroup), and
otherwise derived from what this host has actually been observed doing, rounded
up to a round number. Move the code to a different box and the scales follow.
"""

import glob
import logging
import math
import os
import re
import subprocess
import time
from threading import Lock

logger = logging.getLogger(__name__)

# Temperature charts start at room temperature rather than absolute zero. A CPU
# never approaches 0 C, so a zero-based axis would waste half of every card.
AMBIENT_FLOOR_C = 20

# How long a probed or derived limit is reused before being recomputed. Hardware
# limits never move; derived ones only creep upward as new peaks arrive.
CACHE_TTL_S = 600

# Metrics with no discoverable hardware ceiling, derived from observed peaks.
# `step` rounds the ceiling up to something legible; `min_span` stops an idle
# machine from producing a scale so tight that ordinary use clips.
_DERIVED = {
    "cpu_temp": {"floor": AMBIENT_FLOOR_C, "step": 5, "min_span": 40},
    "cpu_power": {"floor": 0, "step": 10, "min_span": 30},
    "fan1_rpm": {"floor": 0, "step": 250, "min_span": 1000},
    "fan2_rpm": {"floor": 0, "step": 250, "min_span": 1000},
}

# Metrics deliberately left with a floating ceiling. Throughput depends entirely
# on which model is loaded — a 0.6B embedder and a 31B chat model share no scale
# — and the token and cost series accumulate across the window, so their top
# depends on how busy the window was.
_FLOOR_ONLY = {"gpu_tokens_per_sec"}

_cache = {}
_cache_lock = Lock()


def _cached(key, produce):
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = produce()
    with _cache_lock:
        _cache[key] = (now + CACHE_TTL_S, value)
    return value


def _round_up(value, step):
    return int(math.ceil(value / step) * step)


def _nvidia_smi(args):
    try:
        result = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        logger.debug("nvidia-smi %s unavailable: %s", args, e)
    return None


def _gpu_hardware_limits():
    """Per-GPU power limit (W), VRAM total (MB) and max operating temp (C)."""
    out = {}

    csv = _nvidia_smi([
        "--query-gpu=index,power.limit,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if csv:
        for line in csv.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            entry = out.setdefault(idx, {})
            for key, raw in (("power", parts[1]), ("vram", parts[2])):
                try:
                    entry[key] = float(raw)
                except ValueError:
                    pass

    # Thermal ceilings are only in the verbose dump, one block per GPU.
    dump = _nvidia_smi(["-q", "-d", "TEMPERATURE"])
    if dump:
        blocks = re.split(r"\nGPU [0-9a-fA-F:.]+\n", "\n" + dump)[1:]
        for idx, block in enumerate(blocks):
            m = re.search(r"GPU Max Operating Temp\s*:\s*(\d+)", block)
            if m:
                out.setdefault(idx, {})["temp"] = float(m.group(1))

    return out


def _host_memory_total_mb():
    """Denominator for system_ram_mb, which reports MemTotal minus MemAvailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_memory_max_mb():
    """Denominator for container_ram_mb — this process's own memory ceiling."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        # An unconstrained cgroup reports a sentinel near the word size.
        if value > 0 and value < (1 << 62):
            return value / (1024 * 1024)
    # No limit set: the host's RAM is the real ceiling.
    return _host_memory_total_mb()


def _derived_ceiling(metric, gpu_index, observed_max):
    spec = _DERIVED[metric]
    if observed_max is None:
        return None
    ceiling = _round_up(observed_max, spec["step"])
    return max(ceiling, spec["floor"] + spec["min_span"])


def scale_for(metric, gpu_index, observed_max=None):
    """Return {"min": float, "max": float|None} for a metric, or None for auto.

    A null `max` means "anchor the floor, let the top follow the data".
    """
    if metric in _FLOOR_ONLY:
        return {"min": 0, "max": None}

    if metric == "disk_used_pct":
        return {"min": 0, "max": 100}

    if metric.startswith("gpu_"):
        limits = _cached("gpu", _gpu_hardware_limits).get(gpu_index, {})
        if metric == "gpu_power" and "power" in limits:
            return {"min": 0, "max": limits["power"]}
        if metric == "gpu_vram" and "vram" in limits:
            return {"min": 0, "max": limits["vram"]}
        if metric == "gpu_temp" and "temp" in limits:
            return {"min": AMBIENT_FLOOR_C, "max": limits["temp"]}
        return None

    if metric == "system_ram_mb":
        total = _cached("host_mem", _host_memory_total_mb)
        return {"min": 0, "max": total} if total else None

    if metric == "container_ram_mb":
        total = _cached("cgroup_mem", _cgroup_memory_max_mb)
        return {"min": 0, "max": total} if total else None

    if metric in _DERIVED:
        ceiling = _derived_ceiling(metric, gpu_index, observed_max)
        if ceiling is None:
            return None
        return {"min": _DERIVED[metric]["floor"], "max": ceiling}

    return None


# Metrics that should share one ceiling, so the cards can be read against each
# other. Two case fans on different axes invite exactly the wrong comparison.
_SHARED_CEILING = (("fan1_rpm", "fan2_rpm"),)


def peer_metrics(metric):
    """Metrics whose peaks jointly set this metric's ceiling, or () if none."""
    if metric not in _DERIVED:
        return ()
    for group in _SHARED_CEILING:
        if metric in group:
            return group
    return (metric,)


def needs_observed_max(metric):
    """Whether scale_for wants an observed peak for this metric."""
    return metric in _DERIVED
