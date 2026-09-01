"""Absolute chart scales.

Every ceiling here has to come from the machine, not from a constant someone
typed for one host: read from the hardware when it will say, and otherwise
derived from what this host has actually done. These tests pin that, and pin the
cases that must deliberately keep a floating ceiling.
"""

import pytest

from grimoire import limits


@pytest.fixture(autouse=True)
def _clear_cache():
    limits._cache.clear()
    yield
    limits._cache.clear()


def _stub_gpu(monkeypatch, table):
    monkeypatch.setattr(limits, "_gpu_hardware_limits", lambda: table)


def test_gpu_ceilings_come_from_the_driver(monkeypatch):
    _stub_gpu(monkeypatch, {0: {"power": 350.0, "vram": 24576.0, "temp": 93.0}})
    assert limits.scale_for("gpu_power", 0) == {"min": 0, "max": 350.0}
    assert limits.scale_for("gpu_vram", 0) == {"min": 0, "max": 24576.0}
    assert limits.scale_for("gpu_temp", 0) == {"min": limits.AMBIENT_FLOOR_C, "max": 93.0}


def test_a_different_card_gets_different_ceilings(monkeypatch):
    """Nothing is pinned to one machine's 3090."""
    _stub_gpu(monkeypatch, {0: {"power": 450.0, "vram": 49152.0, "temp": 88.0}})
    assert limits.scale_for("gpu_power", 0)["max"] == 450.0
    assert limits.scale_for("gpu_vram", 0)["max"] == 49152.0
    assert limits.scale_for("gpu_temp", 0)["max"] == 88.0


def test_mixed_gpus_are_scaled_independently(monkeypatch):
    _stub_gpu(monkeypatch, {0: {"power": 350.0}, 1: {"power": 170.0}})
    assert limits.scale_for("gpu_power", 0)["max"] == 350.0
    assert limits.scale_for("gpu_power", 1)["max"] == 170.0


def test_absent_gpu_data_falls_back_to_auto(monkeypatch):
    """No driver, no invented ceiling."""
    _stub_gpu(monkeypatch, {})
    assert limits.scale_for("gpu_power", 0) is None
    assert limits.scale_for("gpu_temp", 0) is None


def test_disk_is_inherently_a_percentage():
    assert limits.scale_for("disk_used_pct", 0) == {"min": 0, "max": 100}


def test_throughput_keeps_a_floating_ceiling():
    """Throughput depends on the loaded model, and one bad sample once read
    37044 t/s — neither hardware nor history can set a ceiling here."""
    assert limits.scale_for("gpu_tokens_per_sec", 0) == {"min": 0, "max": None}


@pytest.mark.parametrize(
    "metric,peak,expected",
    [
        ("cpu_temp", 91.4, 95),      # rounded up to the next 5
        ("cpu_temp", 60.0, 60),      # floor 20 + min span 40
        ("cpu_power", 84.8, 90),     # next 10
        ("cpu_power", 12.0, 30),     # min span keeps an idle box legible
        ("fan2_rpm", 1939.0, 2000),  # next 250
        ("fan2_rpm", 0.0, 1000),     # a dead fan still gets a usable axis
    ],
)
def test_derived_ceilings_round_up_from_observed(metric, peak, expected):
    assert limits.scale_for(metric, 0, peak)["max"] == expected


def test_derived_ceiling_needs_a_peak():
    assert limits.scale_for("cpu_temp", 0, None) is None


def test_temperatures_start_at_room_temperature(monkeypatch):
    """A zero-based axis would waste half of every temperature card."""
    _stub_gpu(monkeypatch, {0: {"temp": 93.0}})
    assert limits.scale_for("gpu_temp", 0)["min"] == 20
    assert limits.scale_for("cpu_temp", 0, 91.4)["min"] == 20


def test_power_and_memory_start_at_zero(monkeypatch):
    _stub_gpu(monkeypatch, {0: {"power": 350.0, "vram": 24576.0}})
    assert limits.scale_for("gpu_power", 0)["min"] == 0
    assert limits.scale_for("gpu_vram", 0)["min"] == 0
    assert limits.scale_for("cpu_power", 0, 84.8)["min"] == 0


def test_fans_share_one_ceiling():
    """Two fans on different axes invite exactly the wrong comparison."""
    assert set(limits.peer_metrics("fan1_rpm")) == {"fan1_rpm", "fan2_rpm"}
    assert set(limits.peer_metrics("fan2_rpm")) == {"fan1_rpm", "fan2_rpm"}


def test_unrelated_metrics_are_not_grouped():
    assert limits.peer_metrics("cpu_temp") == ("cpu_temp",)
    assert limits.peer_metrics("gpu_power") == ()


def test_only_derived_metrics_ask_for_history():
    assert limits.needs_observed_max("cpu_temp")
    assert limits.needs_observed_max("fan1_rpm")
    assert not limits.needs_observed_max("gpu_power")
    assert not limits.needs_observed_max("disk_used_pct")


def test_system_ram_uses_installed_memory(monkeypatch):
    monkeypatch.setattr(limits, "_host_memory_total_mb", lambda: 31995.0)
    assert limits.scale_for("system_ram_mb", 0) == {"min": 0, "max": 31995.0}


def test_container_ram_uses_the_cgroup_limit(monkeypatch):
    monkeypatch.setattr(limits, "_cgroup_memory_max_mb", lambda: 28672.0)
    assert limits.scale_for("container_ram_mb", 0) == {"min": 0, "max": 28672.0}


def test_unlimited_cgroup_falls_back_to_host_ram(monkeypatch, tmp_path):
    """An unconstrained container is bounded by the machine, not by a sentinel."""
    monkeypatch.setattr(limits, "_host_memory_total_mb", lambda: 31995.0)
    monkeypatch.setattr("builtins.open", _fake_open({"/sys/fs/cgroup/memory.max": "max"}))
    assert limits._cgroup_memory_max_mb() == 31995.0


def _fake_open(contents):
    import io
    real = open

    def _open(path, *a, **kw):
        if path in contents:
            return io.StringIO(contents[path])
        return real(path, *a, **kw)

    return _open


def test_results_are_cached(monkeypatch):
    calls = []

    def probe():
        calls.append(1)
        return {0: {"power": 350.0}}

    monkeypatch.setattr(limits, "_gpu_hardware_limits", probe)
    limits.scale_for("gpu_power", 0)
    limits.scale_for("gpu_power", 0)
    limits.scale_for("gpu_vram", 0)
    assert len(calls) == 1, "nvidia-smi must not be shelled out per request"
