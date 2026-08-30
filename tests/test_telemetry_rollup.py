"""Per-minute rollup behind the dashboard's long windows.

The dashboard only ever draws 60 points, so wide windows gain nothing from
5-second resolution but used to pay a full scan of every raw sample for it. These
tests pin the two properties that make the aggregate safe to read from: it
reproduces the raw averages exactly, and it never lets retention delete a sample
that has not been folded in yet.
"""

import importlib
import sqlite3

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIMOIRE_TELEMETRY_PATH", str(tmp_path / "telemetry.sqlite3"))
    telemetry = importlib.reload(importlib.import_module("grimoire.telemetry"))
    yield telemetry.TelemetryStore(str(tmp_path / "telemetry.sqlite3")), telemetry
    importlib.reload(importlib.import_module("grimoire.telemetry"))


def _fill(store, base_ts, count, interval=5, metric="cpu_temp", gpu=0):
    """Write `count` samples with a value that varies, so averages are distinctive."""
    for i in range(count):
        store.record(base_ts + i * interval, [(gpu, metric, float(20 + (i % 37)))])


def _raw_binned(store, metric, gpu, ts_from, ts_to, bins, telemetry):
    """Same query, forced down the raw path, for use as ground truth."""
    original = telemetry.ROLLUP_TIERS
    telemetry.ROLLUP_TIERS = ()
    try:
        return store.binned_avg(metric, gpu, ts_from, ts_to, bins)
    finally:
        telemetry.ROLLUP_TIERS = original


def test_rollup_matches_raw_exactly(store):
    """A rollup-backed window must equal the raw rows over the same grid."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 1440)                       # two hours at 5s
    now = base + 1440 * 5

    while s.roll_up(now_ts=now):
        pass

    bins = 60
    ts_from = base
    ts_to = base + bins * bucket               # bins exactly one bucket wide

    from_rollup = s.binned_avg("cpu_temp", 0, ts_from, ts_to, bins)
    from_raw = _raw_binned(s, "cpu_temp", 0, ts_from, ts_to, bins, telemetry)

    assert from_rollup == pytest.approx(from_raw, rel=1e-12)


def test_short_windows_still_read_raw(store):
    """Sub-bucket bins keep full 5-second detail."""
    s, telemetry = store
    base = 1_700_000_000 // telemetry.ROLLUP_BUCKET_S * telemetry.ROLLUP_BUCKET_S
    _fill(s, base, 60)
    now = base + 300
    while s.roll_up(now_ts=now):
        pass

    # 5 minutes over 60 bins = 5s bins, narrower than a bucket.
    series = s.binned_avg("cpu_temp", 0, base, base + 300, 60)
    assert series == _raw_binned(s, "cpu_temp", 0, base, base + 300, 60, telemetry)


def test_unrolled_tail_is_still_included(store):
    """Samples newer than the watermark come from raw, so nothing goes missing."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 720)                        # one hour
    now = base + 3600

    # Roll up only the first half of the history.
    s.roll_up(now_ts=base + 1800)

    bins = 60
    combined = s.binned_avg("cpu_temp", 0, base, base + bins * bucket, bins)
    raw_only = _raw_binned(s, "cpu_temp", 0, base, base + bins * bucket, bins, telemetry)
    assert combined == pytest.approx(raw_only, rel=1e-12)
    assert all(v is not None for v in combined)


def test_roll_up_is_idempotent(store):
    """Re-folding a chunk must replace, not double-count."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 720)
    now = base + 3600
    while s.roll_up(now_ts=now):
        pass

    before = s.binned_avg("cpu_temp", 0, base, base + 60 * bucket, 60)

    # Forget every watermark and fold the same history again from scratch.
    with sqlite3.connect(s.path) as conn:
        conn.execute("DELETE FROM rollup_state")
    while s.roll_up(now_ts=now):
        pass

    assert s.binned_avg("cpu_temp", 0, base, base + 60 * bucket, 60) == pytest.approx(before)


def test_incomplete_bucket_is_not_folded(store):
    """The bucket still receiving samples must stay out of the aggregate."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 24)                          # two minutes
    now = base + 90                             # mid-way through the second bucket

    while s.roll_up(now_ts=now):
        pass

    with sqlite3.connect(s.path) as conn:
        watermark = conn.execute(
            "SELECT rolled_through_ts FROM rollup_state WHERE bucket_s = ?", (bucket,)
        ).fetchone()[0]
    assert watermark <= now // bucket * bucket


def test_prune_never_outruns_the_rollup(store):
    """Retention must not drop a sample that has not been folded in yet."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 720)
    now = base + 3600

    s.roll_up(now_ts=base + 600)                # watermark deliberately far behind
    s.prune(now)                                # ask to delete everything

    with sqlite3.connect(s.path) as conn:
        watermark = conn.execute(
            "SELECT rolled_through_ts FROM rollup_state WHERE bucket_s = ?", (bucket,)
        ).fetchone()[0]
        oldest = conn.execute("SELECT MIN(ts) FROM system_samples").fetchone()[0]
    assert oldest is not None and oldest >= watermark - bucket


def test_earliest_ts_survives_pruned_raw(store):
    """Lifetime range comes from the rollup once raw samples are gone."""
    s, telemetry = store
    bucket = telemetry.ROLLUP_BUCKET_S
    base = 1_700_000_000 // bucket * bucket
    _fill(s, base, 720)
    now = base + 3600
    while s.roll_up(now_ts=now):
        pass
    s.prune(now)

    with sqlite3.connect(s.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM system_samples").fetchone()[0] < 720
    assert s.earliest_ts() is not None
    assert s.earliest_ts() <= base + bucket


def test_wal_is_enabled(store):
    """Readers must not block the sampler."""
    s, _ = store
    with sqlite3.connect(s.path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


# --- tiering ---------------------------------------------------------------


def test_hourly_tier_is_built_from_the_minute_tier(store):
    """The coarse tier derives from the fine one, not from raw samples."""
    s, telemetry = store
    minute, hour = telemetry.ROLLUP_TIERS
    base = 1_700_000_000 // hour * hour
    _fill(s, base, 2160)                        # three hours at 5s
    now = base + 3 * hour

    while s.roll_up(now_ts=now):
        pass

    with sqlite3.connect(s.path) as conn:
        conn.row_factory = sqlite3.Row
        per_tier = {
            r["bucket_s"]: r["n"]
            for r in conn.execute(
                "SELECT bucket_s, COUNT(*) AS n FROM system_rollup GROUP BY bucket_s"
            )
        }
        # An hourly bucket must hold exactly what its minute buckets hold.
        hourly = conn.execute(
            "SELECT sum_value, sample_count FROM system_rollup WHERE bucket_s=? AND bucket_ts=?",
            (hour, base),
        ).fetchone()
        minutes = conn.execute(
            """
            SELECT SUM(sum_value) AS s, SUM(sample_count) AS c FROM system_rollup
            WHERE bucket_s=? AND bucket_ts >= ? AND bucket_ts < ?
            """,
            (minute, base, base + hour),
        ).fetchone()

    assert per_tier[minute] > per_tier[hour]
    assert hourly["sum_value"] == pytest.approx(minutes["s"])
    assert hourly["sample_count"] == minutes["c"]


def test_coarse_tier_never_outruns_its_source(store):
    """An hourly bucket must not be built from a half-written minute tier."""
    s, telemetry = store
    minute, hour = telemetry.ROLLUP_TIERS
    base = 1_700_000_000 // hour * hour
    _fill(s, base, 4320)                        # six hours
    now = base + 6 * hour

    s.roll_up(now_ts=now)                       # a single pass, tiers still behind

    with sqlite3.connect(s.path) as conn:
        conn.row_factory = sqlite3.Row
        marks = {
            r["bucket_s"]: r["rolled_through_ts"]
            for r in conn.execute("SELECT bucket_s, rolled_through_ts FROM rollup_state")
        }
    if hour in marks:
        assert marks[hour] <= marks[minute]


def test_tier_choice_prefers_exact_nesting(store):
    """A tier is only picked when its buckets divide the bin width evenly."""
    _, telemetry = store
    minute, hour = telemetry.ROLLUP_TIERS
    bins = 60

    # 7d over 60 bins = 10080s bins. 10080 / 3600 = 2.8, so hourly does not nest
    # and must not be chosen despite being coarser.
    assert telemetry.TelemetryStore._usable_tiers(604800 / bins)[0] == minute
    # 30d = 43200s bins, a whole 12 hourly buckets.
    assert telemetry.TelemetryStore._usable_tiers(2592000 / bins)[0] == hour
    # 6h = 360s bins: too narrow for hourly at all.
    assert hour not in telemetry.TelemetryStore._usable_tiers(21600 / bins)


def test_arbitrary_width_falls_back_to_bucket_count(store):
    """The lifetime window has no exact tier, so it leans on the ratio rule."""
    _, telemetry = store
    minute, hour = telemetry.ROLLUP_TIERS

    # A ~97 day lifetime window: no tier divides it evenly, but a bin holds
    # dozens of hourly buckets, so the coarse tier is safe.
    width = 97 * 86400 / 60
    assert width % hour != 0
    assert telemetry.TelemetryStore._usable_tiers(width)[0] == hour

    # A bin only a few buckets wide, and not an exact multiple, must not use it.
    assert hour not in telemetry.TelemetryStore._usable_tiers(hour * 3 + 1)
    assert minute not in telemetry.TelemetryStore._usable_tiers(minute * 3 + 1)


def test_hourly_window_matches_raw(store):
    """A 30d-shaped window read through the hourly tier equals the raw rows."""
    s, telemetry = store
    hour = telemetry.ROLLUP_TIERS[1]
    base = 1_700_000_000 // hour * hour
    # 60 bins of 12 hourly buckets each, sampled sparsely to keep the test quick.
    _fill(s, base, 3000, interval=600)
    now = base + 3000 * 600

    while s.roll_up(now_ts=now):
        pass

    bins = 60
    width = 12 * hour
    ts_to = base + bins * width
    assert telemetry.TelemetryStore._usable_tiers(width)[0] == hour

    tiered = s.binned_avg("cpu_temp", 0, base, ts_to, bins)
    raw = _raw_binned(s, "cpu_temp", 0, base, ts_to, bins, telemetry)
    assert tiered == pytest.approx(raw, rel=1e-12)
