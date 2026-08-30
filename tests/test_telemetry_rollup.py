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
    original = telemetry.ROLLUP_BUCKET_S
    telemetry.ROLLUP_BUCKET_S = 10**9
    try:
        return store.binned_avg(metric, gpu, ts_from, ts_to, bins)
    finally:
        telemetry.ROLLUP_BUCKET_S = original


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

    # Rewind the watermark and fold the same history again.
    with sqlite3.connect(s.path) as conn:
        conn.execute("UPDATE rollup_state SET rolled_through_ts = ?", (base,))
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
        watermark = conn.execute("SELECT rolled_through_ts FROM rollup_state").fetchone()[0]
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
        watermark = conn.execute("SELECT rolled_through_ts FROM rollup_state").fetchone()[0]
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
