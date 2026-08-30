"""Lifetime window snapping.

Every fixed window divides evenly by the rollup bucket, so its bins and buckets
nest and the aggregate reproduces the raw average exactly. The lifetime window
starts wherever the first sample landed, so it divides evenly by nothing — and
an unsnapped lifetime window measured about 2% of full scale off on the live
database, because buckets straddling a bin edge were credited to one side.
"""

import math

from grimoire.routes.dashboard import _snap_lifetime_start
from grimoire.telemetry import ROLLUP_TIERS

BINS = 60
COARSEST = ROLLUP_TIERS[-1]


def test_bins_become_a_whole_number_of_buckets():
    ts_to = 1_700_000_000.123
    ts_from = ts_to - 97 * 86400 - 517        # deliberately ragged
    snapped = _snap_lifetime_start(ts_from, ts_to, BINS)

    width = (ts_to - snapped) / BINS
    assert width % COARSEST == 0


def test_window_only_ever_widens():
    """Snapping must not hide data by starting after the first sample."""
    ts_to = 1_700_000_000.0
    for offset in (3601, 86400, 97 * 86400 + 17, 1234567):
        ts_from = ts_to - offset
        assert _snap_lifetime_start(ts_from, ts_to, BINS) <= ts_from


def test_widening_is_under_one_bucket_per_bin():
    """The graph starts early, but never by more than one bucket per bin."""
    ts_to = 1_700_000_000.0
    ts_from = ts_to - 97 * 86400
    snapped = _snap_lifetime_start(ts_from, ts_to, BINS)
    assert ts_from - snapped < COARSEST * BINS


def test_already_aligned_window_is_left_alone():
    ts_to = 1_700_000_000.0
    ts_from = ts_to - COARSEST * BINS * 12     # exactly 12 buckets per bin
    assert _snap_lifetime_start(ts_from, ts_to, BINS) == ts_from


def test_snapped_width_selects_the_coarsest_tier():
    """The whole point: the lifetime view becomes eligible for the coarse tier."""
    from grimoire.telemetry import TelemetryStore

    ts_to = 1_700_000_000.0
    ts_from = ts_to - 97 * 86400 - 517
    width = (ts_to - _snap_lifetime_start(ts_from, ts_to, BINS)) / BINS
    assert TelemetryStore._usable_tiers(width)[0] == COARSEST


def test_short_lifetime_still_snaps():
    """A database only a few hours old must not snap to a degenerate window."""
    ts_to = 1_700_000_000.0
    ts_from = ts_to - 900
    snapped = _snap_lifetime_start(ts_from, ts_to, BINS)
    width = (ts_to - snapped) / BINS
    assert width == COARSEST
    assert snapped < ts_from
