"""Payload cache in front of /stats/dashboard.

The page polls once a second. Rebuilding a 24h payload that often is pure waste —
a point on that graph covers 24 minutes and cannot move meaningfully between
polls — and it used to be waste paid on the event loop, ahead of chat requests.
"""

import time

from grimoire.routes import dashboard as D


def test_ttl_tracks_bin_width():
    """A payload is held for a fraction of one bin, so graphs stay honest."""
    hour = D._cache_ttl(0, 3600, 60)          # 60s bins
    day = D._cache_ttl(0, 86400, 60)          # 24m bins
    assert hour < day


def test_ttl_is_clamped_both_ways():
    """Never staler than 30s, never so short that polling rebuilds every time."""
    assert D._cache_ttl(0, 60, 60) == D._CACHE_MIN_TTL_S        # 1s bins
    assert D._cache_ttl(0, 10**9, 60) == D._CACHE_MAX_TTL_S     # lifetime
    assert D._CACHE_MIN_TTL_S <= D._cache_ttl(0, 86400, 60) <= D._CACHE_MAX_TTL_S


def test_hit_then_expiry():
    key = ("user-hash", "24h")
    D._cache_put(key, {"marker": 1}, ttl=10)
    assert D._cache_get(key) == {"marker": 1}

    D._cache_put(key, {"marker": 2}, ttl=-1)
    assert D._cache_get(key) is None


def test_users_do_not_share_payloads():
    """Usage and cost figures are per key — one user's payload is not another's."""
    D._cache_put(("alice", "1h"), {"whose": "alice"}, ttl=10)
    assert D._cache_get(("bob", "1h")) is None
    assert D._cache_get(("alice", "1h")) == {"whose": "alice"}


def test_windows_do_not_share_payloads():
    D._cache_put(("alice", "1h"), {"w": "1h"}, ttl=10)
    D._cache_put(("alice", "24h"), {"w": "24h"}, ttl=10)
    assert D._cache_get(("alice", "1h")) == {"w": "1h"}
    assert D._cache_get(("alice", "24h")) == {"w": "24h"}


def test_expired_entries_do_not_accumulate():
    for i in range(80):
        D._cache_put((f"user{i}", "1h"), {"i": i}, ttl=-1)
    D._cache_put(("live", "1h"), {"i": "live"}, ttl=30)
    assert len(D._cache) < 80
    assert D._cache_get(("live", "1h")) == {"i": "live"}
