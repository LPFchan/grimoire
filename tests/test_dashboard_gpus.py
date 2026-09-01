"""GPU tiles track the hardware that is actually present.

The payload used to hard-code indexes 0 and 1 into the GPU list regardless of
what the host had. On a single-GPU machine that drew a full set of GPU1 tiles
showing whatever index 1 last reported — which, after a card was pulled, was a
frozen reading from the day it disappeared rather than anything current.
"""

import pytest

from grimoire.routes import dashboard as D


class _Manager:
    def __init__(self, gpu_count):
        self.gpu_count = gpu_count


@pytest.fixture
def payload(monkeypatch):
    """Build a payload against stub stores, varying only the GPU count."""

    class _Telemetry:
        def latest(self, metric, gpu_index):
            return 1.0

        def binned_avg(self, metric, gpu_index, ts_from, ts_to, bins):
            return [1.0] * bins

        def earliest_ts(self):
            return 1_700_000_000

    class _Usage:
        def earliest_event_ts(self, user_hash=None):
            return 1_700_000_000

        def summary(self, user_hash=None):
            return {"total": {}}

        def binned_window(self, user_hash, ts_from, ts_to, bins):
            keys = (
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "input_cost", "output_cost", "cache_read_input_cost",
            )
            out = {f"{k}_series": [0] * bins for k in keys}
            out.update({f"total_{k}": 0 for k in keys})
            return out

        def load_card_arrangement(self, user_hash):
            return None

    monkeypatch.setattr(D, "telemetry_store", _Telemetry())
    monkeypatch.setattr(D, "usage_store", _Usage())

    def build(gpu_count, window="1h"):
        monkeypatch.setattr(D, "_get_manager", lambda: _Manager(gpu_count))
        return D._build_dashboard_payload("user-hash", window)

    return build


@pytest.mark.parametrize("gpu_count", [0, 1, 2, 4])
def test_one_entry_per_detected_gpu(payload, gpu_count):
    assert [g["index"] for g in payload(gpu_count)["gpus"]] == list(range(gpu_count))


def test_single_gpu_host_reports_no_gpu1(payload):
    """The case that prompted this: a second card removed from the machine."""
    indexes = [g["index"] for g in payload(1)["gpus"]]
    assert indexes == [0]
    assert 1 not in indexes


def test_headless_host_reports_no_gpus(payload):
    assert payload(0)["gpus"] == []


def test_extra_gpus_are_not_truncated(payload):
    """Nothing caps the list at two any more either."""
    assert len(payload(4)["gpus"]) == 4


def test_each_gpu_carries_its_four_series(payload):
    for gpu in payload(2)["gpus"]:
        assert set(gpu) == {"index", "temp", "power", "vram", "tokens_per_sec"}
