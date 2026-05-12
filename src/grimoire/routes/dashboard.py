"""Dashboard and stats route handlers (/stats/*)."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from grimoire.auth import require_api, require_admin
from grimoire.config import DASHBOARD_WINDOWS_S, DASHBOARD_BINS
from grimoire.telemetry import telemetry_store
from grimoire.usage import usage_store

router = APIRouter()


def _get_manager():
    from grimoire.entrypoint import manager
    return manager


@router.get("/stats")
async def get_stats(request: Request):
    """Return per-key token and equivalent-cost usage totals."""
    _, user_hash = require_api(request)
    return usage_store.summary(user_hash=user_hash)


@router.get("/stats/global")
async def get_global_stats(request: Request):
    """Return global token and equivalent-cost usage totals."""
    require_admin(request)
    return usage_store.summary()


@router.get("/stats/dashboard")
async def get_dashboard_stats(request: Request):
    """Combined token/cost + system telemetry time series for the dashboard.

    Query params:
        window: one of "5m","15m","1h","6h","24h","7d","30d","all" (default "1h")
    """
    _, user_hash = require_api(request)
    window = (request.query_params.get("window") or "1h").lower()

    now_ts = datetime.now(timezone.utc).timestamp()
    if window in {"all", "lifetime"}:
        earliest = usage_store.earliest_event_ts(user_hash=user_hash)
        sample_earliest = telemetry_store.earliest_ts()
        candidates = [t for t in (earliest, sample_earliest) if t]
        ts_from = min(candidates) if candidates else now_ts - DASHBOARD_WINDOWS_S["1h"]
        if ts_from >= now_ts:
            ts_from = now_ts - DASHBOARD_WINDOWS_S["1h"]
        window_label = "all"
    else:
        seconds = DASHBOARD_WINDOWS_S.get(window)
        if seconds is None:
            raise HTTPException(status_code=400, detail=f"Unknown window: {window}")
        ts_from = now_ts - seconds
        window_label = window

    bins = DASHBOARD_BINS
    usage = usage_store.binned_window(user_hash, ts_from, now_ts, bins)
    summary = usage_store.summary(user_hash=user_hash)
    lifetime = summary.get("total", {})

    def _system(metric, gpu_index):
        return {
            "current": telemetry_store.latest(metric, gpu_index),
            "series": telemetry_store.binned_avg(metric, gpu_index, ts_from, now_ts, bins),
        }

    manager = _get_manager()
    gpu_indexes = sorted({0, 1, *range(manager.gpu_count)})
    gpus = [
        {
            "index": idx,
            "temp": _system("gpu_temp", idx),
            "power": _system("gpu_power", idx),
            "vram": _system("gpu_vram", idx),
            "tokens_per_sec": _system("gpu_tokens_per_sec", idx),
        }
        for idx in gpu_indexes
    ]

    def _cumulative(series):
        running = 0
        return [running := running + v for v in series]

    return {
        "window": window_label,
        "from": ts_from,
        "to": now_ts,
        "bins": bins,
        "tokens": {
            "input": {
                "current": usage["total_input_tokens"],
                "series": _cumulative(usage["input_tokens_series"]),
            },
            "output": {
                "current": usage["total_output_tokens"],
                "series": _cumulative(usage["output_tokens_series"]),
            },
        },
        "cost": {
            "total": usage["total_input_cost"] + usage["total_output_cost"],
            "input": usage["total_input_cost"],
            "output": usage["total_output_cost"],
            "lifetime": float(lifetime.get("total_cost") or 0.0),
            "series": _cumulative([
                a + b
                for a, b in zip(usage["input_cost_series"], usage["output_cost_series"])
            ]),
        },
        "gpus": gpus,
        "cpu": {
            "temp": _system("cpu_temp", 0),
            "power": _system("cpu_power", 0),
        },
        "fans": {
            "fan1": _system("fan1_rpm", 0),
            "fan2": _system("fan2_rpm", 0),
        },
        "ram": {
            "system": _system("system_ram_mb", 0),
            "container": _system("container_ram_mb", 0),
        },
    }
