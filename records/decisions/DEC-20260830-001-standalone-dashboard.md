# DEC-20260830-001: Move The Dashboard To Its Own Site And Aggregate Its Telemetry

Opened: 2026-08-30 17-05-00 KST
Recorded by agent: claude-code

## Metadata

- Status: accepted
- Deciders: operator
- Related ids: DEC-20260525-001

## Decision

Two changes, taken together.

1. The dashboard leaves the chat webui and becomes its own deployed static site
   at `dash.lost.plus`, built from `dash/` and served by the `dash` compose
   service. It reads the gateway's existing `/stats` endpoints cross-origin, with
   a bearer token it holds itself. The gateway gains a `GRIMOIRE_CORS_ORIGINS`
   allowlist so the browser will permit that.

2. Telemetry gains a per-minute rollup table. Windows an hour or wider read the
   aggregate instead of raw 5-second samples, the payload is assembled on a
   worker thread rather than the event loop, and results are cached for a
   fraction of one bin.

## Context

The dashboard was a page inside `webui/`, which is a fork of upstream
llama.cpp's SvelteKit UI. Every dashboard change was therefore a change to a
fork that has to be merged forward whenever upstream moves, for a page that
shares no components with chat — its only tie to the rest of the webui was the
`apiFetch` helper.

While extracting it, a second and more serious problem surfaced. A worker agent
reported that an open dashboard tab was blocking model requests. Measured on the
live database:

- `state/telemetry.sqlite3` held 18,297,747 rows across 97 days — 1.3 GB.
- One "all" window render issued 15 binned queries. Each scanned the full history
  for its metric. Total: **15.3 seconds** of synchronous SQLite work.
- `get_dashboard_stats` was `async def` but every store call inside it was
  blocking, so that work ran **on the manager's event loop** — the same loop that
  serves chat.
- The page polled every **1000 ms** regardless of window.
- The database was in `journal_mode=delete`, so long reads and the 5-second
  sampler blocked each other.

A single open tab on "All" therefore saturated the manager permanently, and
every model request queued behind it. Relocating the page would not have fixed
any of this: the telemetry comes from `nvidia-smi`, `/sys/class/hwmon` and the
powercap mount, so the backend has to stay inside the gateway container.

## Options Considered

### Serve dash.lost.plus from the same container as a second hostname

- Upside: no CORS, no second sign-in, no new deploy unit
- Upside: still removes the page from the webui fork, which was the original goal
- Downside: not independently deployable; dies whenever the gateway does

### Deploy the dashboard as its own service

- Upside: independently deployable and restartable
- Upside: a webui rebuild cannot take the dashboard down
- Downside: needs a CORS allowlist, its own sign-in, and its own ingress

### Leave the dashboard in the webui and only fix the query cost

- Upside: smallest diff
- Downside: leaves the fork carrying a page that does not belong to it

### For the query cost: index the raw table instead of aggregating

- Upside: no new table, no backfill
- Downside: a covering index on 18M rows is large and still scans the whole
  metric range; it does not change the shape of the problem

## Rationale

The operator chose the standalone service for independent deployability,
accepting the cross-origin cost. That cost is small and bounded: one allowlist
env var, and a sign-in screen that stores a bearer token. Credentials stay off
the CORS config, because the dashboard never uses the `gw_session` cookie.

On the query side, the dashboard draws exactly 60 points for any window. Beyond
an hour, 5-second resolution is invisible on screen but pays a full scan for the
privilege. A per-minute rollup storing `(sum, count)` rather than a pre-divided
average lets any bin width be recombined without weighting error.

Bin origins are snapped to a bucket boundary on the aggregate path. Without that,
a bucket straddling a bin edge is credited wholesale to one side, which at the 1h
window (bins exactly one bucket wide) moved plotted values by up to 17%. Snapped,
every standard window has a bin width that is a whole number of buckets, and the
aggregate reproduces the raw averages exactly — verified to 0.000000000 absolute
error across the 1h, 6h, 24h, 7d and 30d windows on the live database.

The window is split at a rollup watermark, so a partially backfilled database
still reads correctly: older than the watermark from the aggregate, newer from
raw rows. Retention is clamped to that watermark, so it can never discard a
sample that has not been folded in yet.

Measured after the change, same database, same 15 series:

| Window | Before | After |
| --- | --- | --- |
| 1h | — | 5 ms |
| 24h | — | 11 ms |
| 30d | — | 192 ms |
| all | 15,310 ms | 691 ms |

## Consequences

- The webui fork loses `src/routes/dashboard/` and its sidebar entry, shrinking
  the merge surface against upstream.
- There is no longer a one-click path from chat to the dashboard. The sidebar
  navigates with SvelteKit's `goto`, which cannot leave the site, so a
  cross-site link would need changes to both nav components. Deferred.
- The gateway accepts cross-origin reads from exactly the origins named in
  `GRIMOIRE_CORS_ORIGINS`. It must never be set to `*`: `/stats` serves private
  per-key usage and cost figures.
- Telemetry is now WAL, so dashboard reads and the sampler stop blocking each
  other.
- First startup after this change backfills the rollup. On the 1.3 GB database
  that took 21 seconds, chunked so the sampler keeps running throughout.
- `GRIMOIRE_TELEMETRY_RETENTION_DAYS` becomes safe to enable: rollups are never
  pruned, so turning it on costs 5-second resolution on old windows but keeps
  lifetime graphs intact. It remains 0 (keep everything) by default; enabling it
  is the operator's call and would reclaim most of the 1.3 GB.
- The lifetime ("All") window is 691 ms, not instant. A second hourly rollup tier
  would bring it into the tens of milliseconds. Not built — the payload cache
  means this is paid once per 30 seconds on a worker thread, never on the event
  loop.
