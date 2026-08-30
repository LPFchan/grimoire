# DEC-20260830-002: Add An Hourly Rollup Tier And Snap The Lifetime Window

Opened: 2026-08-30 18-40-00 KST
Recorded by agent: claude-code

## Metadata

- Status: accepted
- Deciders: operator
- Related ids: DEC-20260830-001

## Decision

Generalise `system_rollup` into tiers, add a 3600s tier built from the 60s tier,
and widen the lifetime window until its bins are a whole number of hours.

This closes the open item left in DEC-20260830-001, which shipped a single
per-minute tier and recorded the lifetime window at 691 ms as not-instant.

## Context

DEC-20260830-001 took the lifetime window from 15,310 ms to 691 ms with a single
per-minute rollup. 691 ms is fine behind a cache but is still the slowest thing
the dashboard does, and it was the one window the operator asked to be genuinely
instant.

The rollup table did not yet exist in the live database — the gateway had not
been restarted since DEC-20260830-001 landed — so the schema could be reshaped
with no migration.

## Options Considered

### A separate hourly table with its own watermark

- Upside: no change to the existing table
- Downside: two query shapes and two maintenance paths that must not drift

### A `bucket_s` column, tiers as rows in one table

- Upside: one schema, one query shape, one maintenance loop; more tiers are free
- Downside: a wider primary key
- Downside: would have needed a migration if the table already had data

### Leave it at one tier and accept 691 ms

- Upside: nothing to build
- Downside: leaves the operator's request unmet

## Rationale

Sum and count compose, so a coarse tier can be built by summing a finer one
rather than rescanning raw samples: the hourly tier costs almost nothing to
maintain once the per-minute tier exists. A derived tier is capped by its
source's watermark, so it can never aggregate a bucket the tier below has not
finished writing.

Tier selection turned out to be the subtle part. The first rule tried was
"prefer exact nesting, else coarsest", which never reached the hourly tier for
the lifetime window, because its bin width happened to divide evenly by 60. The
rule that works is "coarsest tier that is safe", where safe means either the
buckets divide the bin width evenly, or a bin holds at least 8 buckets.

That second clause was then measured and found wanting. On the live database an
unsnapped lifetime window read through the hourly tier drifted by up to 347 MB
on VRAM and 4.4 W on GPU power — roughly 2% of full scale, from buckets
straddling bin edges. Rather than accept that, the endpoint now rounds the
lifetime bin width up to a whole hour. Bins and buckets then nest exactly and
the tier is exact; the only cost is that the graph starts up to one bucket-width
before the first sample.

Note that 7d deliberately stays on the per-minute tier: its bins are 2.8 hours,
which 3600s does not divide, and it is already 45 ms.

Measured on the live 1.3 GB, 18.3M row database, full render of 15 series:

| Window | Tier | DEC-...-001 | Now |
| --- | --- | --- | --- |
| 1h | 60s | 5 ms | 4 ms |
| 24h | 60s | 11 ms | 11 ms |
| 7d | 60s | 42 ms | 45 ms |
| 30d | 3600s | 192 ms | 9 ms |
| all | 3600s | 691 ms | 18 ms |

Verified exact against raw rows on an identical grid: 0.000000000 absolute error
across 1h, 6h, 24h, 7d, 30d and the snapped lifetime window.

## Consequences

- Every dashboard window now renders in under 50 ms, against a database that
  previously needed 15.3 seconds for its worst case.
- Backfill on first restart went from 21 s to 44 s, since it now builds both
  tiers. Still chunked, still leaves the sampler running.
- The lifetime graph starts slightly before the first recorded sample. On the
  current database that is 9.5 hours ahead of a 97-day window — invisible.
- Adding further tiers is now a one-line change to `ROLLUP_TIERS`, though
  nothing currently needs one.
- Measuring a rollup against a differently-aligned baseline produces nonsense.
  Two separate readings during this work looked like large regressions and were
  purely an artefact of the comparison grid shifting. Any future check must pin
  the origin and range on both sides.
