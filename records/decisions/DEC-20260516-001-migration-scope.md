# Migration Scope

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Decision

Retirement target is A + B + C: DFlash decode parity, compact-full persistence parity, and preserved PFlash parity must all be green before Lucebox cutover. Decode-only cutover is not allowed.

## Context

The served path today depends on the Lucebox dflash daemon for decode, the lucebox snapshot infrastructure for compact-full persistence, and the llama-side PFlash path for prompt compression. Migrating only decode (A) creates a split deployment where persistence and compression still depend on Lucebox. A partial cutover increases operational risk without delivering the full retirement benefit.

## Options Considered

| Option | Outcome |
| --- | --- |
| A + B + C (chosen) | Single cutover, one rollback plan, full Lucebox retirement |
| A only | Delivers faster but leaves persistence/compression on legacy path indefinitely; no real retirement |
| A + B | Decode and persistence on canonical, but PFlash still on Lucebox; split deployment |

## Rationale

- The decision document (MIGRATION_EXECUTION_CHECKLIST.md line 55) explicitly required A+B+C from the start.
- Prompt layout changes for native decode (A) also affect the persisted effective IDs (B) and the compression boundary semantics (C). Validating them independently is misleading.
- One cutover is less risky than three staggered cutovers.

## Consequences

- Phase 2, 3, and 4 must complete before cutover can be considered.
- Lucebox stays live longer, but avoids a multi-phase production migration.
- Rollback is simpler: flip one config, not three.
