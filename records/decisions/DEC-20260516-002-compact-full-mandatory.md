# compact-full Is Mandatory

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Decision

`snapshot-mode=compact-full` is required for the DFlash served path. No other snapshot mode is accepted in the migration.

## Context

The DFlash served model `dflash-pflash-qwen3.6-27B` uses `compact-full` snapshot mode in production. This mode preserves all state needed for correct continuation (KV cache, recurrent/native state, target_feat equivalents) as a single atomic snapshot at each prompt boundary. Alternatives (e.g., lightweight snapshots that drop some state) would break session restore semantics for existing users.

## Options Considered

| Option | Consequence |
| --- | --- |
| `compact-full` (chosen) | Full parity with current served behavior; larger per-snapshot footprint |
| Lighter snapshot mode | Smaller snapshots, but breaks session continuity and prefix-cache semantics |

## Rationale

- Every existing session was created with `compact-full`. Changing the snapshot mode would invalidate all persisted snapshots.
- The registry validator (`registry.py` line 712) already enforces `compact-full` for DFlash models.
- MIGRATION_EXECUTION_CHECKLIST.md line 29: "It is not optional in this migration."

## Consequences

- Native save/load on the canonical TheTom path must support all recurrent/native state, not just KV.
- Snapshot footprint is larger, putting more pressure on VRAM during save/load.
- Restart-resilience verification must confirm that a `compact-full` snapshot can survive daemon restart.
