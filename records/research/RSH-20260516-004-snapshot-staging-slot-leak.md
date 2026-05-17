# Snapshot Staging-Slot Leak

Opened: 2026-05-16 00-00-00 KST
Recorded by agent: opencode

## Finding

One-off served DFlash requests were taking a transient staging snapshot without persisting or freeing it. This stranded daemon slot state and caused a second short request to OOM on GPU 1.

## Fix

`src/grimoire/proxy/dflash.py` — ensure transient staging snapshots are freed after every non-session request, regardless of success or failure path.

## Result

After the fix, two sequential short non-session DFlash requests succeeded on the isolated container where the second previously OOM'd.

## Implication

The remaining hardware blocker is rollback/restore memory pressure on longer session turns, not transient staging-slot leaks. This narrowed the investigation significantly — all OOM failures above ~319 tokens are now attributable to the rollback-cache allocation ceiling, not a software leak.
