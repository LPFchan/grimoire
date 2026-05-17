# Grimoire Inbox

Ephemeral scratch disk for untriaged capture. Not a backlog, roadmap, or project digest.

## Rules

- Append freely from external capture, operator notes, or agent capture.
- Group related source events into one `IBX-*` entry when possible.
- Triage meaningful clusters, not raw source events or full external histories.
- Route, research, plan, discard, or leave; do not produce a project digest.
- Report counts or clusters of held/noisy/discarded items instead of summarizing every item.
- Preserve `IBX-*` as permanent provenance even after the entry is deleted.
- Remove entries once reflected into durable repo artifacts.

## Active Capture

### IBX-20260516-001

- Opened: `2026-05-16 14-00-00 KST`
- Recorded by agent: opencode
- Source: Phase 0 baseline gaps
- Summary: No trustworthy cold moderate/long-prompt restore baseline — served DFlash session turns hit GPU memory failures on 3090. No final stable five-run median set — intermittent daemon/OOM failures even on short runs.
- Confidence: `high`
- Triage status: `new`
- Triage decision: `route`
- Suggested destination: `STATUS.md` / `research/`
- Notes: Hardware-gated until memory wall is addressed or native TheTom path bypasses it.

### IBX-20260516-002

- Opened: `2026-05-16 14-00-00 KST`
- Recorded by agent: opencode
- Source: Phase 2 blocker
- Summary: Native canary `dflash-native-qwen3.6-27B-canary` has never been launched on GPU. The TheTom patched server may not accept `qwen35` arch as a speculative draft. Build pipeline needs to be run.
- Confidence: `high`
- Triage status: `new`
- Triage decision: `route`
- Suggested destination: `STATUS.md` / near-term plan
- Notes: This is the primary path forward. If TheTom accepts `qwen35`, the memory wall investigation may become moot.

### IBX-20260516-003

- Opened: `2026-05-16 14-00-00 KST`
- Recorded by agent: opencode
- Source: Fresh-state failure envelope
- Summary: After Q8_0 rollback fix, fresh-state turn-1 fails above ~319 tokens. Next band at ~385 tokens. The failure is in rollback/verify cache allocation.
- Confidence: `high`
- Triage status: `new`
- Triage decision: `research`
- Suggested destination: `research/` (already seeded as RSH-20260516-002)
- Notes: The native TheTom path uses a different code path and may not have this issue.

### IBX-20260516-004

- Opened: `2026-05-16 14-00-00 KST`
- Recorded by agent: opencode
- Source: PFlash baseline gaps
- Summary: PFlash compression baseline recorded (1.87x on 13-message prompt). No repeated-call compressor VRAM drift measurement yet. No cold-long-prompt run on the native stack yet.
- Confidence: `medium`
- Triage status: `new`
- Triage decision: `leave`
- Suggested destination: `research/` (already seeded as RSH-20260516-003)
- Notes: Wait for Track A hardware sign-off before addressing.

### IBX-20260516-005

- Opened: `2026-05-16 14-00-00 KST`
- Recorded by agent: opencode
- Source: TheTom native binary build
- Summary: We need to apply `patches/spec-dflash-contract.patch` to `tmp/spec-analysis/thetom-shallow/` and build `llama-server` with CUDA. The build environment and steps need to be documented.
- Confidence: `high`
- Triage status: `new`
- Triage decision: `plan`
- Suggested destination: `PLANS.md`
- Notes: This is the immediate next step. The build process should be captured as a decision or README.

## Daily Pressure Review Scratch

Use during review, clear after routing.

- Review date:
- Reviewer:
- Inbox pressure summary:
- Clusters reviewed:
- Promotion candidates:
- Research candidates:
- Plan candidates:
- Discard or purge candidates:
- Held without full summary:
- Operator route questions:

## Purge Rule

Once an item has been reflected into `SPEC.md`, `STATUS.md`, `PLANS.md`, `research/`, `decisions/`, a committed `LOG-*`, `upstream-intake/`, or a deliberate discard/hold note, remove the entry.
