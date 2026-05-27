# RSH-20260528-004: Agentic Permission Rules Checkpoint
Opened: 2026-05-28 02-49-33 KST
Recorded by agent: gpt-5-codex

## Question

Does llama-ui need an opencode permission-system port, or are the existing MCP-style allow / deny / once / always / always-server semantics enough for the first prototype?

## Finding

The existing llama-ui permission model is enough for the prototype.

llama-ui already has the core interaction contract:

- pending permission request state per conversation
- allow once
- always allow one tool
- always allow a provider/server group
- deny a tool call
- persisted always-allow rules in local storage
- Svelte action-card UI for blocking user decisions

opencode adds richer policy machinery: named permission keys, pattern rules, config merging, session pending queues, project persistence, explicit deny rules, and correction feedback. Those are useful later, but a literal port would bring backend storage, Effect services, project/session coupling, and file/path-specific policy assumptions that do not belong in this llama-ui-first prototype yet.

## Implemented

This checkpoint refines the existing llama-ui model instead of replacing it:

- Added `AGENTIC_INTERNAL_TOOL_NAMES` so internal agent tools can bypass execution permission explicitly.
- Added pure permission helpers:
  - `isInternalAgenticTool()`
  - `resolveStoredToolPermission()`
  - `shouldRequestToolPermission()`
  - `formatToolPermissionDeniedResult()`
- Updated `AgenticStore.requestPermission()` so stored allow rules return `ToolPermissionDecision.ALWAYS`, not `ONCE`.
- Updated denied tool results to include the tool name in model-visible output.
- Added unit coverage for internal-tool bypasses, stored permission resolution, request gating, and denial formatting.

## Classification

Adapted port.

The ported idea is opencode's distinction between tools that require permission and tools that should be available as internal agent protocol. The runtime remains llama-ui-native: client-side pending state, local persisted approvals, existing action-card prompts, and no backend permission service.

## Rejected For Now

- Pattern-based allow/deny config for bash, file paths, or URL scopes.
- Project database-backed permission storage.
- Permission correction feedback text.
- Multi-pending permission queue auto-resolution after an always approval.

Those make sense when sandbox, filesystem tools, web tools, and persistent product database work begins. They are unnecessary before the first prototype proves the llama-ui tool loop plus `question` prompt.

## Follow-Up Route

Next subsystem: tool lifecycle polish.

The lifecycle checkpoint already exposes normalized states. The next useful work is tightening how pending/running/success/error cards render and how internal tools appear in the timeline.

## Verification

- `npm run test:unit -- --run tests/unit/tool-permissions.test.ts tests/unit/agentic-question.test.ts tests/unit/agentic-sections.test.ts`
- Result: 25 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
