# RSH-20260529-005: Agentic Basic Task Tool
Opened: 2026-05-29 01-07-05 KST
Recorded by agent: Codex

## Topic

Add a first basic task/subagent capability without sandbox, shell, filesystem, or MCP access.

## Change

- Added an internal `task` tool with `description`, `prompt`, and optional `subagent_type`.
- Registered `task` as an internal provider tool and permission bypass.
- Executes the delegated task as a separate non-streaming model call with no tools.
- Returns an opencode-style `<task>` result block to the primary agent.
- Extended provider execution context with active model/API options so the task call can use the same runtime model.

## Opencode Source Provenance

This slice is an adapted minimal port; no opencode task runtime code was copied.

- Referenced opencode's task parameter shape from `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:34-52`.
- Referenced opencode's foreground/background result wrappers from `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:54-89`.
- Referenced opencode's permission and subagent lookup flow from `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:96-180`.
- Referenced opencode's subagent prompt execution flow from `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:183-201`.
- Referenced opencode's foreground task return shape from `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:267-279`.

## Design Notes

- The webui first slice intentionally rejects opencode's session tree, background job service, cancellation injection, subagent permission derivation, and tool inheritance.
- The delegated task call receives a system prompt that explicitly denies tools, shell commands, filesystem access, sandbox execution, and external MCP resources.
- The result is text-only for now. Rich subagent transcript UI can build on normalized tool events later.

## Verification

- `npm run test:unit -- --run tests/unit/tool-permissions.test.ts`
- `npm run build`
