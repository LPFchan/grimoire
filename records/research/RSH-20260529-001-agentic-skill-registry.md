# RSH-20260529-001: Agentic Skill Registry Slice
Opened: 2026-05-29 00-38-45 KST
Recorded by agent: Codex

## Topic

First implementation slice for opencode-style `SKILL.md` support in the downstream llama-ui product.

## Findings

- The webui already had partial skill plumbing:
  - `parseSkillMarkdown()` parses `SKILL.md` frontmatter into `AgenticSkillInfo`.
  - `formatAvailableSkills()` injects available skill names and descriptions into agent context.
  - `agenticStore` has `setSkills()` and `skills()` but no registry, persistence, UI, or model-callable skill tool.
- A browser-local skill registry is the smallest useful product slice:
  - persists skills in localStorage under the app storage namespace
  - syncs registered skills into `agenticStore`
  - registers an internal `skill` tool provider
  - exposes the same data through a new `/skills` management page
- The `skill` tool is intentionally one operation-based tool for now:
  - `list`
  - `load`
  - `create`
  - `edit`
  - `delete`
- `load` returns the full generated `SKILL.md` text in the tool result so the model can ingest instructions only when needed, rather than always injecting every skill body into system context.

## Opencode Source Provenance

This slice is an adapted port, not a literal code copy.

- Referenced opencode's skill data shape from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:36-41`, where `Info` is `{ name, description?, location, content }`. The webui registry keeps those fields and adds `updatedAt` for local UI persistence.
- Referenced opencode's frontmatter validation expectations from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:52-58` and loading path from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:104-140`. The webui already had its own lightweight parser, so this slice reused and extended local `parseSkillMarkdown()` instead of copying opencode's `ConfigMarkdown`/Effect implementation.
- Referenced opencode's discovery locations and glob strategy from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:21-25` and `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:173-227`. The first webui slice intentionally does not copy filesystem or URL discovery; it starts with browser-local registration because llama-ui is currently browser-first and has no project filesystem contract.
- Referenced opencode's remote skill index support from `/home/yeowool/opencode/packages/opencode/src/skill/discovery.ts:12-19` and download flow from `/home/yeowool/opencode/packages/opencode/src/skill/discovery.ts:54-104`. This was explicitly deferred; follow-up import/export can adapt it once product persistence and network policy are decided.
- Referenced opencode's skill service API from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:96-102` and service methods from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:285-313`. The webui store implements equivalent local concerns (`get`, list via `skills`, create/update/delete) but keeps the API Svelte-store-shaped.
- Referenced opencode's available-skills prompt injection from `/home/yeowool/opencode/packages/opencode/src/session/system.ts:65-76` and formatter from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:326-350`. The webui already had `formatAvailableSkills()`; this slice changed its default copy to tell agents to use the `skill` tool to load matching skills.
- Referenced opencode's model-callable skill tool from `/home/yeowool/opencode/packages/opencode/src/tool/skill.ts:10-23` and its load result shape from `/home/yeowool/opencode/packages/opencode/src/tool/skill.ts:47-68`. The webui `skill` tool is deliberately broader (`list`, `load`, `create`, `edit`, `delete`) because the operator requested registry management through the agent, while opencode's tool only loads by name.
- Referenced opencode's permission gate for skill loading from `/home/yeowool/opencode/packages/opencode/src/tool/skill.ts:29-34` and permission-filtered availability from `/home/yeowool/opencode/packages/opencode/src/skill/index.ts:306-310`. The current webui slice does not yet require permission for skill mutation; that is tracked as follow-up below.
- Referenced opencode config support for additional skill paths/URLs from `/home/yeowool/opencode/packages/opencode/src/config/skills.ts:3-10`. This slice does not add config fields; localStorage is used as a first product registry.

## Decisions In This Slice

- Keep the registry browser-local for the first slice. This avoids product database and multi-user concerns while making the agent-facing behavior real.
- Store canonical skill records as `{ name, description, content, location, updatedAt }`, and generate `SKILL.md` text on demand.
- Add a sidebar-level Skills page rather than hiding skills under Settings, because the operator asked for skills as a new sidebar entry and because agents can mutate the registry.

## Open Follow-Ups

- Add import/export for whole skill folders and raw `SKILL.md` files.
- Decide whether skill mutations by agents should require explicit permission prompts before writes/deletes.
- Add server-backed persistence if skills need to travel across browsers or users.
- Add rendered markdown preview and validation warnings beyond frontmatter parsing.
- Continue next requested slices: richer compaction/summaries, tool-output compaction thresholds, tool event normalization, and basic task/subagent support.

## Verification

- `npm run build` passed for the webui after adding the registry, tool, and `/skills` route.
- `npm run check` still reports existing unrelated dashboard/models/chat-store diagnostics; no new skill-slice diagnostics were introduced.
