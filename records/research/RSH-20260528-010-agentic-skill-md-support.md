# RSH-20260528-010: Agentic SKILL.md Support Checkpoint
Opened: 2026-05-28 03-11-10 KST
Recorded by agent: gpt-5-codex

## Question

What is the right first slice of opencode `SKILL.md` support for a llama-ui-based browser product?

## Implemented

This checkpoint adds browser-safe skill support:

- `AgenticSkillInfo`
  - `name`
  - optional `description`
  - `location`
  - `content`
- `parseSkillMarkdown(location, markdown)` parses simple `SKILL.md` frontmatter and body.
- `formatAvailableSkills(skills, opts)` formats compact markdown or verbose XML-like available-skill context.
- `AgenticStore` now has a client-side skill registry:
  - `agenticSetSkills(skills)`
  - `agenticSkills()`
- Agentic requests inject an ephemeral system message containing available skills when skills are registered.
- Unit coverage verifies parsing, compact formatting, verbose formatting, and missing-name rejection.

## Code-Level Shape

This intentionally does not port opencode's full skill subsystem:

- no filesystem crawl
- no global `~/.claude` / `.agents` discovery
- no URL pulling
- no backend service layer
- no skill permission rules
- no executable skill tool

Those pieces require a server-side product boundary. The useful first slice is the SKILL.md data shape plus prompt-context formatting, which can be driven by a later file picker, backend loader, or product database.

## Classification

Adapted port.

The parser and prompt formatting mirror opencode's useful model-facing behavior, but discovery and persistence are deferred. This keeps the work compatible with llama-ui's browser-first architecture and avoids coupling upstreamable UI code to local filesystem policy.

## Follow-Up Route

This completes the requested subsystem sequence through `SKILL.md` support.

Likely next work:

- add a product-side skill loader source
- add UI for registered skills
- decide whether skill content should be injected automatically, selected by the user, or requested by a future `skill` tool

## Verification

- `npm run test:unit -- --run tests/unit/agentic-skills.test.ts tests/unit/agentic-tool-registry.test.ts tests/unit/agentic-todo.test.ts tests/unit/agentic-question.test.ts tests/unit/tool-permissions.test.ts tests/unit/agentic-sections.test.ts`
- Result: 37 tests passed.

- `npm run check`
- Result: failed on pre-existing diagnostics outside this subsystem:
  - `src/lib/stores/chat.svelte.ts`
  - `src/routes/dashboard/+page.svelte`
  - `src/routes/models/+page.svelte`
