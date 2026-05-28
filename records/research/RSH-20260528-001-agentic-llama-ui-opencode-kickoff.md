# RSH-20260528-001: Agentic llama-ui / opencode Research Kickoff
Opened: 2026-05-28 02-16-50 KST
Recorded by agent: gpt-5-codex

## Question

What should be ported from opencode into a downstream product built on llama-ui, and where should llama-ui stay upstream-shaped instead of absorbing opencode's full architecture?

The requested posture was: default to porting opencode code or behavior directly unless there is a concrete reason not to. Classify each opencode subsystem as literal port, adapted port, wrapper, or reject/reimplement.

## Sources Inspected

Local llama-ui / llama.cpp:

- `webui/src/lib/utils/agentic.ts`
- `webui/src/lib/stores/agentic.svelte.ts`
- `webui/src/lib/services/chat.service.ts`
- `webui/src/lib/stores/tools.svelte.ts`
- `webui/src/lib/stores/permissions.svelte.ts`
- `webui/src/lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte`
- `webui/src/lib/types/database.d.ts`
- `webui/src/lib/stores/server.svelte.ts`
- `webui/src/lib/stores/models.svelte.ts`
- `webui/src/lib/hooks/use-processing-state.svelte.ts`
- `src/grimoire/pflash/deps/llama.cpp/tools/server/server-tools.cpp`
- `src/grimoire/pflash/deps/llama.cpp/tools/server/README.md`
- `src/grimoire/pflash/deps/llama.cpp/tools/server/README-dev.md`

Local opencode checkout:

- Repository cloned to `/home/yeowool/opencode`
- SHA inspected: `62a7781e04f03bcc4943093b3919067a587ddf8c`
- `packages/opencode/src/tool/*`
- `packages/opencode/src/tool/registry.ts`
- `packages/opencode/src/session/tools.ts`
- `packages/opencode/src/session/processor.ts`
- `packages/opencode/src/session/message-v2.ts`
- `packages/opencode/src/session/compaction.ts`
- `packages/opencode/src/session/status.ts`
- `packages/opencode/src/session/todo.ts`
- `packages/opencode/src/permission/index.ts`
- `packages/core/src/permission.ts`
- `packages/opencode/src/question/index.ts`
- `packages/opencode/src/tool/question.ts`
- `packages/opencode/src/skill/index.ts`
- `packages/opencode/src/mcp/index.ts`
- `packages/ui/src/components/message-part.tsx`
- `packages/app/src/pages/session/composer/session-permission-dock.tsx`
- `packages/app/src/pages/session/composer/session-question-dock.tsx`

## Findings

llama-ui already has enough of the first-prototype substrate to avoid a heavy opencode runtime port:

- OpenAI-compatible tool-call request and response handling exists in `ChatService.sendMessage`.
- Streaming tool call delta aggregation exists in `ChatService.handleStreamResponse`.
- Multi-turn agentic loop orchestration exists in `agenticStore`.
- MCP and builtin/custom tool definitions are merged through `toolsStore`.
- Permission prompts already support the important user-facing choices: allow, deny, once, always-style persistence.
- Tool cards, reasoning blocks, pending tool calls, permission prompts, continue prompts, and tool result rendering already exist in `ChatMessageAgenticContent.svelte`.
- Context and budget display already uses llama-server props, prompt progress, and timings.

This means the product should not begin by replacing llama-ui's agent loop with opencode's Effect/SQLite/bus runtime. The near-term useful opencode work is to port missing semantics, polish, schemas, and edge-case behavior into the existing llama-ui loop.

The biggest missing piece for the stated prototype is the `question` tool. opencode's schema and model-facing result are small enough to port literally, while the pending-request service and UI should be adapted to Svelte and llama-ui's current agentic store.

## llama-ui Extension Points

| Surface | Existing capability | Extension route |
| --- | --- | --- |
| `ChatService.sendMessage` | Sends OpenAI-compatible chat requests with `tools`, `tool_calls`, `tool_call_id`, and `reasoning_content`. | Keep as the model/runtime boundary. Add only protocol fields needed by product agent events. |
| `agenticStore` | Runs LLM -> tool calls -> tool execution -> tool results -> next LLM turn. | Main attach point for opencode-style lifecycle normalization. |
| `toolsStore` | Combines builtin, MCP, and custom tools. | Add an external tool-provider hook before relying on llama-server `/tools` as product API. |
| `permissionsStore` / permission prompt UI | Already supports allow/deny/once/always-equivalent behavior. | Tighten semantics to opencode's pattern/ruleset model only where current keys are too coarse. |
| `ChatMessageAgenticContent.svelte` | Renders reasoning, tool call cards, pending cards, results, permission prompt, continue prompt. | Port opencode card taxonomy and specialized displays incrementally. |
| `DatabaseMessage.extra` | Stores attachments from user and tool messages. | Reuse for tool artifacts and question metadata before adding a new persistence model. |
| Server/model stores | Expose context size, modalities, prompt progress, timings. | Improve display and compaction warnings; no opencode dependency required. |

## opencode Port Classification

| Subsystem | Classification | Code-level reason |
| --- | --- | --- |
| Agent event normalization | Adapted port | opencode's `SessionEvent` has clean event names for step/text/reasoning/tool/compaction, but llama-ui stores OpenAI-style messages, not event-sourced parts. |
| Tool lifecycle | Adapted port | opencode's pending/running/completed/error handling, abort cleanup, metadata, and attachments in `session/processor.ts` are valuable behavior, but tied to AI SDK stream events, Effect, bus, and DB services. |
| Permission rules | Adapted port, maybe narrow | Pure `allow | deny | ask` wildcard evaluation in `packages/core/src/permission.ts` is portable. The pending/deferred request service is backend-shaped and should be adapted only if llama-ui's current permission keys prove too coarse. |
| `question` tool | Literal protocol, adapted service/UI | `tool/question.ts` has the right compact schema and model-facing result. `question/index.ts` and Solid dock UI need Svelte/store adaptation. |
| Tool definitions/registry | Adapted port | `Tool.define` and registry behavior are useful patterns, but direct code brings Effect layers, plugin triggers, provider transforms, project/worktree context, and dynamic imports. |
| Frontend tool rendering | Adapted port | opencode's `message-part.tsx` has good card taxonomy, hidden tool rules, context grouping, todo/question special cases, and file/artifact presentation. It is Solid, so port behavior/design to Svelte. |
| Todo/status tool | Adapted port | `todowrite` and session todo service are simple, but llama-ui should persist per conversation and render in its own side panel/card style. Not needed for first prototype. |
| Skills | Adapted port later | Discovery/loading of `SKILL.md` is useful, but assumes backend filesystem access, config directories, cache, remote indexes, permissions, and project/worktree context. |
| Webfetch | Wrapper later | Requires backend fetch, size limits, HTML-to-Markdown, CORS bypass, permission, and attachment handling. Do not start here. |
| Websearch | Product-only wrapper later | Current implementation is Exa/Parallel vendor integration with env flags. Not upstream-shaped. |
| MCP | Wrapper/adapt later | llama-ui already has MCP client surfaces. opencode adds stdio/remote/OAuth/account handling, which is product-only and too broad for first prototype. |
| Session persistence / MessageV2 | Adapted/reimplement later | opencode MessageV2 parts are stronger than llama-ui's current shape, but a literal port implies a DB and renderer migration. Start with a sidecar/event-normalized view over existing messages. |
| Summarization/compaction | Reimplement later | opencode compaction depends on MessageV2, provider abstractions, token estimator, snapshots, plugins, and auto-continue. llama-ui should first use llama-server context telemetry. |
| Shell/edit/write/apply_patch | Reject for first prototype | Unsafe execution model. llama-server has experimental builtin tools, but upstream docs explicitly say `/tools` is internal and not for downstream apps. Product sandbox/backend required. |
| LSP, task/subagents, repo tools | Reject for first prototype | Strong project/git/LSP/session coupling; outside the requested first prototype. |

## Architecture Mismatches

opencode is a backend agent runtime. It assumes Effect services, SQLite persistence, a bus, filesystem/project/worktree context, provider adapters, plugin hooks, snapshots, and frontend clients over a server API.

llama-ui is a Svelte browser app using IndexedDB and direct llama-server calls. Its current agent loop is client-side and OpenAI-message-shaped.

Because of this mismatch, literal ports are appropriate only for small pure schemas or protocol fragments. Most high-value opencode pieces should be behavior ports into llama-ui's existing loop.

The llama-server builtin `/tools` endpoint is not a product boundary. Its own README states the endpoint is internal to the web UI and should not be used by downstream applications. It is useful as upstream reference material, not as the product's external tool execution API.

## Opencode Source Provenance

This research record referenced opencode source for subsystem classification; it did not copy implementation code into the repo.

- Event/lifecycle classification referenced `/home/yeowool/opencode/packages/core/src/session-event.ts:103-145`, `/home/yeowool/opencode/packages/core/src/session-event.ts:148-177`, `/home/yeowool/opencode/packages/core/src/session-event.ts:213-307`, and `/home/yeowool/opencode/packages/core/src/session-event.ts:332-363` for step, text, tool, and compaction event shapes.
- Tool lifecycle and attachment classification referenced `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:168-190`, `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:231-278`, `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:377-422`, and `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:452-500`.
- Permission classification referenced `/home/yeowool/opencode/packages/opencode/src/permission/index.ts:36-53`, `/home/yeowool/opencode/packages/opencode/src/permission/index.ts:171-211`, `/home/yeowool/opencode/packages/opencode/src/permission/index.ts:213-269`, and `/home/yeowool/opencode/packages/opencode/src/config/permission.ts:4-35`.
- Question-tool classification referenced `/home/yeowool/opencode/packages/opencode/src/tool/question.ts:6-41`, `/home/yeowool/opencode/packages/opencode/src/question/index.ts:16-93`, and `/home/yeowool/opencode/packages/opencode/src/question/index.ts:155-220`.
- Todo/status classification referenced `/home/yeowool/opencode/packages/opencode/src/tool/todo.ts:9-55`, `/home/yeowool/opencode/packages/opencode/src/session/todo.ts:10-31`, and `/home/yeowool/opencode/packages/opencode/src/session/todo.ts:41-75`.
- Tool registry, skill, and task/subagent classification referenced `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:73-80`, `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:117-134`, `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:225-269`, `/home/yeowool/opencode/packages/opencode/src/tool/registry.ts:282-360`, `/home/yeowool/opencode/packages/opencode/src/tool/skill.ts:10-68`, and `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:34-68`, `/home/yeowool/opencode/packages/opencode/src/tool/task.ts:96-180`.
- Compaction classification referenced `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:35-77`, `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:123-140`, `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:245-294`, `/home/yeowool/opencode/packages/opencode/src/session/compaction.ts:344-614`, and `/home/yeowool/opencode/packages/opencode/src/session/overflow.ts:6-31`.

## Upstreamable PR Candidates

- Agent event/section normalization for llama-ui.
- Better generic tool-call cards: pending/running/success/error, args/result, attachments.
- Permission prompt protocol and UI refinements.
- `question` tool UI/protocol.
- Artifact/file cards for tool outputs.
- Context budget display improvements using existing `/props`, `prompt_progress`, and timings.
- External tool-provider hook so downstreams can register tools without depending on internal `/tools`.

## Product-Only Features

- Backend agent runner and durable product database.
- Filesystem/shell/edit sandbox orchestration.
- Web search vendor integration.
- MCP OAuth/account/auth handling.
- Login/OAuth/multi-user.
- Persistent memory editor.
- Chat search.
- Product-specific model/session policy beyond upstream llama-ui shape.

## Recommended First Prototype

Use llama-ui's current agentic loop instead of replacing it.

Minimum slice:

1. Add an opencode-shaped internal lifecycle/event model over the existing messages.
2. Normalize existing llama-ui states into `step`, `reasoning`, `text`, `tool_input`, `tool_running`, `tool_success`, `tool_error`, and `blocked_for_permission`.
3. Port the opencode `question` tool schema and model-facing response.
4. Add a Svelte `question` pending-request UI using existing permission/continue prompt patterns.
5. Improve tool cards by porting opencode card taxonomy and special cases incrementally.

Do not start with sandbox, web search, MCP auth, memory, LSP, task/subagents, or full MessageV2 persistence.

## Answer To "Is Existing llama-ui Enough?"

Mostly yes for the first prototype.

The existing allow/deny/once/always semantics plus MCP-agent lifecycle are enough to avoid a ground-up permission/lifecycle port. The work should be:

- formalize the states already present in llama-ui so downstream product code can depend on a stable internal contract
- close gaps where current permissions are tool-name-only or server-label-only but opencode uses pattern-aware rules
- add the `question` tool, because it is not just another permission prompt; it is model-visible structured user input that returns a tool result back into the agent loop
- make renderer behavior richer and more predictable by porting opencode's card taxonomy

In other words: llama-ui already has the skeleton. opencode should supply the missing bones and polish, not replace the body.

## Open Questions

- Should product code introduce a small backend agent service immediately, or stay browser-only until unsafe tools are in scope?
- What exact stable event shape should be upstreamable to llama-ui without committing upstream to opencode's MessageV2 model?
- Should permissions remain localStorage-backed for the prototype, or move to conversation-scoped persistence before `question` lands?
