# RSH-20260621-001: `Failed to parse tool call arguments as JSON` — 500-on-replay diagnosis

Opened: 2026-06-21 05-36-23 KST
Recorded by agent: claude-code
Related ids: —

## Research Question

A webui chat (Qwen3.6-27B) returns `Server Error: Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 2: syntax error while parsing object key - unexpected end of input; expected string literal`, and every subsequent turn in that conversation 500s. What causes it, who is at fault, how do we unbreak it and prevent recurrence, and how upstreamable is the fix?

## Mechanism (verified in the serving binary)

"Column 2 / `{` then EOF" means the offending arguments string is exactly `{`.

The message is emitted by `func_args_not_string()`, which force-parses every incoming `tool_calls[].function.arguments` that `is_string()` via `json::parse()` and rethrows on failure. This is **mainline ggml-org/llama.cpp** (`namespace workaround`, Autoparser era) — *not* fork-specific. Verified directly: `TheTom/llama-cpp-turboquant@4595fff/common/chat.cpp` (the serving binary, Dockerfile:9,95) contains `func_args_not_string`, the `namespace workaround` block, and the exact error string. Local refs in the vendored `src/grimoire/pflash/deps/llama.cpp` copy: body `chat.cpp:2022-2040`, callsite `:2166-2168`.

It runs at template-apply time, gated on `caps.supports_object_arguments` (detected at `caps.cpp:247-251` by test-rendering the template against object-valued `arguments.arg` and checking sub-key access). The buun template's mapping branch (`tc.arguments.items()`, `templates/buun-Qwen3.6-chat_template.jinja:288`) flips the cap true, so the preprocessing **always runs** for Qwen-family models. It runs over the **history the client replays each turn**; one prior assistant turn carries `arguments == "{"`; `json::parse("{")` throws → 500 before generation.

The template's own string-arguments branch (`tc.arguments is string`, template:304-306) cannot save it: `func_args_not_string` runs *before* the template, so a parseable string is pre-converted to an object (branch dead) and an unparseable `{` throws first.

## Why the conversation is permanently bricked (server-only fix)

webui stores conversations in **browser IndexedDB (Dexie)** (`webui/src/lib/constants/database.ts:14-23`) and replays full history every turn, so the poison message lives in the user's browser and 500s every turn before generation. The grimoire SQLite store is recording-only: `_record_response_stream` (defined in `entrypoint.py`, imported at `proxy/llama.py:99`) is gated `record_history = upstream.status_code < 400` (`proxy/llama.py:319`) — failing replays are never recorded, and the poison turn was only stored on its *original successful* turn. Editing server SQLite cannot reach the replayed payload.

## Fault attribution

Two compounding **upstream llama.cpp** defects; the buun template is a *trigger*, not the originator; TheTom's fork is not implicated.

1. **Producing the malformed `{`** — the Autoparser reconstructs the model's Qwen3-Coder XML output (`<function=…><parameter=…>`) into JSON `arguments`, buggy in the autoparser era. Open cluster: ggml-org/llama.cpp #19382, #22072, #20359.
2. **500-on-replay brittleness** — `func_args_not_string` hard-fails the *entire* request on one unparseable historical arg instead of degrading. Issues #22948, #21680, discussion #19287.

The **buun template** is mostly exonerated: hardened Qwen3-Coder template (its "FIX6: arguments.items()" mirrors public fixed templates), uses the format Qwen3.6 was trained on, tolerates string args — it did not create `{`. Its only exposure: a heavily-customized template can defeat parser auto-detection, causing a generic-parser fallback that mis-reconstructs.

## Candidate origins of `{` (leaning B)

- **B. Parser reconstruction (leading).** Empty/no-`<parameter>` call or XML→JSON reconstruction bug yields `{`; supported by #19382 (the buun format); produces exactly `{`.
- **A. Truncation.** Mid-generation cutoff (`predict=16384`, early EOG, MTP/`nextn` early stop); persisted string could be `{` *or* content-bearing (`{"path":"/foo/ba`).
- **C. Client-side serialization** of an aborted call into `"{"`.

## Upstream fix landscape — none conclusive

- **#22948** is reported against **Qwen3.6-27B (our exact model)** with the exact error: **open, no maintainer response, no PR**; the reporter's proposed graceful-degradation is unimplemented.
- The only fixed neighbor — **#19382 → PR #19765** — is a *different model* (Qwen3-Coder-Next) and a generation-side parser swap that does not touch the input-replay path. Does not transfer.
- HF template updates are generation-side band-aids the buun template already carries.

## Fix (ours to own)

1. **Primary — gateway sanitization, content-aware.** In a `before_request` plugin (correct point: `_proxy_chat` calls `plugin_manager.before_request` at `proxy/llama.py:119`, *before* the pflash prefill-layout at `:132+` and the upstream send, so both consumers see the sanitized payload). Walk `messages[].tool_calls[].function.arguments`; if a string fails `json.loads`:
   - **empty-ish / exactly `{` / whitespace** → normalize to `{}` (safe; this conversation's case → auto-heals next turn).
   - **content-bearing but unparseable** (origin A) → do **not** silently zero it (that turns a visible 500 into a silent wrong empty-call). Best-effort brace/quote-balance and re-parse; else drop the tool_call or insert an explicit error marker so the model retries.

   *Precedent (separate path):* grimoire's pflash prompt builder already survives bad string args because `prompt/qwen.py:184` gates parameter rendering on `isinstance(arguments, dict)`, emitting a structurally valid empty `<function=name></function>` with no `json.parse`. pflash is a different code path from the webui chat request that 500s inside llama.cpp's Jinja path — cited only as the safe-handling pattern to copy.
2. **Unbreak now:** deploy #1 (auto-heals), or delete the offending message / clear the conversation in webui IndexedDB.
3. **Output-side prevention** (after triage): verify `predict` headroom, investigate MTP early-EOG during tool calls, prefer `finish_reason:"length"` with no malformed `tool_calls`.

## Upstreamability of the fix

- **Gateway sanitization (our primary fix): not upstreamable.** It lives in grimoire's proxy and is a workaround over upstream behavior. Stays local. This is the right home for the *content-aware* richness (distinguishing empty-ish from content-bearing truncation), which is application policy, not library policy.
- **Graceful `func_args_not_string` (fixes defect #2): moderately–highly upstreamable.** Lowest-friction PR shape: on `json::parse` failure, `LOG_WRN` and fall back to `json::object()` (`{}`) instead of throwing — a ~3-line change, strictly better than a hard crash, localized, addressing open #22948. Risks/friction: (a) **policy is undecided** — maintainers must choose repair-to-`{}` vs drop-the-call vs return **400** (the data came from the client, so a 4xx is arguably "more correct" than silent repair) vs error-to-model; #22948 has *no maintainer engagement yet*, signaling the decision is unmade. (b) The autoparser/tool-calling area is under active refactor (#20198, #20359), raising the coordination bar. Net: the change is small and clearly-better-than-crash, but expect bikeshedding on the *semantics*, not the mechanics.
- **Autoparser `{`-production (defect #1): not realistically ours to upstream.** It is deep in the model-format-specific qwen3_coder PEG reconstruction. Better to file a focused repro (we run the exact model in #22948) than to attempt the parser fix ourselves.

Highest-leverage upstream move: **add our clean repro + root-cause trace to #22948** — we run the exact model, which is more likely to get it triaged than a cold PR landing into an undecided-policy area.

**Done (2026-06-21):** root-cause comment posted to #22948 — https://github.com/ggml-org/llama.cpp/issues/22948#issuecomment-4759933779 — locating the throw in `func_args_not_string`, explaining why server restarts don't help (client-side IndexedDB replay), and proposing the localized graceful-degradation fix. Local fix #1 shipped as `src/grimoire/plugins/tool_arg_sanitize.py` (`ToolArgSanitizePlugin`, key `TOOL_ARG_SANITIZE`, default on) with `tests/test_tool_arg_sanitize.py`.

## Decisive check (do first)

Which tool parser does the serving binary select for the buun template? It logs the detected chat format at startup. Not `qwen3_coder` ⇒ generic fallback ⇒ origin B confirmed and template-detection is the lever. Cross-check by capturing one failing request body: bare `{` ⇒ B; content-bearing-truncated ⇒ A. This also tells whether fix #1's content-bearing branch and the output-side prevention are needed.

## Relationship to in-flight work

Gateway sanitization is **orthogonal** to the uncommitted `proxy/sse.py` error-surfacing work (`_extract_error_message` / `_sse_error_frames`, sse.py:116-184): that makes backend errors *visible* to the client; sanitization makes this one *not happen*. Complementary, not the same workstream.

## Provenance

Diagnosis converged through a 3-round neutral-subagent sharpen loop (`skills/sharpen-the-tip`). Corrections across rounds: provenance of `func_args_not_string` upgraded from "fork-specific" (wrong) to "mainline, verified in the serving fork"; the `{}` repair refined to content-aware after the lossy-truncation finding; root-cause re-leaned from A (truncation) toward B (parser) after the upstream-issue survey.
