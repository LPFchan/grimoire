"""Tool-call argument sanitizer — repairs malformed `arguments` strings in replayed history.

llama.cpp's mainline `func_args_not_string` (common/chat.cpp, `namespace workaround`)
force-parses every incoming `tool_calls[].function.arguments` string with `json::parse()`
at template-apply time and throws a 500 ("Failed to parse tool call arguments as JSON")
if any historical tool call carries an unparseable string (e.g. a truncated `{`). Because
clients replay the full conversation each turn, one poisoned message bricks the whole
conversation — every subsequent turn 500s before generation.

This plugin neutralizes the poison before it reaches the backend: any string `arguments`
that fails `json.loads` is repaired best-effort, or normalized to `{}` when it cannot be
safely recovered. This keeps the request alive and auto-heals already-poisoned
conversations without touching the client's (browser-side) history store.

See records/research/RSH-20260621-001-tool-call-args-json-500.md.
"""

import json

from grimoire.plugins.base import Plugin, env_flag, logger


def _try_repair_json_object(raw: str):
    """Best-effort recovery of a truncated JSON value.

    Returns a parsed Python object if recoverable, else None. Conservative: only
    appends missing closing brackets, and bails when a string literal is left
    unterminated (which cannot be safely closed without inventing content).
    """
    s = raw.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    in_string = False
    escaped = False
    stack = []
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_string:
        return None  # unterminated string — not safely repairable
    candidate = s + "".join(reversed(stack))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class ToolArgSanitizePlugin(Plugin):
    """Repair malformed tool-call `arguments` strings in incoming history.

    Prevents llama.cpp's `func_args_not_string` from 500ing the whole request (and
    bricking the conversation) when a prior assistant turn carries an unparseable
    arguments string such as a truncated `{`.
    """

    def _default_enabled(self) -> bool:
        return env_flag("TOOL_ARG_SANITIZE", True)

    def _info(self) -> dict:
        return {
            "name": "Tool Arg Sanitizer",
            "key": "TOOL_ARG_SANITIZE",
            "description": "Repairs malformed tool_call arguments strings in replayed history to prevent backend 500s (llama.cpp func_args_not_string).",
        }

    def before_request(self, payload, model_name, model_cfg):
        if not self._is_enabled():
            return payload
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if not isinstance(args, str):
                    continue

                stripped = args.strip()
                if stripped:
                    try:
                        json.loads(stripped)
                        continue  # already valid — leave untouched
                    except json.JSONDecodeError:
                        pass

                name = fn.get("name", "?")
                repaired = _try_repair_json_object(stripped) if stripped else None
                if repaired:  # non-empty object/array recovered from content
                    fn["arguments"] = json.dumps(repaired, ensure_ascii=False)
                    logger.warning(
                        "tool-arg-sanitize: recovered truncated arguments for tool %s "
                        "(%d chars in, %d out)",
                        name, len(stripped), len(fn["arguments"]),
                    )
                else:  # empty, bare `{`, or unrecoverable — normalize, but loudly
                    fn["arguments"] = "{}"
                    logger.warning(
                        "tool-arg-sanitize: normalized unparseable arguments to {} for "
                        "tool %s (raw=%r)",
                        name, stripped[:80],
                    )
        return payload
