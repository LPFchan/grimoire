"""Chat-template defaults shared by model startup and request proxying."""

import json
from collections.abc import Iterable, Mapping


CHAT_TEMPLATE_KWARGS_FLAG = "--chat-template-kwargs"
REQUEST_ONLY_TEMPLATE_KWARGS = {"reasoning_effort", "reasoning_strength"}


def _json_object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def split_chat_template_kwargs(args: Iterable[object] | None) -> tuple[list[str], dict]:
    """Separate repeated llama-server template kwargs from ordinary CLI args."""
    values = [str(arg) for arg in (args or [])]
    remaining = []
    merged = {}
    index = 0
    while index < len(values):
        arg = values[index]
        if arg != CHAT_TEMPLATE_KWARGS_FLAG or index + 1 >= len(values):
            remaining.append(arg)
            index += 1
            continue

        raw = values[index + 1]
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            remaining.extend([arg, raw])
        else:
            if isinstance(parsed, Mapping):
                merged.update(parsed)
            else:
                remaining.extend([arg, raw])
        index += 2
    return remaining, merged


def configured_chat_template_kwargs(cfg: Mapping, family_defaults: Mapping | None = None) -> dict:
    """Merge family defaults with model-specific template kwargs."""
    merged = {}
    for source in (family_defaults or {}, cfg):
        _, kwargs = split_chat_template_kwargs(source.get("extra-args", []))
        merged.update(kwargs)
        explicit = source.get("chat-template-kwargs")
        if isinstance(explicit, Mapping):
            merged.update(explicit)
    return merged


def _strict_chat_template_kwargs(source: Mapping) -> tuple[dict, str | None]:
    """Read configured template kwargs while reporting malformed metadata.

    ``configured_chat_template_kwargs`` intentionally preserves the startup
    compatibility behavior of ignoring malformed values.  Public capability
    metadata has to fail closed instead: a malformed alias must not be
    advertised as having a reasoning level that was guessed from the rest of
    its configuration.
    """
    merged = {}
    raw_args = source.get("extra-args", [])
    if raw_args is None:
        raw_args = []
    if not isinstance(raw_args, (list, tuple)):
        return {}, "extra-args must be a list"
    values = [str(arg) for arg in raw_args]
    index = 0
    while index < len(values):
        if values[index] != CHAT_TEMPLATE_KWARGS_FLAG:
            index += 1
            continue
        if index + 1 >= len(values):
            return {}, "chat-template-kwargs is missing its JSON value"
        try:
            parsed = json.loads(
                values[index + 1], object_pairs_hook=_json_object_without_duplicates
            )
        except (TypeError, ValueError):
            return {}, "chat-template-kwargs must be a JSON object"
        if not isinstance(parsed, Mapping) or not all(isinstance(key, str) for key in parsed):
            return {}, "chat-template-kwargs must be a JSON object"
        merged.update(parsed)
        index += 2

    if "chat-template-kwargs" in source:
        explicit = source["chat-template-kwargs"]
        if not isinstance(explicit, Mapping) or not all(isinstance(key, str) for key in explicit):
            return {}, "chat-template-kwargs must be an object"
        merged.update(explicit)
    return merged, None


def configured_reasoning_capability(
    cfg: Mapping, family_defaults: Mapping | None = None
) -> dict:
    """Describe fixed reasoning configured by an alias, without guessing.

    Reasoning kwargs are request-only values in llama.cpp templates.  A
    configured alias therefore exposes one fixed native level, while an alias
    without either recognized kwarg explicitly advertises no reasoning
    control.  Any malformed or conflicting configuration remains unknown.
    """
    merged = {}
    for source in (family_defaults or {}, cfg):
        if not isinstance(source, Mapping):
            return {}
        values, error = _strict_chat_template_kwargs(source)
        if error:
            return {}
        merged.update(values)

    configured = [
        (key, merged[key])
        for key in ("reasoning_effort", "reasoning_strength")
        if key in merged
    ]
    if not configured:
        return {"supported": False, "supported_efforts": []}
    if len(configured) != 1:
        return {}
    value = configured[0][1]
    if not isinstance(value, str) or not value or value.strip() != value:
        return {}
    return {
        "supported_efforts": [value],
        "default_effort": value,
        "default_enabled": True,
        "mandatory": True,
    }


def apply_chat_template_kwargs(payload: dict, cfg: Mapping, family_defaults: Mapping | None = None) -> dict:
    """Apply configured defaults while preserving explicit request overrides."""
    configured = configured_chat_template_kwargs(cfg, family_defaults)
    requested = payload.get("chat_template_kwargs")
    if isinstance(requested, Mapping):
        configured.update(requested)
    if configured:
        payload["chat_template_kwargs"] = configured
    return payload
