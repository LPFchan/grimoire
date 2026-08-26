"""Chat-template defaults shared by model startup and request proxying."""

import json
from collections.abc import Iterable, Mapping


CHAT_TEMPLATE_KWARGS_FLAG = "--chat-template-kwargs"
REQUEST_ONLY_TEMPLATE_KWARGS = {"reasoning_effort", "reasoning_strength"}


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


def apply_chat_template_kwargs(payload: dict, cfg: Mapping, family_defaults: Mapping | None = None) -> dict:
    """Apply configured defaults while preserving explicit request overrides."""
    configured = configured_chat_template_kwargs(cfg, family_defaults)
    requested = payload.get("chat_template_kwargs")
    if isinstance(requested, Mapping):
        configured.update(requested)
    if configured:
        payload["chat_template_kwargs"] = configured
    return payload
