"""PFlash awareness plugin — injects runtime note into system prompts."""

import copy

from grimoire.plugins.base import Plugin

PFLASH_AWARENESS_MARKER = "Grimoire PFlash runtime note:"
PFLASH_RECALL_TOOL = "conversation_recall"


def payload_tool_names(payload):
    names = set()
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]:
            names.add(fn["name"])
    for fn in payload.get("functions") or []:
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"]:
            names.add(fn["name"])
    return names


def content_contains_text(content, text):
    if isinstance(content, str):
        return text in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and text in item["text"]:
                return True
    return False


def append_text_to_content(content, text):
    if isinstance(content, str):
        return f"{content}\n\n{text}" if content else text
    if content is None:
        return text
    if isinstance(content, list):
        return [
            *copy.deepcopy(content),
            {"type": "text", "text": f"\n\n{text}" if content else text},
        ]
    return f"{content}\n\n{text}" if content else text


class PflashAwarenessPlugin(Plugin):
    """Inject a runtime note when retrieval-aware sessions run on PFlash."""

    def _default_enabled(self) -> bool:
        return True

    def _info(self) -> dict:
        return {"name": "PFlash Awareness", "key": "PFLASH_AWARENESS", "description": "Injects a runtime note about PFlash long-context compression into system prompts when retrieval tools are used"}

    def before_request(self, payload, model_name, model_cfg):
        if not self._is_enabled():
            return payload
        if model_cfg.get("speculative-type") != "pflash":
            return payload
        if model_cfg.get("prefill-compression", model_cfg.get("prefill_compression")) == "never":
            return payload
        if not model_cfg.get("drafter"):
            return payload
        if PFLASH_RECALL_TOOL not in payload_tool_names(payload):
            return payload

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            if content_contains_text(message.get("content"), PFLASH_AWARENESS_MARKER):
                return payload

        threshold = model_cfg.get("prefill-threshold", model_cfg.get("prefill_threshold"))
        try:
            threshold = int(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold = None

        threshold_hint = (
            f"On long prompts (around {threshold:,}+ rendered tokens before compression), "
            if threshold and threshold > 0
            else "On long prompts, "
        )
        context = (
            f"{PFLASH_AWARENESS_MARKER} This session runs on Grimoire with PFlash long-context compression available. "
            f"{threshold_hint}older middle context may be compressed before target prefill, while the head, recent tail, "
            "and protected tool blocks are preferentially kept exact. If you need exact older wording or the original "
            "contents of an older message block, use the `conversation_recall` tool instead of assuming the compressed "
            "middle is verbatim."
        )

        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            messages[0]["content"] = append_text_to_content(messages[0].get("content"), context)
            return payload

        payload["messages"] = [{"role": "system", "content": context}, *messages]
        return payload
