"""Grimoire prompt rendering — shared helpers for chat template tokenization."""

from grimoire.config import DFLASH_PROTECTED_TOOLS


def _tool_name_from_message(msg, tool_call_names=None):
    if not isinstance(msg, dict):
        return None
    if msg.get("role") == "tool" and isinstance(msg.get("tool"), dict):
        return msg["tool"].get("name")
    if msg.get("type") == "tool" and isinstance(msg.get("tool"), str):
        return msg["tool"]
    if msg.get("role") == "tool" and tool_call_names and isinstance(msg.get("tool_call_id"), str):
        return tool_call_names.get(msg.get("tool_call_id"))
    return None
