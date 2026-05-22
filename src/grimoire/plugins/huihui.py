"""Huihui model plugin — request defaults and streamed response cleanup."""

import re

from grimoire.plugins.base import Plugin, logger

HUIHUI_STOP_SEQUENCES = [
    "<|im_center|>",
    "<|im_id|>",
    "<|im_set|>",
]
HUIHUI_CONTROL_PATTERNS = [
    re.compile(r"<\|channel>thought\s*<channel\|>", re.IGNORECASE),
    re.compile(r"<\|channel>thought\s*", re.IGNORECASE),
    re.compile(r"<channel\|>", re.IGNORECASE),
    re.compile(r"<\|im_[a-z_]+\|>", re.IGNORECASE),
    re.compile(r"<\|im_start\|>\s*(?:thought|assistant)\s*", re.IGNORECASE),
]


class HuihuiPlugin(Plugin):
    """Apply Huihui model request defaults and streamed response cleanup."""

    def _default_enabled(self) -> bool:
        return True

    def _info(self) -> dict:
        return {"name": "Huihui", "key": "HUIHUI", "description": "Applies Huihui model request defaults and strips control tokens from streaming output"}

    def before_request(self, payload, model_name, model_cfg):
        if not self._is_enabled():
            return payload
        if model_name != "huihui-gemma-4-31B":
            return payload

        stops = payload.get("stop")
        if isinstance(stops, str):
            stops = [stops]
        elif not isinstance(stops, list):
            stops = []
        for stop in HUIHUI_STOP_SEQUENCES:
            if stop not in stops:
                stops.append(stop)
        payload["stop"] = stops

        payload.setdefault("repeat_penalty", 1.12)
        payload.setdefault("presence_penalty", 0.2)
        payload.setdefault("temperature", 0.7)
        payload.setdefault("top_p", 0.8)
        return payload

    def wrap_response_stream(self, stream, model_name, model_cfg):
        if not model_name.startswith("huihui-"):
            return stream
        return self._sanitize_stream(stream)

    async def _sanitize_stream(self, stream):
        tail_keep = 96
        pending = ""
        async for chunk in stream:
            pending += chunk.decode("utf-8", errors="ignore")
            if len(pending) <= tail_keep:
                continue

            emit = pending[:-tail_keep]
            pending = pending[-tail_keep:]
            emit = self._sanitize_text(emit)
            if emit:
                yield emit.encode("utf-8")

        if pending:
            pending = self._sanitize_text(pending)
            if pending:
                yield pending.encode("utf-8")

    @staticmethod
    def _sanitize_text(text):
        for pattern in HUIHUI_CONTROL_PATTERNS:
            text = pattern.sub("", text)
        return re.sub(r"\n{3,}", "\n\n", text)
