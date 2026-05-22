"""Structured CoT plugin — injects CoT grammar for Qwen-family models."""

import os

from grimoire.plugins.base import Plugin, env_flag, logger


class QwenStructuredCotPlugin(Plugin):
    """Inject structured CoT grammar for Qwen-family models when configured."""

    def __init__(self):
        super().__init__()
        self.tool_fallback = env_flag("STRUCTURED_TOOL_FALLBACK", False)
        self.grammar = self._load_grammar()
        self._missing_warned = False
        if self._is_enabled() and not self.grammar:
            logger.warning(
                "STRUCTURED_COT is enabled but no grammar was found at "
                "$GRIMOIRE_STRUCTURED_COT_GRAMMAR, /etc/grimoire/grammars/fsm_grammar.gbnf, or "
                "/home/yeowool/structured-cot/grammars/fsm_grammar.gbnf — qwen requests will "
                "run without structured CoT. Mount the grammar dir or set STRUCTURED_COT=0."
            )

    def _default_enabled(self) -> bool:
        return env_flag("STRUCTURED_COT", True)

    def _info(self) -> dict:
        return {"name": "Structured CoT", "key": "STRUCTURED_COT", "description": "Injects structured chain-of-thought grammar for Qwen-family models"}

    def _load_grammar(self):
        paths = [
            os.environ.get("GRIMOIRE_STRUCTURED_COT_GRAMMAR"),
            "/etc/grimoire/grammars/fsm_grammar.gbnf",
            "/home/yeowool/structured-cot/grammars/fsm_grammar.gbnf",
        ]
        for path in paths:
            if not path:
                continue
            try:
                with open(path) as f:
                    logger.info(f"Loaded structured CoT grammar from {path}")
                    return f.read()
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Failed to load structured CoT grammar from {path}: {e}")
        return None

    def before_request(self, payload, model_name, model_cfg):
        if not self._is_enabled():
            return payload
        if model_cfg.get("family") != "qwen":
            return payload
        if not self.grammar:
            if not self._missing_warned:
                logger.warning(
                    "Skipping structured CoT for qwen-family model %s: grammar not loaded.",
                    model_name,
                )
                self._missing_warned = True
            return payload
        if "grammar" in payload:
            return payload

        has_tools = any(key in payload for key in ("tools", "tool_choice", "functions", "function_call"))
        if has_tools and not self.tool_fallback:
            return payload

        messages = payload.get("messages", [])
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") in ("tool", "function"):
            return payload

        payload["grammar"] = self.grammar
        return payload
