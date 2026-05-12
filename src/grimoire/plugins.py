"""Model-family plugin hooks for Grimoire request and response transforms."""

import logging
import os
import copy
import re

logger = logging.getLogger(__name__)

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

DFLASH_AWARENESS_MARKER = "Grimoire DFlash runtime note:"
DFLASH_RECALL_TOOL = "conversation_recall"


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


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


class Plugin:
    """Base plugin hook interface."""

    def before_request(self, payload, model_name, model_cfg):
        return payload

    def wrap_response_stream(self, stream, model_name, model_cfg):
        return stream

    async def before_backend_request(self, payload, model_name, model_cfg, backend_model_id, client, url, headers):
        return payload


class PluginManager:
    """Apply plugin hooks in a stable order."""

    def __init__(self, plugins):
        self.plugins = plugins

    def before_request(self, payload, model_name, model_cfg):
        for plugin in self.plugins:
            payload = plugin.before_request(payload, model_name, model_cfg)
        return payload

    def wrap_response_stream(self, stream, model_name, model_cfg):
        for plugin in self.plugins:
            stream = plugin.wrap_response_stream(stream, model_name, model_cfg)
        return stream

    async def before_backend_request(self, payload, model_name, model_cfg, backend_model_id, client, url, headers):
        for plugin in self.plugins:
            payload = await plugin.before_backend_request(
                payload, model_name, model_cfg, backend_model_id, client, url, headers
            )
        return payload


class QwenStructuredCotPlugin(Plugin):
    """Inject structured CoT grammar for Qwen-family models when configured."""

    def __init__(self):
        self.enabled = env_flag("STRUCTURED_COT", True)
        self.tool_fallback = env_flag("STRUCTURED_TOOL_FALLBACK", False)
        self.grammar = self._load_grammar()
        self._missing_warned = False
        if self.enabled and not self.grammar:
            logger.warning(
                "STRUCTURED_COT is enabled but no grammar was found at "
                "$GRIMOIRE_STRUCTURED_COT_GRAMMAR, /etc/grimoire/grammars/fsm_grammar.gbnf, or "
                "/home/yeowool/structured-cot/grammars/fsm_grammar.gbnf — qwen requests will "
                "run without structured CoT. Mount the grammar dir or set STRUCTURED_COT=0."
            )

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
        if not self.enabled:
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
        if has_tools and self.tool_fallback:
            return payload

        payload["grammar"] = self.grammar
        return payload


class DflashPflashAwarenessPlugin(Plugin):
    """Inject a runtime note when retrieval-aware sessions run on DFlash/PFlash."""

    def before_request(self, payload, model_name, model_cfg):
        if model_cfg.get("backend") != "dflash":
            return payload
        if model_cfg.get("prefill-compression", model_cfg.get("prefill_compression")) == "never":
            return payload
        if not model_cfg.get("drafter"):
            return payload
        if DFLASH_RECALL_TOOL not in payload_tool_names(payload):
            return payload

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            if content_contains_text(message.get("content"), DFLASH_AWARENESS_MARKER):
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
            f"{DFLASH_AWARENESS_MARKER} This session runs on Grimoire DFlash with PFlash long-context compression available. "
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


class HuihuiPlugin(Plugin):
    """Apply Huihui model request defaults and streamed response cleanup."""

    def before_request(self, payload, model_name, model_cfg):
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


class StructuredToolPlanPlugin(Plugin):
    """Optional Qwen-family tool-routing prepass using a compact grammar."""

    TOOL_PLAN_RE = re.compile(
        r"^<think>\n"
        r"GOAL: (?P<goal>[^\n]{1,220})\n"
        r"NEXT: (?P<next>[^\n]{1,220})\n"
        r"TOOL_HINT: (?P<tool_hint>[^\n]{1,220})\n"
        r"RISK: (?P<risk>[^\n]{1,220})\n"
        r"</think>\s*$"
    )

    def __init__(self):
        self.enabled = env_flag("STRUCTURED_TOOL", False)
        self.max_tokens = int(os.environ.get("TOOL_STRUCTURED_PLAN_MAX_TOKENS", "192"))
        self.timeout_s = float(os.environ.get("TOOL_STRUCTURED_PLAN_TIMEOUT_S", "8"))
        self.tiny_max_tokens = int(os.environ.get("TOOL_STRUCTURED_PLAN_TINY_MAX_TOKENS", "64"))
        self.grammar = self._load_grammar()

    def _load_grammar(self):
        paths = [
            os.environ.get("GRIMOIRE_TOOL_PLAN_GRAMMAR"),
            "/etc/grimoire/grammars/tool_plan_grammar.gbnf",
            "/home/yeowool/structured-cot/grammars/tool_plan_grammar.gbnf",
        ]
        for path in paths:
            if not path:
                continue
            try:
                with open(path) as f:
                    logger.info(f"Loaded tool planning grammar from {path}")
                    return f.read()
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Failed to load tool planning grammar from {path}: {e}")
        return None

    async def before_backend_request(self, payload, model_name, model_cfg, backend_model_id, client, url, headers):
        skip_reason = self._skip_reason(payload, model_cfg)
        if skip_reason:
            if self._has_tool_request(payload) and skip_reason != "no_tools":
                logger.info("tool-plan skipped: %s", skip_reason)
            return payload

        plan = await self._try_get_tool_plan(client, url, headers, payload, backend_model_id)
        if not plan:
            return payload
        return self._inject_tool_plan(payload, plan)

    def _skip_reason(self, payload, model_cfg):
        if not self.enabled:
            return "disabled"
        if not self.grammar:
            return "missing_grammar"
        if model_cfg.get("family") != "qwen":
            return "unsupported_model"
        if not self._has_tool_request(payload):
            return "no_tools"
        if payload.get("tool_choice") == "none" or payload.get("function_call") == "none":
            return "tool_choice_none"
        if self._forces_specific_tool(payload):
            return "forced_tool_choice"
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") in {"tool", "function"}:
                return "after_tool_result"
        max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if isinstance(max_tokens, int) and max_tokens <= self.tiny_max_tokens:
            return "tiny_max_tokens"
        return None

    @staticmethod
    def _has_tool_request(payload):
        return any(key in payload for key in ("tools", "tool_choice", "functions", "function_call"))

    @staticmethod
    def _forces_specific_tool(payload):
        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, dict):
            return bool(tool_choice.get("function", {}).get("name") or tool_choice.get("name"))
        function_call = payload.get("function_call")
        if isinstance(function_call, dict):
            return bool(function_call.get("name"))
        return False

    @staticmethod
    def _compact_schema_hint(schema):
        if not isinstance(schema, dict):
            return ""
        props = schema.get("properties")
        if isinstance(props, dict) and props:
            names = ", ".join(str(name) for name in list(props.keys())[:8])
            return f" args: {names}"
        return ""

    def _build_tool_manifest(self, payload):
        lines = ["Available tools:"]
        for tool in payload.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            desc = " ".join(str(fn.get("description") or "").split())
            line = f"- {fn['name']}{self._compact_schema_hint(fn.get('parameters'))}"
            if desc:
                line += f": {desc[:180]}"
            lines.append(line[:360])
        for fn in payload.get("functions") or []:
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            desc = " ".join(str(fn.get("description") or "").split())
            line = f"- {fn['name']}{self._compact_schema_hint(fn.get('parameters'))}"
            if desc:
                line += f": {desc[:180]}"
            lines.append(line[:360])
        if len(lines) == 1:
            lines.append("- tools are present, but no compact schema summary was available")
        return "\n".join(lines)[:4000]

    def _make_plan_payload(self, payload, backend_model_id):
        plan_payload = copy.deepcopy(payload)
        for key in ("tools", "tool_choice", "functions", "function_call", "grammar", "json_schema", "response_format"):
            plan_payload.pop(key, None)
        messages = copy.deepcopy(plan_payload.get("messages") or [])
        messages.append({
            "role": "user",
            "content": (
                "Create only a compact tool-routing plan using the required format. "
                "Do not answer the user and do not call tools in this pass.\n\n"
                f"{self._build_tool_manifest(payload)}"
            ),
        })
        plan_payload["messages"] = messages
        plan_payload["model"] = backend_model_id
        plan_payload["stream"] = False
        plan_payload["max_tokens"] = self.max_tokens
        plan_payload.pop("max_completion_tokens", None)
        plan_payload["grammar"] = self.grammar
        return plan_payload

    @staticmethod
    def _extract_message_content(completion):
        choices = completion.get("choices") if isinstance(completion, dict) else None
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content") or message.get("reasoning_content")
            if isinstance(content, str):
                return content
        text = first.get("text")
        return text if isinstance(text, str) else ""

    def _normalize_tool_plan(self, plan):
        plan = plan.strip()
        if self.TOOL_PLAN_RE.match(plan):
            return plan
        if plan.startswith("<think>") and not plan.endswith("</think>"):
            closed = f"{plan.rstrip()}\n</think>"
            if self.TOOL_PLAN_RE.match(closed):
                return closed
        return None

    async def _try_get_tool_plan(self, client, url, headers, payload, backend_model_id):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=self._make_plan_payload(payload, backend_model_id),
                timeout=self.timeout_s,
            )
            if response.status_code >= 400:
                logger.info("tool-plan fallback: pass1_status=%s", response.status_code)
                return None
            return self._normalize_tool_plan(self._extract_message_content(response.json()))
        except Exception as e:
            logger.info("tool-plan fallback: %s", type(e).__name__)
            return None

    @staticmethod
    def _inject_tool_plan(payload, plan):
        final_payload = copy.deepcopy(payload)
        messages = copy.deepcopy(final_payload.get("messages") or [])
        context = (
            "Structured tool-routing plan from a bounded prepass. Use it only as routing context; "
            "native tool calls and final answers must still follow the current request.\n\n"
            f"{plan}"
        )
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            existing = messages[0].get("content") or ""
            messages[0]["content"] = f"{existing}\n\n{context}" if existing else context
        else:
            messages.insert(0, {"role": "system", "content": context})
        final_payload["messages"] = messages
        return final_payload


plugin_manager = PluginManager([
    QwenStructuredCotPlugin(),
    DflashPflashAwarenessPlugin(),
    StructuredToolPlanPlugin(),
    HuihuiPlugin(),
])
