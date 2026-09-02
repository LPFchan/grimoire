"""Llama-server proxy path — chat completions and generic v1 forwarding."""

import asyncio
import copy
import math
import json
import logging
import re

from fastapi.responses import StreamingResponse

from grimoire import config
from grimoire.chat_template import apply_chat_template_kwargs
from grimoire.proxy.client import get_proxy_client
from grimoire.cache import KVCacheStore
from grimoire.plugins import plugin_manager
from grimoire.registry import registry

logger = logging.getLogger(__name__)

# llama-server slot reserved for conversation KV save/restore. The slot lock
# serialises access to it, and requests carrying a conversation id are pinned
# here so the cache is restored into the slot that actually serves them.
CONVERSATION_SLOT = 0


def _slot_lock(active):
    lock = getattr(active, "_kv_slot_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(active, "_kv_slot_lock", lock)
    return lock


def _kv_store(active):
    store = getattr(active, "kv_cache_store", None)
    if store is None:
        cfg = active.cfg
        store = KVCacheStore(
            ram_dir="/dev/shm/grimoire-slots",
            disk_dir=cfg.get("kv-cache-disk-dir", ""),
            disk_budget_gb=cfg.get("kv-cache-disk-budget-gb", 30.0),
            disk_ttl_hours=cfg.get("kv-cache-disk-ttl-hours", 24.0),
            cap=cfg.get("kv-cache-cap", 8),
            kv_k_type=cfg.get("cache-type-k", "q8_0"),
            kv_v_type=cfg.get("cache-type-v", "q8_0"),
            fa_window=cfg.get("fa-window", 2048),
        )
        active.kv_cache_store = store
    return store


def _backend_request_headers(headers):
    """Return request headers safe to forward to an unauthenticated backend."""
    clean = {}
    blocked = config.HOP_BY_HOP_HEADERS | config.SENSITIVE_PROXY_HEADERS
    for key, value in headers.items():
        if key.lower() in blocked:
            continue
        clean[key] = value
    return clean


def _backend_response_headers(headers):
    clean = {}
    for key, value in headers.items():
        if key.lower() in config.HOP_BY_HOP_HEADERS:
            continue
        clean[key] = value
    return clean


_LOGIT_BIAS_CLI_RE = re.compile(r"^(?P<token>\d+)(?P<sign>[+-])(?P<bias>inf|\d+(?:\.\d+)?)$", re.IGNORECASE)

def _json_safe_bias(value):
    value = float(value)
    if math.isinf(value):
        return 100.0 if value > 0 else -100.0
    return value

def _parse_logit_bias_entries(raw):
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return {str(int(token)): _json_safe_bias(bias) for token, bias in raw.items()}
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return {}

    parsed = {}
    for item in raw:
        if isinstance(item, str):
            match = _LOGIT_BIAS_CLI_RE.match(item.strip())
            if not match:
                continue
            token = str(int(match.group("token")))
            bias_text = match.group("bias").lower()
            bias = math.inf if bias_text == "inf" else float(bias_text)
            if match.group("sign") == "-":
                bias = -bias
            parsed[token] = _json_safe_bias(bias)
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            token, bias = item
            parsed[str(int(token))] = _json_safe_bias(bias)
            continue
        if isinstance(item, dict):
            parsed.update(_parse_logit_bias_entries(item))
    return parsed

def _apply_model_logit_bias(payload, model_cfg):
    """Merge configured model logit bias into a request payload.

    `logit-bias` is also used for llama-server CLI flags, so it commonly uses
    strings like `262143+5` or `111038-inf`. The backend JSON API expects a
    finite JSON object, so map infinities to hard bias values and let explicit
    request bias override model defaults.
    """
    if not isinstance(payload, dict):
        return payload
    configured = _parse_logit_bias_entries(model_cfg.get("request-logit-bias", model_cfg.get("logit-bias")))
    if not configured:
        return payload
    merged = dict(configured)
    merged.update(_parse_logit_bias_entries(payload.get("logit_bias")))
    if merged:
        payload["logit_bias"] = merged
    return payload


def _telemetry_gpu_index(active):
    """Return a physical GPU only when one card owns all reported throughput."""
    gpus = getattr(active, "gpus", [] if active.gpu is None else [active.gpu])
    return active.gpu if len(gpus) == 1 else None


async def _proxy_chat(
    requested_model,
    payload,
    active,
    user_hash=None,
    conversation_id=None,
    history_conversation_id=None,
    record_usage=True,
):
    """Proxy chat completions while keeping the upstream client open."""
    # Local imports avoid circular dependency with entrypoint.
    from grimoire.entrypoint import _record_response_stream

    active_cfg = active.cfg
    requested_name = registry.resolve(requested_model) or requested_model
    model_cfg = active_cfg if requested_name == active.name else (registry.get(requested_name) or active_cfg)

    log = logging.getLogger(__name__)

    payload = copy.deepcopy(payload)
    if payload.get("stream"):
        so = payload.get("stream_options")
        if isinstance(so, dict):
            so["include_usage"] = True
        else:
            payload["stream_options"] = {"include_usage": True}
    family_defaults = registry.get_family_defaults(model_cfg.get("family"))
    payload = apply_chat_template_kwargs(payload, model_cfg, family_defaults)
    payload = plugin_manager.before_request(payload, requested_name, model_cfg)
    backend_model_id = await active.get_backend_model_id()
    payload["model"] = backend_model_id
    url = f"http://127.0.0.1:{active.port}/v1/chat/completions"
    headers = {}
    validated_conversation_id = conversation_id if isinstance(conversation_id, str) else None
    if validated_conversation_id:
        validated_conversation_id = "\0".join(
            (active.name, user_hash or "anonymous", validated_conversation_id)
        )

    store = _kv_store(active)

    client = get_proxy_client()
    slot_guard = None
    slot_url = None
    needs_slot_guard = validated_conversation_id is not None
    if needs_slot_guard:
        slot_guard = _slot_lock(active)
        await slot_guard.acquire()
    try:
        payload = await plugin_manager.before_backend_request(
            payload, requested_name, model_cfg, backend_model_id, client, url, headers
        )
        payload = _apply_model_logit_bias(payload, model_cfg)

        if needs_slot_guard:
            # Three-tier KV cache: VRAM -> RAM (tmpfs) -> SSD.
            # The guard is only for slot save/restore mutations; ordinary chat
            # completions carry no conversation id, skip this branch entirely,
            # and stay concurrent across llama-server's configured slots.
            #
            # Pin the request to the same slot we save and restore. Without
            # this, llama-server assigns any idle slot (`id_slot` defaults to
            # -1), so on a model running more than one slot the cache would be
            # restored into slot 0 while the request ran somewhere else. That is
            # invisible today only because every registered model sets
            # parallel=1.
            payload["id_slot"] = CONVERSATION_SLOT
            slot_url = f"http://127.0.0.1:{active.port}/slots/{CONVERSATION_SLOT}"
            prev_conv = getattr(active, "_current_conv_id", None)
            if validated_conversation_id and validated_conversation_id != prev_conv:
                # Same-model conversation switch: save old to tmpfs, restore target.
                if prev_conv:
                    await store.save_conv(client, slot_url, prev_conv)
                await store.restore_conv(client, slot_url, validated_conversation_id)
                active._current_conv_id = validated_conversation_id

        upstream = await client.send(
            client.build_request(
                "POST",
                url,
                headers=headers,
                json=payload,
            ),
            stream=True,
        )
    except Exception:
        if slot_guard is not None:
            slot_guard.release()
        raise

    non_streaming = not payload.get("stream", True)

    async def body_iter():
        try:
            stream = upstream.aiter_raw()
            stream = plugin_manager.wrap_response_stream(stream, requested_name, model_cfg)
            if user_hash:
                stream = _record_response_stream(
                    stream,
                    user_hash,
                    history_conversation_id,
                    requested_name,
                    model_cfg,
                    payload,
                    gpu_index=_telemetry_gpu_index(active),
                    record_history=upstream.status_code < 400,
                    record_usage=record_usage,
                )
            if non_streaming:
                body_parts = []
                async for chunk in stream:
                    body_parts.append(chunk)
                body = b"".join(body_parts)
                try:
                    data = json.loads(body)
                    if "choices" in data:
                        data["context_window"] = model_cfg.get("ctx-size", config.DEFAULT_CTX_SIZE)
                    body = json.dumps(data).encode()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                yield body
            else:
                async for chunk in stream:
                    yield chunk
        finally:
            try:
                await upstream.aclose()
            finally:
                if slot_guard is not None:
                    slot_guard.release()

    resp_headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        resp_headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers)
