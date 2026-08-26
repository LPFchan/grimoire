"""Llama-server proxy path — chat completions and generic v1 forwarding."""

import asyncio
import copy
import math
import json
import logging
import re

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from grimoire import config
from grimoire.chat_template import apply_chat_template_kwargs
from grimoire.proxy.client import get_proxy_client
from grimoire.dflash.kv_cache_store import KVCacheStore
from grimoire.dflash.prefill import materialize_blocks, maybe_compress
from grimoire.plugins import plugin_manager
from grimoire.prompt.generic import _prefix_cache_boundaries, _prompt_layout_from_messages
from grimoire.registry import registry, resolve_path

logger = logging.getLogger(__name__)


def _slot_lock(active):
    lock = getattr(active, "_pflash_slot_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(active, "_pflash_slot_lock", lock)
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


async def _try_restore_kv(client, slot_url, prompt_ids, store, log, override_hash=None):
    h = override_hash or store.hash_prefix(prompt_ids)
    path = store.lookup(h)
    if not path:
        return None
    try:
        rr = await client.post(f"{slot_url}?action=restore",
            json={"filename": path.name}, timeout=5)
        if rr.status_code == 200:
            return h
    except Exception:
        pass
    return None


async def _save_kv(sc, slot_url, hash_bytes, store, log):
    filename = store.kv_filename(hash_bytes)
    try:
        rr = await sc.post(f"{slot_url}?action=save",
            json={"filename": filename}, timeout=10)
        if rr.status_code == 200:
            store.register(hash_bytes)
            return True
    except Exception as e:
        log.warning(f"kv cache: save failed for {filename}: {e}")
    return False


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

    # Capture daemon and pcfg BEFORE first await to avoid race with eviction.
    # After start_model releases the lock, a concurrent request can evict this
    # model from its GPU, setting pflash_daemon=None and prefill_config=None
    # on the same ActiveModel object. Reading them before the first yield
    # (await below) closes the race window.
    daemon = getattr(active, 'pflash_daemon', None)
    pcfg = getattr(active, 'prefill_config', None)
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

    _save_hash: Optional[bytes] = None

    log.warning(f"pflash-proxy: daemon={daemon} running={daemon.is_running() if daemon else 'N/A'} pcfg={pcfg}")

    store = _kv_store(active)

    if pcfg and pcfg.enabled:
        try:
            tokenizer = active.get_tokenizer()
            messages = payload.get("messages", [])
            prompt_ids, prompt_blocks = _prompt_layout_from_messages(
                tokenizer, messages, add_generation_prompt=True,
                model_cfg=model_cfg, active=active,
            )
            _save_hash = store.hash_prefix(prompt_ids)
            if len(prompt_ids) >= pcfg.threshold:
                if not daemon or not daemon.is_running():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"pflash compression required for {active.name} "
                            f"(prompt={len(prompt_ids)} >= threshold={pcfg.threshold}) "
                            "but pflash daemon is not running. Check that the drafter model "
                            "file exists and the daemon started at model load time."
                        ),
                    )

                is_warm = store.lookup(_save_hash) is not None

                if not is_warm:
                    # ── COLD TURN: park + full compress + unpark ────────────
                    park_ok = False
                    if model_cfg.get("park-unpark"):
                        try:
                            park_ok = await asyncio.to_thread(active._park_llama)
                            if park_ok:
                                log.warning("pflash park: llama parked")
                        except Exception as e:
                            log.warning(f"pflash park: failed ({e}) — continuing without park")

                    try:
                        compressed_ids, fired, blocks = await maybe_compress(
                            prompt_ids, daemon, pcfg, blocks=prompt_blocks,
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=503,
                            detail=f"pflash compression failed for {active.name}: {e}",
                        )
                    finally:
                        if park_ok:
                            try:
                                if await asyncio.to_thread(active._unpark_llama):
                                    log.warning("pflash park: llama unparked")
                            except Exception as e:
                                log.warning(f"pflash park: unpark failed ({e})")

                else:
                    # ── WARM TURN: skip park, only compress delta blocks ────
                    log.warning(f"pflash warm: KV cache exists for {validated_conversation_id}")
                    try:
                        compressed_ids, fired, blocks = await maybe_compress(
                            prompt_ids, daemon, pcfg, blocks=prompt_blocks,
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=503,
                            detail=f"pflash compression failed for {active.name}: {e}",
                        )

                log.warning(f"pflash debug: fired={fired} orig={len(prompt_ids)} compressed={len(compressed_ids)}")
                if not fired:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"pflash compression required for {active.name} "
                            f"(prompt={len(prompt_ids)} >= threshold={pcfg.threshold}) "
                            "but compression declined: no compressible blocks found. "
                            "The prompt is too long for uncompressed serving. "
                            "Reduce prompt size or adjust prefill-threshold/keep-ratio/tail-budget."
                        ),
                    )

                if fired:
                    msg_groups: dict[int, dict] = {}
                    for block in blocks:
                        if block.kind == "generation_prompt":
                            continue
                        key = block.message_start
                        if key not in msg_groups:
                            msg_groups[key] = {
                                "role": block.role,
                                "token_ids": [],
                                "metadata": {},
                                "has_tool_calls": False,
                                "has_reasoning": False,
                                "tool_calls": [],
                            }
                        msg_groups[key]["token_ids"].extend(
                            compressed_ids[block.start:block.end]
                        )
                        meta = block.metadata or {}
                        if meta.get("reasoning"):
                            msg_groups[key]["has_reasoning"] = True
                        if block.kind == "tool_call":
                            msg_groups[key]["has_tool_calls"] = True
                            tool_name = meta.get("tool_name")
                            if tool_name:
                                msg_groups[key]["tool_calls"].append({
                                    "name": tool_name,
                                })
                        msg_groups[key]["metadata"].update(meta)
                    new_messages = []
                    for key in sorted(msg_groups):
                        m = msg_groups[key]
                        text = tokenizer.decode(m["token_ids"])
                        entry: dict = {"role": m["role"], "content": text}
                        for md_key in ("reasoning_content", "tool_call_id",
                                       "tool_names", "message_indexes"):
                            if md_key in m["metadata"]:
                                entry[md_key] = m["metadata"][md_key]
                        if m["has_reasoning"]:
                            entry["reasoning_content"] = text
                        if m["has_tool_calls"]:
                            entry["tool_calls"] = m["tool_calls"]
                        new_messages.append(entry)
                    log.warning(f"pflash debug: messages {len(payload['messages'])} -> {len(new_messages)}")
                    payload["messages"] = new_messages
                    _save_hash = store.hash_prefix(prompt_ids)
                    log.warning(f"pflash kv: will save with hash {_save_hash.hex()[:16]}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"pflash compression setup failed for {active.name}: {e}",
            )

    client = get_proxy_client()
    slot_guard = None
    slot_url = None
    pflash_slots_enabled = bool(pcfg and pcfg.enabled)
    needs_slot_guard = pflash_slots_enabled or validated_conversation_id is not None
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
            # The guard is only for slot-0 save/restore mutations; ordinary
            # chat completions must stay concurrent so llama-server can use
            # its configured parallel slots.
            slot_url = f"http://127.0.0.1:{active.port}/slots/0"
            prev_conv = getattr(active, "_current_conv_id", None)
            if _save_hash is not None:
                # pflash path: restore by compressed content hash
                await _try_restore_kv(client, slot_url, None, store, log, override_hash=_save_hash)
            elif validated_conversation_id and validated_conversation_id != prev_conv:
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
                # KV prefix cache: save slot by content hash (pflash path only)
                if _save_hash and slot_url:
                    async with httpx.AsyncClient(timeout=5) as sc:
                        await _save_kv(sc, slot_url, _save_hash, store, log)
            finally:
                if slot_guard is not None:
                    slot_guard.release()

    resp_headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        resp_headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers)
