"""Llama-server proxy path — chat completions and generic v1 forwarding."""

import asyncio
import copy
import json
import logging

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from grimoire import config
from grimoire.dflash.kv_cache_store import KVCacheStore
from grimoire.dflash.prefill import materialize_blocks, maybe_compress
from grimoire.plugins import plugin_manager
from grimoire.prompt.generic import _prefix_cache_boundaries, _prompt_layout_from_messages
from grimoire.registry import resolve_path

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


async def _proxy_chat(requested_model, payload, active, user_hash=None, conversation_id=None):
    """Proxy chat completions while keeping the upstream client open."""
    # Local imports avoid circular dependency with entrypoint.
    from grimoire.entrypoint import _record_response_stream

    model_cfg = active.cfg

    payload = copy.deepcopy(payload)
    payload = plugin_manager.before_request(payload, active.name, model_cfg)
    backend_model_id = await active.get_backend_model_id()
    payload["model"] = backend_model_id
    url = f"http://127.0.0.1:{active.port}/v1/chat/completions"
    headers = {}
    validated_conversation_id = conversation_id if isinstance(conversation_id, str) else None

    _save_hash: Optional[bytes] = None

    # PFlash compression: if the model has a pflash daemon and the prompt
    # exceeds the threshold, compress before proxying to llama-server.
    # Compression is mandatory once the threshold is hit. If the daemon is
    # unavailable or compression fails, reject the request instead of silently
    # bypassing the long-prompt path.
    daemon = getattr(active, 'pflash_daemon', None)
    pcfg = getattr(active, 'prefill_config', None)
    log = logging.getLogger(__name__)
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

                _save_hash = store.hash_prefix(prompt_ids)
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

    client = httpx.AsyncClient(timeout=None)
    try:
        slot_guard = _slot_lock(active)
        await slot_guard.acquire()
        try:
            payload = await plugin_manager.before_backend_request(
                payload, active.name, model_cfg, backend_model_id, client, url, headers
            )

            # KV prefix cache: restore by content hash, always save hash for later
            slot_url = f"http://127.0.0.1:{active.port}/slots/0"
            if _save_hash is None:
                try:
                    tokenizer = active.get_tokenizer()
                    messages = payload.get("messages", [])
                    pids, _ = _prompt_layout_from_messages(
                        tokenizer, messages, add_generation_prompt=True,
                        model_cfg=model_cfg, active=active,
                    )
                    _save_hash = store.hash_prefix(pids)
                    await _try_restore_kv(client, slot_url, pids, store, log)
                except Exception as e:
                    log.warning(f"kv cache: setup failed: {e}")
            else:
                await _try_restore_kv(client, slot_url, None, store, log, override_hash=_save_hash)

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
            slot_guard.release()
            raise
    except Exception:
        await client.aclose()
        raise

    non_streaming = not payload.get("stream", True)

    async def body_iter():
        try:
            stream = upstream.aiter_raw()
            stream = plugin_manager.wrap_response_stream(stream, active.name, model_cfg)
            if user_hash:
                stream = _record_response_stream(
                    stream,
                    user_hash,
                    conversation_id,
                    active.name,
                    model_cfg,
                    payload,
                    gpu_index=active.gpu,
                    record_history=upstream.status_code < 400,
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
            await upstream.aclose()
            await client.aclose()
            # KV prefix cache: save slot by content hash after response
            if _save_hash:
                async with httpx.AsyncClient(timeout=5) as sc:
                    await _save_kv(sc, slot_url, _save_hash, store, log)
            slot_guard.release()

    resp_headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        resp_headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers)
