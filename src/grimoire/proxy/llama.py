"""Llama-server proxy path — chat completions and generic v1 forwarding."""

import asyncio
import copy
import json
import logging

import httpx
from fastapi.responses import StreamingResponse

from grimoire import config
from grimoire.dflash.prefill import materialize_blocks, maybe_compress
from grimoire.plugins import plugin_manager
from grimoire.prompt.generic import _prompt_layout_from_messages
from grimoire.registry import BACKEND_DFLASH, resolve_path

logger = logging.getLogger(__name__)


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


async def _proxy_chat(requested_model, payload, active, user_hash=None, conversation_id=None):
    """Proxy chat completions while keeping the upstream client open."""
    # Local imports avoid circular dependency with entrypoint.
    from grimoire.proxy.dflash import _proxy_dflash
    from grimoire.entrypoint import _record_response_stream

    model_cfg = active.cfg

    if active.backend_type == BACKEND_DFLASH:
        return await _proxy_dflash(requested_model, payload, active, user_hash, conversation_id)

    payload = copy.deepcopy(payload)
    payload = plugin_manager.before_request(payload, active.name, model_cfg)
    backend_model_id = await active.get_backend_model_id()
    payload["model"] = backend_model_id
    url = f"http://127.0.0.1:{active.port}/v1/chat/completions"
    headers = {}

    _kv_save_key = None  # set inside if fired: block when compression fires

    # PFlash compression: if the model has a pflash daemon and the prompt
    # exceeds the threshold, compress before proxying to llama-server.
    daemon = getattr(active, 'pflash_daemon', None)
    pcfg = getattr(active, 'prefill_config', None)
    log = logging.getLogger(__name__)
    log.warning(f"pflash-proxy: daemon={daemon} running={daemon.is_running() if daemon else 'N/A'} pcfg={pcfg}")
    if daemon and daemon.is_running() and pcfg and pcfg.enabled:
        try:
            tokenizer = active.get_tokenizer()
            messages = payload.get("messages", [])
            prompt_ids, prompt_blocks = _prompt_layout_from_messages(
                tokenizer, messages, add_generation_prompt=True,
                model_cfg=model_cfg, active=active,
            )
            if len(prompt_ids) > pcfg.threshold:
                # Park llama-server before compression if park-unpark enabled
                park_ok = False
                _park_ctl_fd = None
                _park_ack_fd = None
                if model_cfg.get("park-unpark"):
                    try:
                        import os, select
                        # Keep .ctl fd open across park+unpark so the FIFO reader
                        # (shim listener) doesn't see EOF between commands.
                        _park_ctl_fd = os.open("/tmp/pflash_shim.ctl",
                                               os.O_WRONLY | os.O_NONBLOCK)
                        os.write(_park_ctl_fd, b"park\n")
                        _park_ack_fd = os.open("/tmp/pflash_shim.ack",
                                               os.O_RDONLY | os.O_NONBLOCK)
                        poll = select.poll()
                        poll.register(_park_ack_fd, select.POLLIN)
                        if poll.poll(30000):
                            resp = os.read(_park_ack_fd, 64).decode().strip()
                            park_ok = (resp == "ok")
                        if park_ok:
                            log.warning("pflash park: llama parked")
                    except Exception as e:
                        log.warning(f"pflash park: failed ({e}) — continuing without park")

                compressed_ids, fired, blocks = await maybe_compress(
                    prompt_ids, daemon, pcfg, blocks=prompt_blocks,
                )

                # Unpark llama-server after compression (reuse same .ctl fd)
                if park_ok and _park_ctl_fd is not None:
                    try:
                        import os, select
                        os.write(_park_ctl_fd, b"unpark\n")
                        poll = select.poll()
                        poll.register(_park_ack_fd, select.POLLIN)
                        if poll.poll(30000):
                            resp = os.read(_park_ack_fd, 64).decode().strip()
                            if resp == "ok":
                                log.warning("pflash park: llama unparked")
                    except Exception as e:
                        log.warning(f"pflash park: unpark failed ({e})")
                    finally:
                        if _park_ctl_fd is not None: os.close(_park_ctl_fd)
                        if _park_ack_fd is not None: os.close(_park_ack_fd)

                log.warning(f"pflash debug: fired={fired} orig={len(prompt_ids)} compressed={len(compressed_ids)}")
                if fired:
                    # Reconstruct messages preserving roles and tool call metadata.
                    # Group compressed block tokens by original message boundary,
                    # decode each message's tokens once (avoids BPE boundary artifacts
                    # from per-block decode), then emit messages in original order.
                    msg_groups: dict[int, dict] = {}
                    for block in blocks:
                        if block.kind == "generation_prompt":
                            continue
                        key = block.message_start
                        if key not in msg_groups:
                            msg_groups[key] = {
                                "role": block.role,
                                "token_ids": [],
                                "metadata": block.metadata or {},
                            }
                        msg_groups[key]["token_ids"].extend(
                            compressed_ids[block.start:block.end]
                        )
                    new_messages = []
                    for key in sorted(msg_groups):
                        m = msg_groups[key]
                        text = tokenizer.decode(m["token_ids"])
                        entry: dict = {"role": m["role"], "content": text}
                        # Propagate tool_call/reasoning metadata from block metadata
                        for md_key in ("tool_calls", "reasoning_content", "tool_call_id",
                                       "name", "tool_names", "message_indexes"):
                            if md_key in m["metadata"]:
                                entry[md_key] = m["metadata"][md_key]
                        new_messages.append(entry)
                    log.warning(f"pflash debug: messages {len(payload['messages'])} -> {len(new_messages)}")
                    payload["messages"] = new_messages
                    if payload.get("conversation_id"):
                        _kv_save_key = f"pflash-{payload['conversation_id']}.kv"
                        log.warning(f"pflash kv: will save as {_kv_save_key}")
        except Exception as e:
            _log = __import__('logging').getLogger(__name__)
            _log.warning(f"PFlash compression failed for {active.name}: {e}")

    client = httpx.AsyncClient(timeout=None)
    try:
        payload = await plugin_manager.before_backend_request(
            payload, active.name, model_cfg, backend_model_id, client, url, headers
        )

        # KV prefix cache: restore saved KV slot to skip re-prefixing shared prefix
        _kv_key_raw = payload.get("conversation_id")
        log.warning(f"pflash kv: raw key = {_kv_key_raw}")
        if _kv_key_raw:
            kv_name = f"pflash-{_kv_key_raw}.kv"
            slot_url = f"http://127.0.0.1:{active.port}/slots/0"
            try:
                rr = await client.post(f"{slot_url}?action=restore",
                    json={"filename": kv_name}, timeout=5)
                log.warning(f"pflash kv: restore status {rr.status_code} for {kv_name}")
                if rr.status_code == 200:
                    log.warning(f"pflash kv: restored {kv_name}")
            except Exception as e:
                log.warning(f"pflash kv: restore failed for {kv_name}: {e}")

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
            # KV prefix cache: save slot state after response completes
            log.warning(f"pflash kv: finally block, _kv_save_key={_kv_save_key}")
            if _kv_save_key:
                try:
                    async with httpx.AsyncClient(timeout=5) as sc:
                        slot_url = f"http://127.0.0.1:{active.port}/slots/0"
                        rr = await sc.post(f"{slot_url}?action=save",
                            json={"filename": _kv_save_key})
                        log.warning(f"pflash kv: save status {rr.status_code} for {_kv_save_key}")
                        if rr.status_code == 200:
                            log.warning(f"pflash kv: saved {_kv_save_key}")
                except Exception as e:
                    log.warning(f"pflash kv: save failed for {_kv_save_key}: {e}")

    resp_headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        resp_headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers)
