#!/usr/bin/env python3
"""Grimoire entrypoint - handles model selection, gateway startup, and lifecycle."""

import argparse
import asyncio
import copy
import ctypes
from collections import OrderedDict
from contextlib import asynccontextmanager
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from grimoire import config
from grimoire.auth import (
    require_api,
    require_admin,
    _require_login_enabled,
    _render_login_html,
    router as auth_router,
)
from grimoire.config import (
    LLAMA_SERVER_BIN,
    DEFAULT_CTX_SIZE,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_PREDICT,
    COOKIE_NAME,
    DEFAULT_STARTUP_TIMEOUT,
    MAX_HISTORY_CAPTURE_BYTES,
    MAX_USAGE_CAPTURE_BYTES,
    QWEN_PROMPT_BLOCK_CACHE_SIZE,
    LEGACY_STATS_PATH,
    WEBUI_DIR,
    HOP_BY_HOP_HEADERS,
    SENSITIVE_PROXY_HEADERS,
    PR_SET_PDEATHSIG,
    MODEL_STATUS_UNLOADED,
    MODEL_STATUS_LOADING,
    MODEL_STATUS_LOADED,
    MODEL_STATUS_FAILED,
    DASHBOARD_WINDOWS_S,
    DASHBOARD_BINS,
    DFLASH_PROTECTED_TOOLS,
    DFLASH_SUPPORTED_SAMPLING,
    DFLASH_IGNORED_SAMPLING,
    DEFAULT_GENERATION_PARAMS,
)
from grimoire.dflash import DflashDaemon, PrefixCache, PrefillConfig, SessionKV, SnapshotSwap
from grimoire.dflash.prefill import PromptBlock, materialize_blocks, maybe_compress
from grimoire.history import history_store, identity_hash
from grimoire.ingest import download_model_file, model_filename_from_url
from grimoire.plugins import plugin_manager
from grimoire.registry import (
    MODELS_DIR,
    registry,
    resolve_path,
    _looks_like_local_path,
    BACKEND_LLAMA,
    BACKEND_DFLASH,
)
from grimoire.prompt import _tool_name_from_message
from grimoire.prompt.qwen import (
    _qwen_prompt_block_specs,
    _qwen_prompt_blocks,
    _prompt_block_cache_for,
    _tokenize_qwen_prompt_blocks,
    _encode_qwen_prompt_blocks,
)
from grimoire.prompt.generic import (
    _generic_prompt_blocks,
    _prompt_layout_from_messages,
    _prefix_cache_boundaries,
    _prompt_ids_from_messages,
)
from grimoire.model_manager import (
    build_cmd,
    ActiveModel,
    ModelManager,
    detect_gpu_count,
)
from grimoire.proxy.llama import (
    _proxy_chat,
    _backend_request_headers,
    _backend_response_headers,
)
from grimoire.proxy.dflash import _proxy_dflash, _dflash_collect_stop_ids
from grimoire.routes.history import router as history_router
from grimoire.routes.dashboard import router as dashboard_router
from grimoire.routes.models import router as models_router
from grimoire.proxy.sse import (
    _extract_assistant_text,
    _usage_from_object,
    _extract_usage,
    _extract_tokens_per_sec,
    _extract_chunk_tokens_per_sec,
    _sse_error_frames,
    _delta_sse,
    _final_sse,
)
from grimoire.telemetry import telemetry_sampler, telemetry_store
from grimoire.usage import usage_store

logger = logging.getLogger(__name__)

# Keep a single module identity under `python -m grimoire.entrypoint` so
# extracted modules importing `grimoire.entrypoint` reuse the live gateway
# state instead of creating a second module instance.
sys.modules.setdefault("grimoire.entrypoint", sys.modules[__name__])


def parse_args():
    parser = argparse.ArgumentParser(description="Grimoire multi-GPU inference server")
    parser.add_argument("--model", help="Model name to start (from registry)")
    parser.add_argument("--port", type=int, default=9001, help="Gateway port (default: 9001)")
    parser.add_argument("--host", default="0.0.0.0", help="Gateway host (default: 0.0.0.0)")
    return parser.parse_args()



def _cost_by_model():
    data = registry.snapshot()
    return {
        name: cfg.get("cost", {})
        for name, cfg in data.get("models", {}).items()
        if isinstance(cfg, dict)
    }


manager = ModelManager(gpu_count=detect_gpu_count())
logger.info(f"Grimoire starting with {manager.gpu_count} GPU(s)")


@asynccontextmanager
async def lifespan(_app):
    imported = usage_store.import_legacy_token_stats(
        LEGACY_STATS_PATH,
        identity_hash(config.API_KEY or "anonymous"),
        cost_by_model=_cost_by_model(),
    )
    if imported:
        logger.info(f"Imported legacy token stats from {LEGACY_STATS_PATH}")

    initial_model = getattr(_app.state, "initial_model", None)
    if initial_model:
        await manager.start_model(initial_model)
    sampler_task = asyncio.create_task(telemetry_sampler())
    try:
        yield
    finally:
        sampler_task.cancel()
        try:
            await sampler_task
        except (asyncio.CancelledError, Exception):
            pass
        await manager.shutdown()


app = FastAPI(title="Grimoire Gateway", version="0.1.0", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(models_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_models": manager.list_active()
    }


def _history_conversation_id(request, payload):
    if request.headers.get("x-grimoire-conversation-id"):
        return request.headers["x-grimoire-conversation-id"]
    if isinstance(payload.get("conversation_id"), str):
        return payload["conversation_id"]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("conversation_id"), str):
        return metadata["conversation_id"]
    return None


def _validated_history_conversation_id(user_hash, conversation_id):
    if not conversation_id:
        return None
    if not history_store.conversation_exists(user_hash, conversation_id):
        return None
    return conversation_id


async def _record_response_stream(stream, user_hash, conversation_id, model_name, model_cfg, payload, gpu_index=None, record_history=True):
    captured = bytearray()
    usage_tail = bytearray()
    try:
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if record_history and conversation_id and isinstance(messages, list):
            message = next((m for m in reversed(messages) if isinstance(m, dict)), None)
            if message and message.get("role") != "assistant":
                try:
                    history_store.append_message(
                        user_hash,
                        conversation_id,
                        message.get("role", "user"),
                        message.get("content"),
                        model=model_name,
                    )
                except KeyError:
                    conversation_id = None

        async for chunk in stream:
            if MAX_USAGE_CAPTURE_BYTES > 0:
                usage_tail.extend(chunk)
                if len(usage_tail) > MAX_USAGE_CAPTURE_BYTES:
                    del usage_tail[:len(usage_tail) - MAX_USAGE_CAPTURE_BYTES]
            if len(captured) < MAX_HISTORY_CAPTURE_BYTES:
                remaining = MAX_HISTORY_CAPTURE_BYTES - len(captured)
                captured.extend(chunk[:remaining])
            yield chunk
    finally:
        raw = bytes(captured)
        usage = _extract_usage(raw)
        if not usage:
            usage = _extract_usage(bytes(usage_tail))
        if usage:
            usage_store.record(
                user_hash,
                model_name,
                usage["input_tokens"],
                usage["output_tokens"],
                cost_rates=model_cfg.get("cost"),
            )

        if gpu_index is not None:
            tps = _extract_tokens_per_sec(raw)
            if tps is None:
                tps = _extract_tokens_per_sec(bytes(usage_tail))
            if tps is not None and tps > 0:
                telemetry_store.record(time.time(), [(gpu_index, "gpu_tokens_per_sec", tps)])

        assistant_text = _extract_assistant_text(raw)
        if record_history and assistant_text and conversation_id:
            try:
                history_store.append_message(user_hash, conversation_id, "assistant", assistant_text, model=model_name)
            except KeyError:
                pass


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions to the correct active model."""
    _, user_hash = require_api(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    requested_model = payload.get("model")
    model_name = registry.resolve(requested_model)
    if not model_name:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{requested_model}' was not found in the registry."
        )

    try:
        active = await manager.start_model(model_name)
        conversation_id = _history_conversation_id(request, payload)
        conversation_id = _validated_history_conversation_id(user_hash, conversation_id)
        return await _proxy_chat(requested_model, payload, active, user_hash=user_hash, conversation_id=conversation_id)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to forward request: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_v1(request: Request, path: str):
    """Proxy other OpenAI-compatible routes to the requested or active backend."""
    require_api(request)
    payload = None
    body = await request.body()
    if body and request.headers.get("content-type", "").split(";")[0] == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None

    requested_model = payload.get("model") if isinstance(payload, dict) else None
    model_name = registry.resolve(requested_model) if requested_model else None
    if not model_name:
        active_names = manager.list_active()
        if len(active_names) == 1:
            model_name = active_names[0]
    if not model_name:
        raise HTTPException(status_code=404, detail="No target model resolved for proxy request")

    client = None
    try:
        active = await manager.start_model(model_name)
        if active.backend_type == BACKEND_DFLASH:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"/v1/{path} is not supported on dflash backend "
                    f"'{active.name}'; only /v1/chat/completions and /v1/models are wired up"
                ),
            )
        client = httpx.AsyncClient(timeout=None)
        headers = _backend_request_headers(request.headers)

        if isinstance(payload, dict):
            payload = copy.deepcopy(payload)
            payload["model"] = await active.get_backend_model_id()
            req = client.build_request(
                request.method,
                f"http://127.0.0.1:{active.port}/v1/{path}",
                headers=headers,
                params=request.query_params,
                json=payload,
            )
        else:
            req = client.build_request(
                request.method,
                f"http://127.0.0.1:{active.port}/v1/{path}",
                headers=headers,
                params=request.query_params,
                content=body,
            )

        upstream = await client.send(req, stream=True)
    except HTTPException:
        if client:
            await client.aclose()
        raise
    except Exception as e:
        if client:
            await client.aclose()
        logger.error(f"Failed to proxy /v1/{path}: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = _backend_response_headers(upstream.headers)
    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=response_headers)


def _mount_webui():
    """Mount the built llama.cpp webui as the root chat surface, if available."""
    if not os.path.isdir(WEBUI_DIR):
        logger.warning(
            "GRIMOIRE_WEBUI_DIR=%s does not exist; chat UI will return 404. "
            "Build the webui in your image or set GRIMOIRE_WEBUI_DIR to its build output.",
            WEBUI_DIR,
        )
        return
    app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")
    logger.info("Serving llama.cpp webui from %s", WEBUI_DIR)


_mount_webui()


def main():
    args = parse_args()
    app.state.initial_model = args.model
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
