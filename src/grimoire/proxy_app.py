"""Stateless data-plane proxy for the multi-process gateway.

Run under `uvicorn --workers N`. Owns the public port. For the stateless encoder
endpoints (/v1/embeddings, /v1/rerank) it resolves the target model from the
shared route table and round-robins across its replica backends (data
parallelism across GPUs). Everything else — chat (stateful: KV slots, pflash,
history), the Responses API, and all admin/management routes — is forwarded to
the single manager process, which owns model lifecycle.

Each worker is an independent process with its own event loop and pooled client,
so throughput scales with --workers past the single-process ceiling.
"""

import copy
import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from grimoire import config
from grimoire.auth import require_api
from grimoire.proxy.client import init_proxy_client, close_proxy_client, get_proxy_client
from grimoire.proxy.llama import _backend_request_headers, _backend_response_headers
from grimoire.proxy.routes_table import RouteTableReader
from grimoire.registry import registry

logger = logging.getLogger(__name__)

MANAGER_URL = os.environ.get("GRIMOIRE_MANAGER_URL", "http://127.0.0.1:9000").rstrip("/")
# Stateless /v1/* suffixes the proxy workers serve directly; everything else
# (chat/completions, responses, models, props, ...) forwards to the manager.
STATELESS_SUFFIXES = {"embeddings", "rerank", "reranking"}

app = FastAPI(title="Grimoire Proxy", version="0.1.0")
_routes = RouteTableReader()
_rr: dict[str, int] = {}


@app.on_event("startup")
async def _startup():
    init_proxy_client()


@app.on_event("shutdown")
async def _shutdown():
    await close_proxy_client()


def _next_replica(model: str, replicas: list[dict]) -> dict:
    """Round-robin pick across a model's replica backends (per-worker counter)."""
    i = _rr.get(model, 0)
    _rr[model] = i + 1
    return replicas[i % len(replicas)]


async def _ensure_loaded(client: httpx.AsyncClient, model: str) -> list[dict]:
    """Ask the manager to load a cold model, then re-read the route table."""
    try:
        await client.post(f"{MANAGER_URL}/internal/ensure-loaded", json={"model": model}, timeout=600.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"manager unavailable: {e}")
    return _routes.replicas(model)


def _manager_headers(headers):
    """Headers to forward to the manager: keep Authorization (the manager
    re-authenticates), drop only hop-by-hop + host/content-length (httpx sets)."""
    drop = config.HOP_BY_HOP_HEADERS | {"host", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


async def _forward_to_manager(request: Request, path: str, body: bytes) -> StreamingResponse:
    """Proxy a request verbatim to the manager (chat, responses, admin, ...)."""
    client = get_proxy_client()
    req = client.build_request(
        request.method,
        f"{MANAGER_URL}/{path}",
        headers=_manager_headers(request.headers),
        params=request.query_params,
        content=body,
    )
    upstream = await client.send(req, stream=True)

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code,
                             headers=_backend_response_headers(upstream.headers))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_v1(request: Request, path: str):
    require_api(request)
    body = await request.body()

    suffix = path.split("/")[0]
    if suffix not in STATELESS_SUFFIXES:
        # chat/completions, responses, models, ... -> manager (single authority)
        return await _forward_to_manager(request, f"v1/{path}", body)

    # Stateless encoder path: round-robin across replica backends.
    payload = None
    if body and request.headers.get("content-type", "").split(";")[0] == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None

    model = registry.resolve(payload.get("model")) if isinstance(payload, dict) else None
    if not model:
        raise HTTPException(status_code=404, detail="No target model resolved")

    client = get_proxy_client()
    replicas = _routes.replicas(model)
    if not replicas:
        replicas = await _ensure_loaded(client, model)
    if not replicas:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not available")

    backend = _next_replica(model, replicas)
    headers = _backend_request_headers(request.headers)
    if isinstance(payload, dict):
        payload = copy.deepcopy(payload)
        payload["model"] = backend.get("backend_model_id") or model
        req = client.build_request("POST", f"http://127.0.0.1:{backend['port']}/v1/{path}",
                                   headers=headers, params=request.query_params, json=payload)
    else:
        req = client.build_request("POST", f"http://127.0.0.1:{backend['port']}/v1/{path}",
                                   headers=headers, params=request.query_params, content=body)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        logger.error(f"backend {backend['port']} unavailable: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body_iter(), status_code=upstream.status_code,
                             headers=_backend_response_headers(upstream.headers))


@app.get("/health")
async def health():
    return {"status": "healthy", "pid": os.getpid()}


# Catch-all for non-/v1 routes (props, models management UI, dashboard, ...) ->
# manager. Registered last so /v1 and /health take precedence.
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_rest(request: Request, path: str):
    body = await request.body()
    return await _forward_to_manager(request, path, body)
