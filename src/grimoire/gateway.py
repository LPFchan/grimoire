"""Grimoire gateway - FastAPI app for API proxy, model routing, and management."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from grimoire.registry import registry, MODELS_DIR, REGISTRY_PATH

logger = logging.getLogger(__name__)

app = FastAPI()

# Active models: {model_name: {"process": subprocess.Popen, "port": int, "gpu": int}}
active_models = {}

# Default llama-server settings
LLAMA_SERVER_BIN = "/opt/model-a-llama-cpp/bin/llama-server"
DEFAULT_CTX_SIZE = 131072
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_PREDICT = 16384
DEFAULT_PORT = 8001


def _build_cmd(model_cfg, port):
    """Build llama-server command from model config."""
    model_path = os.path.join(MODELS_DIR, model_cfg["file"])
    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(model_cfg.get("ctx-size", DEFAULT_CTX_SIZE)),
        "--n-gpu-layers", str(model_cfg.get("n-gpu-layers", DEFAULT_N_GPU_LAYERS)),
        "--jinja",
        "--flash-attn", "on",
        "--metrics",
        "--predict", str(model_cfg.get("predict", DEFAULT_PREDICT)),
    ]

    if model_cfg.get("cache-type-k"):
        cmd.extend(["--cache-type-k", model_cfg["cache-type-k"]])
    if model_cfg.get("cache-type-v"):
        cmd.extend(["--cache-type-v", model_cfg["cache-type-v"]])

    if model_cfg.get("mmproj"):
        mmproj_path = os.path.join(MODELS_DIR, model_cfg["mmproj"])
        if os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])

    return cmd


def _find_available_port(base_port, gpu_id):
    """Find an available port for a model on a given GPU."""
    port = base_port + gpu_id * 10
    for _ in range(100):
        # Check if port is in use by other active models
        if not any(m.get("port") == port for m in active_models.values()):
            return port
        port += 1
    raise RuntimeError("No available ports found")


async def start_model(model_name):
    """Start a model's llama-server process."""
    cfg = registry.get(model_name)
    if not cfg:
        raise KeyError(f"Model '{model_name}' not found in registry")

    gpu = cfg.get("gpu", 0)
    port = _find_available_port(DEFAULT_PORT, gpu)

    # Stop other models on same GPU
    for active_name, active_data in list(active_models.items()):
        if active_data.get("gpu") == gpu and active_name != model_name:
            logger.info(f"Stopping {active_name} to free GPU {gpu}")
            active_data["process"].terminate()
            try:
                active_data["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                active_data["process"].kill()
                active_data["process"].wait()
            del active_models[active_name]

    cmd = _build_cmd(cfg, port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info(f"Starting {model_name} on GPU {gpu}, port {port}")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    active_models[model_name] = {
        "process": process,
        "port": port,
        "gpu": gpu,
        "started": datetime.now(timezone.utc).isoformat()
    }
    return active_models[model_name]


async def stop_model(model_name):
    """Stop a model's llama-server process."""
    if model_name not in active_models:
        return False

    active_data = active_models[model_name]
    active_data["process"].terminate()
    try:
        active_data["process"].wait(timeout=30)
    except subprocess.TimeoutExpired:
        active_data["process"].kill()
        active_data["process"].wait()
    del active_models[model_name]
    logger.info(f"Stopped {model_name}")
    return True


@app.get("/v1/models")
async def get_v1_models():
    """Return active models in OpenAI-compatible format."""
    data = []
    for name in active_models.keys():
        data.append({
            "id": name,
            "object": "model",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "owned_by": "grimoire",
        })
    return {"object": "list", "data": data}


@app.get("/models")
async def get_models():
    """Return registry and active model info."""
    return {
        "models": registry.list_all(),
        "active": list(active_models.keys()),
        "gpu_count": len(set(active_models[m].get("gpu", 0) for m in active_models))
    }


@app.post("/switch/{model_name}")
async def switch_model(model_name: str):
    """Start a model (stops other models on same GPU)."""
    try:
        result = await start_model(model_name)
        return {"status": "started", "model": model_name, **result}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop/{model_name}")
async def stop_model_endpoint(model_name: str):
    """Stop an active model."""
    if model_name not in active_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    await stop_model(model_name)
    return {"status": "stopped", "model": model_name}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions to the correct active model."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    requested_model = payload.get("model")
    if not requested_model or requested_model not in active_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{requested_model}' is not active. Use /switch/{{model_name}} to start it."
        )

    active_data = active_models[requested_model]
    port = active_data["port"]

    # Forward to the active model's llama-server
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"http://localhost:{port}/v1/chat/completions",
                json=payload,
                timeout=None,
                stream=True
            )
            return StreamingResponse(
                resp.aiter_raw(),
                media_type="text/event-stream",
                headers={"x-request-id": requested_model}
            )
        except Exception as e:
            logger.error(f"Failed to forward request: {e}")
            raise HTTPException(status_code=502, detail="Model server unavailable")


@app.post("/ingest")
async def ingest_model(request: Request):
    """Download and register a new model."""
    data = await request.json()
    model_alias = data.get("alias")
    model_url = data.get("url")
    gpu = data.get("gpu", 0)
    ctx_size = data.get("ctx-size", DEFAULT_CTX_SIZE)

    if not model_alias or not model_url:
        raise HTTPException(status_code=400, detail="Missing 'alias' or 'url'")

    # Download model file
    model_filename = model_url.split("/")[-1]
    model_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail=f"Model file already exists at {model_path}")

    try:
        logger.info(f"Downloading model from {model_url} to {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

    # Add to registry
    try:
        registry.add(model_alias, {
            "file": f"gguf/{model_filename}",
            "mmproj": None,
            "ctx-size": ctx_size,
            "gpu": gpu,
            "has-multimodal": False,
        })
        logger.info(f"Added model {model_alias} to registry")
        return {"status": "added", "model": model_alias}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/status")
async def status():
    """Return system status."""
    return {
        "models": registry.list_all(),
        "active": list(active_models.keys()),
        "gpu_count": len(set(active_models[m].get("gpu", 0) for m in active_models))
    }


@app.on_event("shutdown")
def shutdown():
    """Gracefully stop all active models on shutdown."""
    for name, data in active_models.items():
        logger.info(f"Shutting down {name}")
        data["process"].terminate()
        try:
            data["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            data["process"].kill()
            data["process"].wait()
