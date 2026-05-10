#!/usr/bin/env python3
"""Grimoire entrypoint - handles model selection, gateway startup, and lifecycle."""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import httpx

from grimoire.registry import registry, MODELS_DIR, REGISTRY_PATH

logger = logging.getLogger(__name__)

LLAMA_SERVER_BIN = "/opt/model-a-llama-cpp/bin/llama-server"
DEFAULT_CTX_SIZE = 131072
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_PREDICT = 16384


def parse_args():
    parser = argparse.ArgumentParser(description="Grimoire multi-GPU inference server")
    parser.add_argument("--model", help="Model name to start (from registry)")
    parser.add_argument("--port", type=int, default=9001, help="Gateway port (default: 9001)")
    parser.add_argument("--host", default="0.0.0.0", help="Gateway host (default: 0.0.0.0)")
    return parser.parse_args()


def build_cmd(cfg, port):
    """Build llama-server command from model config."""
    model_path = os.path.join(MODELS_DIR, cfg["file"])
    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(cfg.get("ctx-size", DEFAULT_CTX_SIZE)),
        "--n-gpu-layers", str(cfg.get("n-gpu-layers", DEFAULT_N_GPU_LAYERS)),
        "--jinja",
        "--flash-attn", "on",
        "--metrics",
        "--predict", str(cfg.get("predict", DEFAULT_PREDICT)),
    ]

    if cfg.get("cache-type-k"):
        cmd.extend(["--cache-type-k", cfg["cache-type-k"]])
    if cfg.get("cache-type-v"):
        cmd.extend(["--cache-type-v", cfg["cache-type-v"]])

    if cfg.get("mmproj"):
        mmproj_path = os.path.join(MODELS_DIR, cfg["mmproj"])
        if os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])

    return cmd


class ActiveModel:
    """Manage a running llama-server process."""

    def __init__(self, name, cfg, port, gpu):
        self.name = name
        self.cfg = cfg
        self.port = port
        self.gpu = gpu
        self.process = None
        self.started = datetime.now(timezone.utc)

    def start(self):
        """Start the llama-server process."""
        cmd = build_cmd(self.cfg, self.port)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        logger.info(f"Starting {self.name} on GPU {self.gpu}, port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        return self.process

    def stop(self):
        """Stop the llama-server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            logger.info(f"Stopped {self.name}")
            self.process = None

    def is_running(self):
        """Check if the process is running."""
        return self.process is not None and self.process.poll() is None


class ModelManager:
    """Manage active models across multiple GPUs.

    GPU allocation priority:
    1. Pinned models use their assigned GPU
    2. Free GPUs are preferred
    3. If no free GPU, evict the oldest-loaded non-pinned model
    """

    def __init__(self, gpu_count=2):
        self.active = {}  # {model_name: ActiveModel}
        self.gpu_count = gpu_count

    def _find_free_gpu(self):
        """Find a GPU that has no active model."""
        used_gpus = {m.gpu for m in self.active.values()}
        for gpu in range(self.gpu_count):
            if gpu not in used_gpus:
                return gpu
        return None

    def _find_oldest_evictable(self):
        """Find the GPU with the oldest-loaded non-pinned model."""
        oldest = None
        oldest_time = None
        for name, active in self.active.items():
            if registry.is_fixed(name):
                continue  # Skip pinned models
            if oldest is None or active.started < oldest_time:
                oldest = name
                oldest_time = active.started
        if oldest:
            return self.active[oldest]
        return None

    def _find_available_port(self, gpu_id):
        """Find an available port for a model on a given GPU."""
        port = 8001 + gpu_id * 10
        for _ in range(100):
            if not any(m.port == port for m in self.active.values()):
                return port
            port += 1
        raise RuntimeError("No available ports found")

    async def start_model(self, model_name):
        """Start a model with GPU allocation priority: pinned → free → evict oldest."""
        cfg = registry.get(model_name)
        if not cfg:
            raise KeyError(f"Model '{model_name}' not found in registry")

        # Already active?
        if model_name in self.active:
            logger.info(f"{model_name} is already active")
            return self.active[model_name]

        # 1. Check if pinned
        pinned_gpu = registry.get_fixed_gpu(model_name)
        if pinned_gpu is not None:
            gpu = pinned_gpu
            # If another model is on this GPU, evict it (only non-pinned can be evicted)
            for name, active in list(self.active.items()):
                if active.gpu == gpu:
                    if registry.is_fixed(name):
                        raise RuntimeError(
                            f"Cannot evict pinned model '{name}' from GPU {gpu}"
                        )
                    logger.info(f"Evicting {name} from GPU {gpu} for pinned model {model_name}")
                    active.stop()
                    del self.active[name]
                    break
        else:
            # 2. Try to find a free GPU
            gpu = self._find_free_gpu()
            if gpu is None:
                # 3. No free GPU — evict oldest non-pinned model
                victim = self._find_oldest_evictable()
                if not victim:
                    raise RuntimeError("All GPUs occupied by pinned models")
                logger.info(f"Evicting {victim.name} from GPU {victim.gpu} (oldest load)")
                victim.stop()
                del self.active[victim.name]
                gpu = victim.gpu

        port = self._find_available_port(gpu)
        active = ActiveModel(model_name, cfg, port, gpu)
        active.start()
        self.active[model_name] = active
        logger.info(f"Started {model_name} on GPU {gpu}, port {port}")
        return active

    async def stop_model(self, model_name):
        """Stop an active model."""
        if model_name not in self.active:
            return False

        active = self.active[model_name]
        active.stop()
        del self.active[model_name]
        logger.info(f"Stopped {model_name}")
        return True

    def get_active(self, model_name):
        """Get active model info."""
        return self.active.get(model_name)

    def list_active(self):
        """List all active models."""
        return list(self.active.keys())

    async def shutdown(self):
        """Gracefully stop all active models."""
        for name, active in self.active.items():
            logger.info(f"Shutting down {name}")
            active.stop()
        self.active.clear()


def detect_gpu_count():
    """Detect number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except Exception:
        pass
    return 2  # Default


# Global model manager
manager = ModelManager(gpu_count=detect_gpu_count())

logger.info(f"Grimoire starting with {manager.gpu_count} GPU(s)")

# Create FastAPI gateway app
app = FastAPI(title="Grimoire Gateway", version="0.1.0")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_models": manager.list_active()
    }


@app.get("/v1/models")
async def get_v1_models():
    """Return active models in OpenAI-compatible format."""
    data = []
    for name in manager.list_active():
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
        "fixed": dict(registry._data.get("fixed", {})),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count
    }


@app.post("/switch/{model_name}")
async def switch_model(model_name: str):
    """Start a model with GPU allocation."""
    try:
        active = await manager.start_model(model_name)
        return {
            "status": "started",
            "model": model_name,
            "gpu": active.gpu,
            "port": active.port
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop/{model_name}")
async def stop_model_endpoint(model_name: str):
    """Stop an active model."""
    if model_name not in manager.active:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    await manager.stop_model(model_name)
    return {"status": "stopped", "model": model_name}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions to the correct active model."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    requested_model = payload.get("model")
    if not requested_model or requested_model not in manager.active:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{requested_model}' is not active. Use POST /switch/{{model_name}} to start it."
        )

    active = manager.get_active(requested_model)
    port = active.port

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
    import urllib.request
    data = await request.json()
    model_alias = data.get("alias")
    model_url = data.get("url")
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
        })
        logger.info(f"Added model {model_alias} to registry")
        return {"status": "added", "model": model_alias}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/status")
async def status():
    """Return system status."""
    active_info = {}
    for name, active in manager.active.items():
        active_info[name] = {
            "gpu": active.gpu,
            "port": active.port,
            "started": active.started.isoformat(),
            "pinned": registry.is_fixed(name)
        }
    return {
        "models": registry.list_all(),
        "fixed": dict(registry._data.get("fixed", {})),
        "active": active_info,
        "gpu_count": manager.gpu_count
    }


def main():
    args = parse_args()

    # Start initial model if specified
    if args.model:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.start_model(args.model))
        except Exception as e:
            logger.error(f"Failed to start initial model {args.model}: {e}")
            sys.exit(1)

    # Start uvicorn gateway
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
