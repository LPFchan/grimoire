#!/usr/bin/env python3
"""Grimoire entrypoint - handles model selection, gateway startup, and lifecycle."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
import subprocess
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from grimoire.ingest import download_model_file, model_filename_from_url
from grimoire.registry import MODELS_DIR, registry

logger = logging.getLogger(__name__)

LLAMA_SERVER_BIN = "/opt/model-a-llama-cpp/bin/llama-server"
DEFAULT_CTX_SIZE = 131072
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_PREDICT = 16384
ADMIN_TOKEN = os.environ.get("GRIMOIRE_ADMIN_TOKEN")


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer for {name}: {value}")
        return default


DEFAULT_STARTUP_TIMEOUT = _env_int("GRIMOIRE_STARTUP_TIMEOUT", 600)


def parse_args():
    parser = argparse.ArgumentParser(description="Grimoire multi-GPU inference server")
    parser.add_argument("--model", help="Model name to start (from registry)")
    parser.add_argument("--port", type=int, default=9001, help="Gateway port (default: 9001)")
    parser.add_argument("--host", default="0.0.0.0", help="Gateway host (default: 0.0.0.0)")
    return parser.parse_args()


def _request_token(request):
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-grimoire-token")


def require_admin(request):
    """Require the shared admin token for mutating management endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="GRIMOIRE_ADMIN_TOKEN is required for management endpoints",
        )
    token = _request_token(request)
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")


def build_cmd(cfg, port):
    """Build llama-server command from model config."""
    model_path = os.path.join(MODELS_DIR, cfg["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

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
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found at {mmproj_path}")
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

        self.process = subprocess.Popen(cmd, env=env, start_new_session=True)
        return self.process

    async def wait_ready(self, timeout=DEFAULT_STARTUP_TIMEOUT):
        """Wait until llama-server reports healthy or exits/fails."""
        deadline = asyncio.get_running_loop().time() + timeout
        url = f"http://127.0.0.1:{self.port}/health"
        last_error = None

        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if not self.is_running():
                    code = self.process.returncode if self.process else "unknown"
                    raise RuntimeError(f"{self.name} exited before becoming ready (code {code})")
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError as e:
                    last_error = e
                await asyncio.sleep(1)

        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(f"Timed out waiting for {self.name} on port {self.port}{detail}")

    def stop(self):
        """Stop the llama-server process."""
        if not self.process:
            return
        if self.process.poll() is None:
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
        self.active = {}
        self.gpu_count = gpu_count
        self._lock = asyncio.Lock()

    def _find_free_gpu(self):
        """Find a GPU that has no active model."""
        used_gpus = {m.gpu for m in self.active.values()}
        for gpu in range(self.gpu_count):
            if gpu not in used_gpus:
                return gpu
        return None

    def _find_oldest_evictable(self):
        """Find the oldest-loaded non-pinned active model."""
        oldest = None
        for name, active in self.active.items():
            if registry.is_fixed(name):
                continue
            if oldest is None or active.started < oldest.started:
                oldest = active
        return oldest

    def _find_available_port(self, gpu_id):
        """Find an available port for a model on a given GPU."""
        port = 8001 + gpu_id * 10
        for _ in range(100):
            if not any(m.port == port for m in self.active.values()):
                return port
            port += 1
        raise RuntimeError("No available ports found")

    async def start_model(self, model_name):
        """Start a model with GPU allocation priority: pinned, free, oldest eviction."""
        async with self._lock:
            if model_name in self.active and self.active[model_name].is_running():
                logger.info(f"{model_name} is already active")
                return self.active[model_name]
            if model_name in self.active:
                del self.active[model_name]

            cfg = registry.get(model_name)
            if not cfg:
                raise KeyError(f"Model '{model_name}' not found in registry")

            valid, reason = registry.validate(model_name, gpu_count=self.gpu_count)
            if not valid:
                raise RuntimeError(reason)

            pinned_gpu = registry.get_fixed_gpu(model_name)
            if pinned_gpu is not None:
                if pinned_gpu >= self.gpu_count:
                    raise RuntimeError(f"Pinned GPU {pinned_gpu} is outside available range")
                gpu = pinned_gpu
                for name, active in list(self.active.items()):
                    if active.gpu != gpu:
                        continue
                    if registry.is_fixed(name):
                        raise RuntimeError(f"Cannot evict pinned model '{name}' from GPU {gpu}")
                    logger.info(f"Evicting {name} from GPU {gpu} for pinned model {model_name}")
                    active.stop()
                    del self.active[name]
            else:
                gpu = self._find_free_gpu()
                if gpu is None:
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
            try:
                startup_timeout = cfg.get("startup-timeout", DEFAULT_STARTUP_TIMEOUT)
                try:
                    startup_timeout = float(startup_timeout)
                except (TypeError, ValueError):
                    startup_timeout = DEFAULT_STARTUP_TIMEOUT
                await active.wait_ready(timeout=startup_timeout)
            except Exception:
                active.stop()
                raise

            self.active[model_name] = active
            logger.info(f"Started {model_name} on GPU {gpu}, port {port}")
            return active

    async def stop_model(self, model_name):
        """Stop an active model."""
        async with self._lock:
            active = self.active.pop(model_name, None)
            if not active:
                return False
            active.stop()
            logger.info(f"Stopped {model_name}")
            return True

    def get_active(self, model_name):
        """Get active model info."""
        active = self.active.get(model_name)
        if active and active.is_running():
            return active
        return None

    def list_active(self):
        """List all running active models."""
        return [name for name, active in self.active.items() if active.is_running()]

    async def shutdown(self):
        """Gracefully stop all active models."""
        async with self._lock:
            for name, active in list(self.active.items()):
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
            gpus = [line for line in result.stdout.splitlines() if line.strip()]
            if gpus:
                return len(gpus)
    except Exception:
        pass
    return 2


manager = ModelManager(gpu_count=detect_gpu_count())
logger.info(f"Grimoire starting with {manager.gpu_count} GPU(s)")


@asynccontextmanager
async def lifespan(_app):
    initial_model = getattr(_app.state, "initial_model", None)
    if initial_model:
        await manager.start_model(initial_model)
    try:
        yield
    finally:
        await manager.shutdown()


app = FastAPI(title="Grimoire Gateway", version="0.1.0", lifespan=lifespan)


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
        active = manager.get_active(name)
        data.append({
            "id": name,
            "object": "model",
            "created": int(active.started.timestamp()) if active else 0,
            "owned_by": "grimoire",
        })
    return {"object": "list", "data": data}


@app.get("/models")
async def get_models():
    """Return registry and active model info."""
    return {
        "models": registry.list_all(),
        "fixed": registry.list_fixed(),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count
    }


@app.post("/switch/{model_name}")
async def switch_model(model_name: str, request: Request):
    """Start a model with GPU allocation."""
    require_admin(request)
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
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop/{model_name}")
async def stop_model_endpoint(model_name: str, request: Request):
    """Stop an active model."""
    require_admin(request)
    if not manager.get_active(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    await manager.stop_model(model_name)
    return {"status": "stopped", "model": model_name}


async def _proxy_chat(requested_model, payload, active):
    """Proxy chat completions while keeping the upstream client open."""
    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = await client.send(
            client.build_request(
                "POST",
                f"http://127.0.0.1:{active.port}/v1/chat/completions",
                json=payload,
            ),
            stream=True,
        )
    except Exception:
        await client.aclose()
        raise

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=headers)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions to the correct active model."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    requested_model = payload.get("model")
    active = manager.get_active(requested_model) if requested_model else None
    if not active:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{requested_model}' is not active. Use POST /switch/{{model_name}} to start it."
        )

    try:
        return await _proxy_chat(requested_model, payload, active)
    except Exception as e:
        logger.error(f"Failed to forward request: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")


@app.post("/ingest")
async def ingest_model(request: Request):
    """Download and register a new model."""
    require_admin(request)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    model_alias = data.get("alias")
    model_url = data.get("url")
    ctx_size = data.get("ctx-size", DEFAULT_CTX_SIZE)

    if not model_alias or not model_url:
        raise HTTPException(status_code=400, detail="Missing 'alias' or 'url'")

    try:
        model_filename = model_filename_from_url(model_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail=f"Model file already exists at {model_path}")

    try:
        logger.info(f"Downloading model from {model_url} to {model_path}")
        await asyncio.to_thread(download_model_file, model_url, model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

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
    for name in manager.list_active():
        active = manager.get_active(name)
        if not active:
            continue
        active_info[name] = {
            "gpu": active.gpu,
            "port": active.port,
            "started": active.started.isoformat(),
            "pinned": registry.is_fixed(name),
            "running": active.is_running(),
        }
    return {
        "models": registry.list_all(),
        "fixed": registry.list_fixed(),
        "active": active_info,
        "gpu_count": manager.gpu_count
    }


def main():
    args = parse_args()
    app.state.initial_model = args.model
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
