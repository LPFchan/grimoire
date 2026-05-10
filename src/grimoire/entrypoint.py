#!/usr/bin/env python3
"""Grimoire entrypoint - handles model selection, gateway startup, and lifecycle."""

import argparse
import asyncio
import copy
import ctypes
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import signal
import subprocess
import urllib.parse
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from grimoire.history import history_store, identity_hash
from grimoire.ingest import download_model_file, model_filename_from_url
from grimoire.plugins import plugin_manager
from grimoire.registry import MODELS_DIR, registry
from grimoire.telemetry import telemetry_sampler, telemetry_store
from grimoire.usage import usage_store

logger = logging.getLogger(__name__)

LLAMA_SERVER_BIN = "/opt/model-a-llama-cpp/bin/llama-server"
DEFAULT_CTX_SIZE = 131072
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_PREDICT = 16384
API_KEY = os.environ.get("GRIMOIRE_API_KEY", "")
ADMIN_TOKEN = os.environ.get("GRIMOIRE_ADMIN_TOKEN") or API_KEY
COOKIE_NAME = "gw_session"


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer for {name}: {value}")
        return default


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


DEFAULT_STARTUP_TIMEOUT = _env_int("GRIMOIRE_STARTUP_TIMEOUT", 600)
MAX_HISTORY_CAPTURE_BYTES = _env_int("GRIMOIRE_HISTORY_CAPTURE_BYTES", 2 * 1024 * 1024)
MAX_USAGE_CAPTURE_BYTES = _env_int("GRIMOIRE_USAGE_CAPTURE_BYTES", 1024 * 1024)
LEGACY_STATS_PATH = os.environ.get("GRIMOIRE_LEGACY_STATS_PATH", "/var/lib/grimoire/token-stats.json")
ALLOW_ANONYMOUS = _env_bool("GRIMOIRE_ALLOW_ANONYMOUS", False)
WEBUI_DIR = os.environ.get("GRIMOIRE_WEBUI_DIR", "/opt/grimoire-webui")

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SENSITIVE_PROXY_HEADERS = {
    "authorization",
    "cookie",
    "x-grimoire-token",
    "x-api-key",
}

PR_SET_PDEATHSIG = 1


def _spawn_child_preexec():
    """Detach into a new session so killpg works, then ask the kernel to SIGTERM
    the child if grimoire dies — prevents orphan llama-server processes from
    holding GPU VRAM after a gateway crash."""
    os.setsid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


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


def _valid_cookie(request):
    token = request.cookies.get(COOKIE_NAME, "")
    return bool(API_KEY and token and hmac.compare_digest(token, API_KEY))


def require_api(request):
    """Require the shared API key for public API and history endpoints."""
    if not API_KEY:
        if not ALLOW_ANONYMOUS:
            raise HTTPException(status_code=503, detail="GRIMOIRE_API_KEY is required")
        return "anonymous", identity_hash("anonymous")
    token = _request_token(request)
    if token and hmac.compare_digest(token, API_KEY):
        return token, identity_hash(token)
    if _valid_cookie(request):
        return API_KEY, identity_hash(API_KEY)
    raise HTTPException(status_code=401, detail="Invalid API token")


def require_admin(request):
    """Require the shared admin token for mutating management endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="GRIMOIRE_ADMIN_TOKEN is required for management endpoints",
        )
    token = _request_token(request)
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        cookie = request.cookies.get(COOKIE_NAME, "")
        if cookie and hmac.compare_digest(cookie, ADMIN_TOKEN):
            return cookie, identity_hash(cookie)
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return token, identity_hash(token)


def _require_login_enabled():
    if not API_KEY and not ALLOW_ANONYMOUS:
        raise HTTPException(status_code=503, detail="GRIMOIRE_API_KEY is required")


def _resolve_config_path(path, base_dir=MODELS_DIR):
    if not path:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _extend_optional_arg(cmd, cfg, key, flag=None):
    value = cfg.get(key)
    if value is not None:
        cmd.extend([flag or f"--{key}", str(value)])


def _backend_request_headers(headers):
    """Return request headers safe to forward to an unauthenticated backend."""
    clean = {}
    blocked = HOP_BY_HOP_HEADERS | SENSITIVE_PROXY_HEADERS
    for key, value in headers.items():
        if key.lower() in blocked:
            continue
        clean[key] = value
    return clean


def _backend_response_headers(headers):
    clean = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        clean[key] = value
    return clean


def _cost_by_model():
    data = registry.snapshot()
    return {
        name: cfg.get("cost", {})
        for name, cfg in data.get("models", {}).items()
        if isinstance(cfg, dict)
    }


def build_cmd(cfg, port):
    """Build llama-server command from model config."""
    model_path = _resolve_config_path(cfg["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(cfg.get("ctx-size", DEFAULT_CTX_SIZE)),
        "--n-gpu-layers", str(cfg.get("n-gpu-layers", DEFAULT_N_GPU_LAYERS)),
        "--parallel", str(cfg.get("parallel", 1)),
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
        mmproj_path = _resolve_config_path(cfg["mmproj"])
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found at {mmproj_path}")
        cmd.extend(["--mmproj", mmproj_path])

    if cfg.get("chat-template-file"):
        template_path = _resolve_config_path(cfg["chat-template-file"], base_dir="/")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Chat template file not found at {template_path}")
        cmd.extend(["--chat-template-file", template_path])

    _extend_optional_arg(cmd, cfg, "image-min-tokens")
    _extend_optional_arg(cmd, cfg, "image-max-tokens")

    for bias in cfg.get("logit-bias", []) or []:
        cmd.extend(["--logit-bias", str(bias)])

    for arg in cfg.get("extra-args", []) or []:
        cmd.append(str(arg))

    return cmd


MODEL_STATUS_UNLOADED = "unloaded"
MODEL_STATUS_LOADING = "loading"
MODEL_STATUS_LOADED = "loaded"
MODEL_STATUS_FAILED = "failed"


class ActiveModel:
    """Manage a running llama-server process."""

    def __init__(self, name, cfg, port, gpu):
        self.name = name
        self.cfg = cfg
        self.port = port
        self.gpu = gpu
        self.process = None
        self.started = datetime.now(timezone.utc)
        self.backend_model_id = None
        self.status = MODEL_STATUS_LOADING

    def start(self):
        """Start the llama-server process."""
        cmd = build_cmd(self.cfg, self.port)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        logger.info(f"Starting {self.name} on GPU {self.gpu}, port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, env=env, preexec_fn=_spawn_child_preexec)
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

    async def get_backend_model_id(self):
        """Resolve the backend llama-server model ID for core alias rewriting."""
        if self.backend_model_id:
            return self.backend_model_id
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://127.0.0.1:{self.port}/v1/models")
            data = response.json()
            items = data.get("data") or data.get("models") or []
            if items:
                first = items[0]
                if isinstance(first, dict):
                    self.backend_model_id = first.get("id") or first.get("model") or first.get("name")
                elif isinstance(first, str):
                    self.backend_model_id = first
        except Exception as e:
            logger.info(f"Could not resolve backend model id for {self.name}: {e}")
        return self.backend_model_id or self.name

    def stop(self):
        """Stop the llama-server process."""
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
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
        resolved_name = registry.resolve(model_name)
        if not resolved_name:
            raise KeyError(f"Model '{model_name}' not found in registry")
        model_name = resolved_name

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
                    await asyncio.to_thread(active.stop)
                    del self.active[name]
            else:
                gpu = self._find_free_gpu()
                if gpu is None:
                    victim = self._find_oldest_evictable()
                    if not victim:
                        raise RuntimeError("All GPUs occupied by pinned models")
                    logger.info(f"Evicting {victim.name} from GPU {victim.gpu} (oldest load)")
                    await asyncio.to_thread(victim.stop)
                    del self.active[victim.name]
                    gpu = victim.gpu

            port = self._find_available_port(gpu)
            active = ActiveModel(model_name, cfg, port, gpu)
            self.active[model_name] = active
            active.start()
            try:
                startup_timeout = cfg.get("startup-timeout", DEFAULT_STARTUP_TIMEOUT)
                try:
                    startup_timeout = float(startup_timeout)
                except (TypeError, ValueError):
                    startup_timeout = DEFAULT_STARTUP_TIMEOUT
                await active.wait_ready(timeout=startup_timeout)
            except Exception:
                active.status = MODEL_STATUS_FAILED
                await asyncio.to_thread(active.stop)
                self.active.pop(model_name, None)
                raise

            active.status = MODEL_STATUS_LOADED
            logger.info(f"Started {model_name} on GPU {gpu}, port {port}")
            return active

    def get_status(self, model_name):
        """Return router-mode status for a registry entry."""
        active = self.active.get(model_name)
        if not active:
            return MODEL_STATUS_UNLOADED
        return active.status

    async def stop_model(self, model_name):
        """Stop an active model."""
        model_name = registry.resolve(model_name) or model_name
        async with self._lock:
            active = self.active.pop(model_name, None)
            if not active:
                return False
            await asyncio.to_thread(active.stop)
            logger.info(f"Stopped {model_name}")
            return True

    def get_active(self, model_name):
        """Get active model info."""
        model_name = registry.resolve(model_name) or model_name
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
                await asyncio.to_thread(active.stop)
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
    imported = usage_store.import_legacy_token_stats(
        LEGACY_STATS_PATH,
        identity_hash(API_KEY or "anonymous"),
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

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grimoire Login</title><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#101014;color:#f6f3ea;font-family:system-ui,sans-serif}
form{display:grid;gap:14px;width:min(360px,calc(100vw - 32px));padding:28px;border:1px solid #2f2d3a;border-radius:18px;background:#191821}
input,button{font:inherit;border-radius:10px;padding:11px 13px}input{border:1px solid #403d4d;background:#111018;color:#fff}button{border:0;background:#e89b41;color:#15100a;font-weight:700;cursor:pointer}.err{color:#ff8c8c}
</style></head><body><form method="post" action="/login"><h1>Grimoire</h1><input name="key" type="password" placeholder="API key" autofocus><button>Login</button>{error}</form></body></html>"""


def _render_login_html(error=""):
    return LOGIN_HTML.replace("{error}", error)

@app.get("/login")
async def login_page():
    if not API_KEY and not ALLOW_ANONYMOUS:
        return HTMLResponse(
            _render_login_html('<p class="err">Set GRIMOIRE_API_KEY or GATEWAY_API_KEY before login.</p>'),
            status_code=503,
        )
    return HTMLResponse(_render_login_html(""))


WEBUI_LOCALSTORAGE_CONFIG_KEY = "LlamaCppWebui.config"

LOGIN_BRIDGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Grimoire</title></head><body>
<noscript>Logged in. <a href="/">Open chat</a>.</noscript>
<script>
try {{
  var k = "{storage_key}";
  var c = {{}};
  try {{ c = JSON.parse(localStorage.getItem(k) || "{{}}") || {{}}; }} catch (e) {{ c = {{}}; }}
  c.apiKey = {key_json};
  localStorage.setItem(k, JSON.stringify(c));
}} catch (e) {{}}
location.replace("/");
</script></body></html>"""


def _render_login_bridge_html(key):
    return LOGIN_BRIDGE_HTML.format(
        storage_key=WEBUI_LOCALSTORAGE_CONFIG_KEY,
        key_json=json.dumps(key),
    )


@app.post("/login")
async def login_submit(request: Request):
    _require_login_enabled()
    form = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    key = (form.get("key") or [""])[0]
    if API_KEY and not hmac.compare_digest(key, API_KEY):
        return HTMLResponse(_render_login_html('<p class="err">Invalid key</p>'), status_code=401)
    response = HTMLResponse(_render_login_bridge_html(key))
    response.set_cookie(COOKIE_NAME, key, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return response


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_models": manager.list_active()
    }


@app.get("/v1/models")
async def get_v1_models(request: Request):
    """Return all registry models in OpenAI-compatible + llama.cpp router shape."""
    require_api(request)
    data = registry.list_metadata()
    active_names = set(manager.list_active())
    for item in data:
        name = item["id"]
        cfg = registry.get(name) or {}
        item["active"] = name in active_names
        item["status"] = {"value": manager.get_status(name)}
        item["in_cache"] = True
        item["path"] = cfg.get("file", "")
        item["context_window"] = cfg.get("ctx-size", DEFAULT_CTX_SIZE)
    return {"object": "list", "data": data}


@app.get("/models")
async def get_models(request: Request):
    """Return registry and active model info."""
    require_api(request)
    return {
        "models": registry.list_all(),
        "metadata": registry.list_metadata(),
        "fixed": registry.list_fixed(),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count
    }


@app.get("/history")
async def list_history(request: Request):
    """List conversations for the authenticated API key (tree-aware shape)."""
    _, user_hash = require_api(request)
    conversations = history_store.list_conversations_tree(user_hash)
    return {"conversations": conversations}


@app.post("/history")
async def create_history(request: Request):
    """Create a conversation for the authenticated API key.

    Webui upsert path: pass {id, name, lastModified, currNode, ...}.
    Legacy gateway path: pass {title, model, messages: [...]}.
    """
    _, user_hash = require_api(request)
    data = await request.json()
    if data.get("id") or data.get("name") is not None or data.get("lastModified") is not None:
        try:
            return history_store.upsert_conversation_tree(user_hash, data)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
    return history_store.create_conversation(
        user_hash,
        title=data.get("title") or "New chat",
        model=data.get("model"),
        messages=data.get("messages") or [],
    )


@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str, request: Request):
    """Return one server-side conversation with tree-shaped messages."""
    _, user_hash = require_api(request)
    try:
        return history_store.get_conversation_tree(user_hash, conversation_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/history/{conversation_id}")
async def update_history(conversation_id: str, request: Request):
    """Replace metadata/messages for one server-side conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return history_store.replace_conversation(
            user_hash,
            conversation_id,
            title=data.get("title"),
            model=data.get("model"),
            messages=data.get("messages"),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.patch("/history/{conversation_id}")
async def patch_history(conversation_id: str, request: Request):
    """Partial-update conversation metadata (webui's updateConversation)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return history_store.patch_conversation_tree(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/history/{conversation_id}")
async def delete_history(conversation_id: str, request: Request):
    """Delete one conversation; pass ?with_forks=true to cascade through forks."""
    _, user_hash = require_api(request)
    with_forks = request.query_params.get("with_forks", "").lower() in {"1", "true", "yes", "on"}
    try:
        history_store.delete_conversation_with_options(user_hash, conversation_id, delete_with_forks=with_forks)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(status_code=204)


@app.patch("/history/messages/{message_id}")
async def patch_history_message_by_id(message_id: str, request: Request):
    """Webui's updateMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = history_store.find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    data = await request.json()
    history_store.update_message_tree(user_hash, conv_id, message_id, data)
    return Response(status_code=204)


@app.delete("/history/messages/{message_id}")
async def delete_history_message_by_id(message_id: str, request: Request):
    """Webui's deleteMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = history_store.find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    deleted = history_store.delete_message_tree(user_hash, conv_id, message_id, cascade=cascade)
    return {"deleted": deleted}


@app.post("/history/{conversation_id}/messages")
async def create_history_message(conversation_id: str, request: Request):
    """Create a message branch under parent_id and update the conversation's currNode."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return history_store.create_message_branch(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.patch("/history/{conversation_id}/messages/{message_id}")
async def patch_history_message(conversation_id: str, message_id: str, request: Request):
    """Partial-update a message (webui's updateMessage)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        history_store.update_message_tree(user_hash, conversation_id, message_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(status_code=204)


@app.delete("/history/{conversation_id}/messages/{message_id}")
async def delete_history_message(conversation_id: str, message_id: str, request: Request):
    """Delete a message; pass ?cascade=true to delete the whole subtree."""
    _, user_hash = require_api(request)
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    try:
        deleted = history_store.delete_message_tree(user_hash, conversation_id, message_id, cascade=cascade)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": deleted}


@app.post("/history/{conversation_id}/fork")
async def fork_history(conversation_id: str, request: Request):
    """Fork a conversation at a specific message into a new conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    at_message_id = data.get("at_message_id") or data.get("atMessageId")
    name = data.get("name") or "Forked chat"
    include_attachments = data.get("include_attachments", data.get("includeAttachments", True))
    if not at_message_id:
        raise HTTPException(status_code=400, detail="Missing 'at_message_id'")
    try:
        return history_store.fork_conversation(
            user_hash, conversation_id, at_message_id, name, include_attachments
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/history/import")
async def import_history(request: Request):
    """Bulk-import conversations in the webui's exported shape."""
    _, user_hash = require_api(request)
    data = await request.json()
    return history_store.import_conversations_tree(user_hash, data)


@app.get("/stats")
async def get_stats(request: Request):
    """Return per-key token and equivalent-cost usage totals."""
    _, user_hash = require_api(request)
    return usage_store.summary(user_hash=user_hash)


@app.get("/stats/global")
async def get_global_stats(request: Request):
    """Return global token and equivalent-cost usage totals."""
    require_admin(request)
    return usage_store.summary()


DASHBOARD_WINDOWS_S = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
}
DASHBOARD_BINS = 60


@app.get("/stats/dashboard")
async def get_dashboard_stats(request: Request):
    """Combined token/cost + system telemetry time series for the dashboard.

    Query params:
        window: one of "5m","15m","1h","6h","24h","7d","30d","all" (default "1h")
    """
    _, user_hash = require_api(request)
    window = (request.query_params.get("window") or "1h").lower()

    now_ts = datetime.now(timezone.utc).timestamp()
    if window in {"all", "lifetime"}:
        earliest = usage_store.earliest_event_ts(user_hash=user_hash)
        sample_earliest = telemetry_store.earliest_ts()
        candidates = [t for t in (earliest, sample_earliest) if t]
        ts_from = min(candidates) if candidates else now_ts - DASHBOARD_WINDOWS_S["1h"]
        if ts_from >= now_ts:
            ts_from = now_ts - DASHBOARD_WINDOWS_S["1h"]
        window_label = "all"
    else:
        seconds = DASHBOARD_WINDOWS_S.get(window)
        if seconds is None:
            raise HTTPException(status_code=400, detail=f"Unknown window: {window}")
        ts_from = now_ts - seconds
        window_label = window

    bins = DASHBOARD_BINS
    usage = usage_store.binned_window(user_hash, ts_from, now_ts, bins)
    summary = usage_store.summary(user_hash=user_hash)
    lifetime = summary.get("total", {})

    def _system(metric, gpu_index):
        return {
            "current": telemetry_store.latest(metric, gpu_index),
            "series": telemetry_store.binned_avg(metric, gpu_index, ts_from, now_ts, bins),
        }

    gpu_indexes = sorted({0, 1, *range(manager.gpu_count)})
    gpus = [
        {
            "index": idx,
            "temp": _system("gpu_temp", idx),
            "power": _system("gpu_power", idx),
        }
        for idx in gpu_indexes
    ]

    return {
        "window": window_label,
        "from": ts_from,
        "to": now_ts,
        "bins": bins,
        "tokens": {
            "input": {
                "current": usage["total_input_tokens"],
                "series": usage["input_tokens_series"],
            },
            "output": {
                "current": usage["total_output_tokens"],
                "series": usage["output_tokens_series"],
            },
        },
        "cost": {
            "total": usage["total_input_cost"] + usage["total_output_cost"],
            "input": usage["total_input_cost"],
            "output": usage["total_output_cost"],
            "lifetime": float(lifetime.get("total_cost") or 0.0),
            "series": [
                a + b
                for a, b in zip(usage["input_cost_series"], usage["output_cost_series"])
            ],
        },
        "gpus": gpus,
        "cpu": {"temp": _system("cpu_temp", -1)},
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


def _model_payload_name(payload):
    if not isinstance(payload, dict):
        return None
    name = payload.get("model")
    return name if isinstance(name, str) and name else None


@app.post("/models/load")
async def models_load(request: Request):
    """Router-mode alias of /switch/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await switch_model(name, request)


@app.post("/models/unload")
async def models_unload(request: Request):
    """Router-mode alias of /stop/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await stop_model_endpoint(name, request)


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
    try:
        history_store.get_conversation(user_hash, conversation_id)
    except KeyError:
        return None
    return conversation_id


def _extract_assistant_text(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    pieces = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in parsed.get("choices", []) or []:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content") or choice.get("text")
            if isinstance(content, str):
                pieces.append(content)

    if pieces:
        return "".join(pieces)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ""
    choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


def _usage_from_object(data):
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
    except (TypeError, ValueError):
        return None
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _extract_usage(raw_bytes):
    """Extract token usage from JSON or final SSE chunks."""
    text = raw_bytes.decode("utf-8", errors="ignore")

    try:
        parsed = json.loads(text)
        usage = _usage_from_object(parsed)
        if usage:
            return usage
    except json.JSONDecodeError:
        pass

    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = _usage_from_object(parsed)
        if usage:
            found = usage
    return found


async def _record_response_stream(stream, user_hash, conversation_id, model_name, model_cfg, payload, record_history=True):
    captured = bytearray()
    usage_tail = bytearray()
    try:
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if record_history and conversation_id and isinstance(messages, list):
            message = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") != "assistant"), None)
            if message:
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

        assistant_text = _extract_assistant_text(raw)
        if record_history and assistant_text and conversation_id:
            try:
                history_store.append_message(user_hash, conversation_id, "assistant", assistant_text, model=model_name)
            except KeyError:
                pass


async def _proxy_chat(requested_model, payload, active, user_hash=None, conversation_id=None):
    """Proxy chat completions while keeping the upstream client open."""
    model_cfg = active.cfg
    payload = copy.deepcopy(payload)
    payload = plugin_manager.before_request(payload, active.name, model_cfg)
    backend_model_id = await active.get_backend_model_id()
    payload["model"] = backend_model_id
    url = f"http://127.0.0.1:{active.port}/v1/chat/completions"
    headers = {}

    client = httpx.AsyncClient(timeout=None)
    try:
        payload = await plugin_manager.before_backend_request(
            payload, active.name, model_cfg, backend_model_id, client, url, headers
        )
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
                    record_history=upstream.status_code < 400,
                )
            if non_streaming:
                body = b""
                async for chunk in stream:
                    body += chunk
                try:
                    data = json.loads(body)
                    if "choices" in data:
                        data["context_window"] = model_cfg.get("ctx-size", DEFAULT_CTX_SIZE)
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

    headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=headers)


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


DEFAULT_GENERATION_PARAMS = {
    "n_predict": DEFAULT_PREDICT,
    "seed": -1,
    "temperature": 0.8,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "typ_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "dry_sequence_breakers": [],
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "stop": [],
    "max_tokens": DEFAULT_PREDICT,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": False,
    "stream": True,
    "logit_bias": [],
    "n_probs": 0,
    "min_keep": 0,
    "grammar": "",
    "grammar_lazy": False,
    "grammar_triggers": [],
    "preserved_tokens": [],
    "chat_format": "",
    "reasoning_format": "auto",
    "reasoning_in_content": False,
    "generation_prompt": "",
    "samplers": [],
    "backend_sampling": False,
    "speculative.n_max": 16,
    "speculative.n_min": 0,
    "speculative.p_min": 0.75,
    "timings_per_token": False,
    "post_sampling_probs": False,
    "lora": [],
}


def _synthetic_props(model_name=None):
    cfg = registry.get(model_name) if model_name else None
    capabilities = (cfg or {}).get("capabilities", []) or []
    has_vision = "multimodal" in capabilities or "vision" in capabilities
    return {
        "default_generation_settings": {
            "id": 0,
            "id_task": -1,
            "n_ctx": (cfg or {}).get("ctx-size", DEFAULT_CTX_SIZE),
            "speculative": False,
            "is_processing": False,
            "params": dict(DEFAULT_GENERATION_PARAMS),
            "prompt": "",
            "next_token": {
                "has_next_token": False,
                "has_new_line": False,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": "",
            },
        },
        "total_slots": (cfg or {}).get("parallel", 1),
        "model_path": (cfg or {}).get("file", ""),
        "role": "router",
        "modalities": {"vision": bool(has_vision), "audio": False},
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": "grimoire",
    }


@app.get("/props")
async def props(request: Request):
    """Router-mode /props for the stock llama.cpp webui.

    Without ?model=<id> returns server-wide router props.
    With ?model=<id>&autoload=false returns synthetic per-model props from registry.
    With ?model=<id> (autoload not false) starts the model and proxies its real /props.
    """
    require_api(request)
    model_name = request.query_params.get("model")
    autoload = request.query_params.get("autoload", "true").lower() not in {"false", "0", "no", "off"}

    if not model_name:
        return _synthetic_props()

    resolved = registry.resolve(model_name)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not in registry")

    if not autoload and not manager.get_active(resolved):
        return _synthetic_props(resolved)

    try:
        active = await manager.start_model(resolved)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {resolved} for /props: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"http://127.0.0.1:{active.port}/props")
        if resp.status_code == 200:
            data = resp.json()
            data["role"] = "router"
            return data
    except Exception as e:
        logger.info(f"Falling back to synthetic /props for {resolved}: {e}")
    return _synthetic_props(resolved)


@app.get("/status")
async def status(request: Request):
    """Return system status."""
    require_api(request)
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
