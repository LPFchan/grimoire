#!/usr/bin/env python3
"""Grimoire entrypoint - handles model selection, gateway startup, and lifecycle."""

import argparse
import asyncio
import copy
import ctypes
from collections import OrderedDict
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import signal
import subprocess
import time
import urllib.parse
import uuid
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
QWEN_PROMPT_BLOCK_CACHE_SIZE = max(0, _env_int("GRIMOIRE_QWEN_PROMPT_BLOCK_CACHE_SIZE", 2048))
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


def build_cmd(cfg, port, alias=None):
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

    effective_alias = alias or cfg.get("alias")
    if effective_alias:
        cmd.extend(["--alias", effective_alias])

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
    """Manage a running model backend process (llama-server or dflash daemon)."""

    def __init__(self, name, cfg, port, gpu):
        self.name = name
        self.cfg = cfg
        self.port = port
        self.gpu = gpu
        self.process = None
        self.started = datetime.now(timezone.utc)
        self.backend_model_id = None
        self.status = MODEL_STATUS_LOADING
        self.backend_type = cfg.get("backend", BACKEND_LLAMA)

        # DFlash-specific state
        self.dflash_daemon: Optional[DflashDaemon] = None
        self.prefix_cache: Optional[PrefixCache] = None
        self.prefill_config: Optional[PrefillConfig] = None
        self.session_kv: Optional[SessionKV] = None
        self.snapshot_swap: Optional[SnapshotSwap] = None
        self._tokenizer = None
        self._qwen_prompt_block_cache = OrderedDict()
        # Serializes generate() calls against the single daemon stdin/stdout
        # pair. Created lazily so the unit-test path that constructs an
        # ActiveModel outside an event loop doesn't crash.
        self._dflash_lock: Optional[asyncio.Lock] = None

    def start(self):
        """Start the backend process."""
        if self.backend_type == BACKEND_DFLASH:
            self._start_dflash()
        else:
            self._start_llama()

    def _start_llama(self):
        """Start the llama-server process."""
        cmd = build_cmd(self.cfg, self.port, alias=self.name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        logger.info(f"Starting {self.name} (llama) on GPU {self.gpu}, port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, env=env, preexec_fn=_spawn_child_preexec)
        return self.process

    def _start_dflash(self):
        """Start the dflash daemon process."""
        target_path = resolve_path(self.cfg, "target")
        draft_path = resolve_path(self.cfg, "draft")
        drafter_path = resolve_path(self.cfg, "drafter")

        pc_cap = max(0, min(int(self.cfg.get("prefix-cache-slots", 4)), 8))
        self.prefix_cache = PrefixCache(
            cap=pc_cap,
            cache_dir=f"/var/lib/grimoire/prefix_cache/{self.name}",
            kv_k_type=self.cfg.get("cache-type-k", "q8_0"),
            kv_v_type=self.cfg.get("cache-type-v", "q8_0"),
            fa_window=self.cfg.get("fa-window", 2048),
        )
        self.prefix_cache.load()

        self.prefill_config = PrefillConfig(
            enabled=self.cfg.get("prefill-compression", "auto") != "off",
            threshold=self.cfg.get("prefill-threshold", 32000),
            keep_ratio=self.cfg.get("prefill-keep-ratio", 0.05),
            drafter_path=drafter_path,
            tail_budget=self.cfg.get("prefill-tail-budget", 12288),
        )

        session_cap = max(0, int(self.cfg.get("session-kv-slots", 2)))
        self.session_kv = SessionKV(
            cap=session_cap,
            prefix_cap=pc_cap,
        )

        max_session_vram = max(0, 8 - pc_cap)
        swap_cap = min(self.cfg.get("swap-max-vram", session_cap), session_cap, max_session_vram)
        self.snapshot_swap = SnapshotSwap(
            swap_dir=f"/var/lib/grimoire/snapshot_swap/{self.name}",
            max_vram_slots=swap_cap,
            slot_offset=pc_cap,
            slot_count=max_session_vram,
        )

        self.dflash_daemon = DflashDaemon(
            target_path=target_path,
            draft_path=draft_path,
            max_ctx=self.cfg.get("ctx-size", 16384),
            budget=self.cfg.get("budget", 22),
            gpu_id=self.gpu,
            prefill_threshold=self.prefill_config.threshold,
            prefill_keep_ratio=self.prefill_config.keep_ratio,
            kv_k_type=self.cfg.get("cache-type-k", "q8_0"),
            kv_v_type=self.cfg.get("cache-type-v", "q8_0"),
            fa_window=self.cfg.get("fa-window", 2048),
        )
        self.dflash_daemon.spawn(timeout=self.cfg.get("startup-timeout", DEFAULT_STARTUP_TIMEOUT))
        self.process = self.dflash_daemon.proc

    def dflash_lock(self) -> asyncio.Lock:
        """Return the per-model lock that serializes daemon I/O."""
        if self._dflash_lock is None:
            self._dflash_lock = asyncio.Lock()
        return self._dflash_lock

    async def wait_ready(self, timeout=DEFAULT_STARTUP_TIMEOUT):
        """Wait until the backend is ready."""
        if self.backend_type == BACKEND_DFLASH:
            # DFlash health check via VRAM (done in spawn)
            if self.dflash_daemon and self.dflash_daemon.is_running():
                return
            raise RuntimeError(f"{self.name} dflash daemon not running")

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
        """Resolve the backend model ID for core alias rewriting."""
        if self.backend_type == BACKEND_DFLASH:
            return self.name
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
        """Stop the backend process."""
        if self.backend_type == BACKEND_DFLASH and self.dflash_daemon:
            self._stop_dflash()
        else:
            self._stop_llama()

    def _stop_llama(self):
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

    def _stop_dflash(self):
        """Stop the dflash daemon and save prefix cache."""
        if self.prefix_cache:
            self.prefix_cache.save()
            self.prefix_cache.cleanup(self.dflash_daemon)
        if self.dflash_daemon:
            self.dflash_daemon.stop()
        self.process = None
        logger.info(f"Stopped {self.name}")

    def is_running(self):
        """Check if the process is running."""
        if self.backend_type == BACKEND_DFLASH and self.dflash_daemon:
            return self.dflash_daemon.is_running()
        return self.process is not None and self.process.poll() is None

    def get_tokenizer(self):
        """Get or load the tokenizer for this model.

        Reads `tokenizer` from the model config. Values containing a path
        separator (or starting with `.`) are treated as local paths under
        MODELS_DIR; everything else is loaded as a Hugging Face repo id.
        Raises RuntimeError if no tokenizer is configured or loading fails.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        spec = self.cfg.get("tokenizer")
        if not spec:
            raise RuntimeError(
                f"Model '{self.name}' has no 'tokenizer' configured; "
                "dflash models require an explicit tokenizer (HF repo id or local path)"
            )
        from transformers import AutoTokenizer
        source = resolve_path(self.cfg, "tokenizer") if _looks_like_local_path(spec) else spec
        trust_remote = bool(self.cfg.get("tokenizer-trust-remote-code", False))
        self._tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote)
        return self._tokenizer


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

            is_dflash = cfg.get("backend") == BACKEND_DFLASH
            port = None if is_dflash else self._find_available_port(gpu)
            active = ActiveModel(model_name, cfg, port, gpu)
            self.active[model_name] = active
            await asyncio.to_thread(active.start)
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
    return 0


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
        # llama backends carry `file`; dflash carries `target`.
        item["path"] = cfg.get("file") or cfg.get("target") or ""
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
    return {"deleted": conversation_id}


@app.patch("/history/messages/{message_id}")
async def patch_history_message_by_id(message_id: str, request: Request):
    """Webui's updateMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = history_store.find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    data = await request.json()
    history_store.update_message_tree(user_hash, conv_id, message_id, data)
    return {"updated": message_id}


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
    return {"updated": message_id}


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
            "vram": _system("gpu_vram", idx),
            "tokens_per_sec": _system("gpu_tokens_per_sec", idx),
        }
        for idx in gpu_indexes
    ]

    def _cumulative(series):
        running = 0
        return [running := running + v for v in series]

    return {
        "window": window_label,
        "from": ts_from,
        "to": now_ts,
        "bins": bins,
        "tokens": {
            "input": {
                "current": usage["total_input_tokens"],
                "series": _cumulative(usage["input_tokens_series"]),
            },
            "output": {
                "current": usage["total_output_tokens"],
                "series": _cumulative(usage["output_tokens_series"]),
            },
        },
        "cost": {
            "total": usage["total_input_cost"] + usage["total_output_cost"],
            "input": usage["total_input_cost"],
            "output": usage["total_output_cost"],
            "lifetime": float(lifetime.get("total_cost") or 0.0),
            "series": _cumulative([
                a + b
                for a, b in zip(usage["input_cost_series"], usage["output_cost_series"])
            ]),
        },
        "gpus": gpus,
        "cpu": {
            "temp": _system("cpu_temp", 0),
            "power": _system("cpu_power", 0),
        },
		"fans": {
			"fan1": _system("fan1_rpm", 0),
			"fan2": _system("fan2_rpm", 0),
		},
		"ram": {
			"system": _system("system_ram_mb", 0),
			"container": _system("container_ram_mb", 0),
		},
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
    if not history_store.conversation_exists(user_hash, conversation_id):
        return None
    return conversation_id


def _qwen_render_content(content, is_system_content=False):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        rendered = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Unexpected item type in content.")
            item_type = item.get("type")
            if "image" in item or "image_url" in item or item_type == "image":
                if is_system_content:
                    raise ValueError("System message cannot contain images.")
                rendered.append("<|vision_start|><|image_pad|><|vision_end|>")
                continue
            if "video" in item or item_type == "video":
                if is_system_content:
                    raise ValueError("System message cannot contain videos.")
                rendered.append("<|vision_start|><|video_pad|><|vision_end|>")
                continue
            if "text" in item:
                rendered.append(str(item.get("text", "")))
                continue
            raise ValueError("Unexpected item type in content.")
        return "".join(rendered)
    raise ValueError("Unexpected content type.")


def _qwen_last_query_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = _qwen_render_content(msg.get("content")).strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            return index
    raise ValueError("No user query found in messages.")


DFLASH_PROTECTED_TOOLS = {"obsidian_read-note"}


def _tool_name_from_message(msg, tool_call_names=None):
    if not isinstance(msg, dict):
        return None
    if msg.get("role") == "tool" and isinstance(msg.get("tool"), dict):
        return msg["tool"].get("name")
    if msg.get("type") == "tool" and isinstance(msg.get("tool"), str):
        return msg["tool"]
    if msg.get("role") == "tool" and tool_call_names and isinstance(msg.get("tool_call_id"), str):
        return tool_call_names.get(msg.get("tool_call_id"))
    return None


def _qwen_prompt_block_specs(messages, add_generation_prompt=False):
    if not messages:
        raise ValueError("No messages provided.")

    specs = []
    tool_call_names = {}
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = _qwen_render_content(messages[0].get("content"), is_system_content=True).strip()
        specs.append({
            "text": f"<|im_start|>system\n{content}<|im_end|>\n",
            "role": "system",
            "kind": "system",
            "message_start": 0,
            "message_end": 1,
            "protected": False,
            "metadata": {"message_index": 0},
        })

    last_query_index = _qwen_last_query_index(messages)
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise ValueError("Unexpected message role.")
        role = message.get("role")
        if role == "system":
            if index != 0:
                raise ValueError("System message must be at the beginning.")
            index += 1
            continue
        if role == "user":
            content = _qwen_render_content(message.get("content")).strip()
            specs.append({
                "text": f"<|im_start|>user\n{content}<|im_end|>\n",
                "role": "user",
                "kind": "user",
                "message_start": index,
                "message_end": index + 1,
                "protected": False,
                "metadata": {"message_index": index},
            })
            index += 1
            continue
        if role == "assistant":
            content = _qwen_render_content(message.get("content")).strip()
            reasoning_content = ""
            raw_reasoning = message.get("reasoning_content")
            if isinstance(raw_reasoning, str):
                reasoning_content = raw_reasoning
            elif "</think>" in content:
                reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = content.split("</think>")[-1].lstrip("\n")
            reasoning_content = reasoning_content.strip()
            if index > last_query_index:
                block = f"<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}"
            else:
                block = f"<|im_start|>assistant\n{content}"

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, dict):
                tool_calls = []
            tool_names = []
            for i, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                tc_id = tool_call.get("id")
                if isinstance(tc_id, str):
                    tool_call_names[tc_id] = name
                tool_names.append(name)
                if i == 0:
                    if content:
                        block += f"\n\n<tool_call>\n<function={name}>\n"
                    else:
                        block += f"<tool_call>\n<function={name}>\n"
                else:
                    block += f"\n<tool_call>\n<function={name}>\n"
                arguments = fn.get("arguments")
                if isinstance(arguments, dict):
                    for arg_name, arg_value in arguments.items():
                        block += f"<parameter={arg_name}>\n"
                        if isinstance(arg_value, str):
                            rendered_value = arg_value
                        else:
                            rendered_value = json.dumps(arg_value, ensure_ascii=False)
                        block += f"{rendered_value}\n</parameter>\n"
                block += "</function>\n</tool_call>"

            block += "<|im_end|>\n"
            specs.append({
                "text": block,
                "role": "assistant",
                "kind": "assistant",
                "message_start": index,
                "message_end": index + 1,
                "protected": False,
                "metadata": {
                    "message_index": index,
                    "reasoning": bool(reasoning_content),
                    "tool_names": tool_names,
                },
            })
            index += 1
            continue
        if role == "tool":
            parts = ["<|im_start|>user"]
            group_start = index
            tool_names = []
            protected = False
            while index < len(messages):
                tool_msg = messages[index]
                if not isinstance(tool_msg, dict) or tool_msg.get("role") != "tool":
                    break
                content = _qwen_render_content(tool_msg.get("content")).strip()
                parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
                tool_name = _tool_name_from_message(tool_msg, tool_call_names)
                if isinstance(tool_name, str):
                    tool_names.append(tool_name)
                    protected = protected or tool_name in DFLASH_PROTECTED_TOOLS
                index += 1
            parts.append("<|im_end|>\n")
            specs.append({
                "text": "".join(parts),
                "role": "tool",
                "kind": "tool_group",
                "message_start": group_start,
                "message_end": index,
                "protected": protected,
                "metadata": {
                    "message_indexes": list(range(group_start, index)),
                    "tool_names": tool_names,
                },
            })
            continue
        raise ValueError("Unexpected message role.")

    if add_generation_prompt:
        specs.append({
            "text": "<|im_start|>assistant\n<think>\n",
            "role": "assistant",
            "kind": "generation_prompt",
            "message_start": len(messages),
            "message_end": len(messages),
            "protected": True,
            "metadata": {"generation_prompt": True},
        })
    return specs


def _qwen_prompt_blocks(messages, add_generation_prompt=False):
    return [spec["text"] for spec in _qwen_prompt_block_specs(messages, add_generation_prompt=add_generation_prompt)]


def _prompt_block_cache_for(active):
    if active is None or QWEN_PROMPT_BLOCK_CACHE_SIZE <= 0:
        return None
    cache = getattr(active, "_qwen_prompt_block_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(active, "_qwen_prompt_block_cache", cache)
    return cache


def _tokenize_qwen_prompt_blocks(tokenizer, blocks, cache=None):
    encoded_blocks = []
    for block in blocks:
        block_ids = None
        if cache is not None:
            block_ids = cache.get(block)
            if block_ids is not None:
                cache.move_to_end(block)
        if block_ids is None:
            block_ids = tuple(tokenizer.encode(block, add_special_tokens=False))
            if cache is not None:
                cache[block] = block_ids
                cache.move_to_end(block)
                while len(cache) > QWEN_PROMPT_BLOCK_CACHE_SIZE:
                    cache.popitem(last=False)
        encoded_blocks.append(block_ids)
    return encoded_blocks


def _encode_qwen_prompt_blocks(tokenizer, blocks, cache=None):
    prompt_ids = []
    for block_ids in _tokenize_qwen_prompt_blocks(tokenizer, blocks, cache=cache):
        prompt_ids.extend(block_ids)
    return prompt_ids


def _generic_prompt_blocks(messages, tokenizer, prompt_ids, add_generation_prompt=False):
    if not messages:
        return []

    blocks = []
    prev = 0
    tool_call_names = {}
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = fn.get("name")
                if isinstance(tc_id, str) and isinstance(fn_name, str):
                    tool_call_names[tc_id] = fn_name
        rendered = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer.encode(rendered, add_special_tokens=False)
        end = len(encoded)
        if end <= prev or end > len(prompt_ids) or prompt_ids[:end] != encoded:
            raise ValueError(f"Unable to build prompt block span for message {index}")

        tool_name = _tool_name_from_message(msg, tool_call_names)
        role = str(msg.get("role") or "unknown")
        kind = role
        if role == "assistant" and msg.get("tool_calls"):
            kind = "assistant"
        if role == "tool":
            kind = "tool"

        metadata = {"message_index": index}
        if isinstance(tool_name, str):
            metadata["tool_name"] = tool_name

        blocks.append(
            PromptBlock(
                block_id=f"message:{index}",
                index=len(blocks),
                start=prev,
                end=end,
                role=role,
                kind=kind,
                message_start=index,
                message_end=index + 1,
                protected=tool_name in DFLASH_PROTECTED_TOOLS,
                metadata=metadata,
            )
        )
        prev = end

    if add_generation_prompt and prev < len(prompt_ids):
        blocks.append(
            PromptBlock(
                block_id="generation:0",
                index=len(blocks),
                start=prev,
                end=len(prompt_ids),
                role="assistant",
                kind="generation_prompt",
                message_start=len(messages),
                message_end=len(messages),
                protected=True,
                metadata={"generation_prompt": True},
            )
        )
        prev = len(prompt_ids)

    if prev != len(prompt_ids):
        raise ValueError("Prompt blocks did not cover the full prompt")
    return blocks


def _prompt_layout_from_messages(tokenizer, messages, add_generation_prompt=False, model_cfg=None, active=None):
    family = model_cfg.get("family") if isinstance(model_cfg, dict) else None
    if family == "qwen":
        specs = _qwen_prompt_block_specs(messages, add_generation_prompt=add_generation_prompt)
        block_texts = [spec["text"] for spec in specs]
        encoded_blocks = _tokenize_qwen_prompt_blocks(
            tokenizer,
            block_texts,
            cache=_prompt_block_cache_for(active),
        )
        prompt_ids = []
        prompt_blocks = []
        cursor = 0
        for index, (spec, block_ids) in enumerate(zip(specs, encoded_blocks)):
            start = cursor
            cursor += len(block_ids)
            prompt_ids.extend(block_ids)
            prompt_blocks.append(
                PromptBlock(
                    block_id=f"block:{index}",
                    index=index,
                    start=start,
                    end=cursor,
                    role=spec["role"],
                    kind=spec["kind"],
                    message_start=spec["message_start"],
                    message_end=spec["message_end"],
                    protected=bool(spec.get("protected")),
                    metadata=spec.get("metadata"),
                )
            )
        return prompt_ids, prompt_blocks

    prompt_ids = _prompt_ids_from_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        model_cfg=model_cfg,
        active=active,
    )
    return prompt_ids, _generic_prompt_blocks(
        messages,
        tokenizer,
        prompt_ids,
        add_generation_prompt=add_generation_prompt,
    )


def _prefix_cache_boundaries(blocks):
    return [
        block.end
        for block in blocks or []
        if block.message_end > block.message_start and block.kind != "generation_prompt" and block.end > 0
    ]


def _prompt_ids_from_messages(tokenizer, messages, add_generation_prompt=False, model_cfg=None, active=None):
    family = model_cfg.get("family") if isinstance(model_cfg, dict) else None
    qwen_error = None
    if family == "qwen":
        try:
            blocks = _qwen_prompt_blocks(messages, add_generation_prompt=add_generation_prompt)
            return _encode_qwen_prompt_blocks(
                tokenizer,
                blocks,
                cache=_prompt_block_cache_for(active),
            )
        except Exception as e:
            qwen_error = e

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        return prompt_ids
    except Exception as e:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return tokenizer.encode(prompt_text, add_special_tokens=False)
        except Exception:
            if qwen_error is not None:
                raise qwen_error
            raise e


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


def _extract_tokens_per_sec(raw_bytes):
    """Extract predicted_per_second from the last timing chunk in SSE data."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    best = None
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
        timings = parsed.get("timings") if isinstance(parsed, dict) else None
        if isinstance(timings, dict):
            tps = timings.get("predicted_per_second")
            if isinstance(tps, (int, float)) and tps > 0:
                best = float(tps)
    return best


def _extract_chunk_tokens_per_sec(chunk):
    """Like _extract_tokens_per_sec but scans a single raw chunk for live updating."""
    return _extract_tokens_per_sec(chunk)


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


async def _proxy_chat(requested_model, payload, active, user_hash=None, conversation_id=None):
    """Proxy chat completions while keeping the upstream client open."""
    model_cfg = active.cfg

    if active.backend_type == BACKEND_DFLASH:
        return await _proxy_dflash(requested_model, payload, active, user_hash, conversation_id)

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

    resp_headers = {"x-request-id": requested_model}
    content_type = upstream.headers.get("content-type")
    if content_type:
        resp_headers["content-type"] = content_type

    return StreamingResponse(body_iter(), status_code=upstream.status_code, headers=resp_headers)


DFLASH_SUPPORTED_SAMPLING = {"temperature", "top_p", "top_k", "seed"}
DFLASH_IGNORED_SAMPLING = {
    "min_p", "repetition_penalty", "frequency_penalty",
    "presence_penalty", "logit_bias", "typical_p", "tfs_z", "mirostat",
}


def _dflash_collect_stop_ids(tokenizer, payload_stop, cfg):
    """Build the daemon's stop-id set from EOS, chat-template ends, and user stop."""
    stop_ids = set()
    stop_seqs = []
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # Common chat-template assistant terminators across model families.
    for candidate in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end_of_text|>"):
        try:
            tok_id = tokenizer.convert_tokens_to_ids(candidate)
        except Exception:
            tok_id = None
        if isinstance(tok_id, int) and tok_id >= 0 and tok_id != tokenizer.unk_token_id:
            stop_ids.add(tok_id)

    # Operator-supplied stop strings on the model config.
    for s in cfg.get("stop-strings", []) or []:
        if not isinstance(s, str) or not s:
            continue
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])
        elif ids:
            stop_seqs.append(tuple(ids))

    # Request-level stop strings.
    raw_stops = [payload_stop] if isinstance(payload_stop, str) else (payload_stop or [])
    for s in raw_stops:
        if isinstance(s, str) and s:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                stop_ids.add(ids[0])
            elif ids:
                stop_seqs.append(tuple(ids))

    return stop_ids, stop_seqs


async def _proxy_dflash(requested_model, payload, active, user_hash, conversation_id):
    """Handle chat completions for the dflash backend.

    Streams via the daemon's stdin/stdout protocol while reusing the same
    history/telemetry/plugin pipeline as the llama path so dashboards work.
    """
    model_cfg = active.cfg
    payload = copy.deepcopy(payload)
    payload = plugin_manager.before_request(payload, active.name, model_cfg)

    messages = payload.get("messages", [])
    want_stream = payload.get("stream", True)
    max_tokens = int(payload.get("max_tokens", model_cfg.get("predict", DEFAULT_PREDICT)) or DEFAULT_PREDICT)
    temperature = payload.get("temperature", 0.8)
    top_p = payload.get("top_p", 0.9)
    top_k = payload.get("top_k", 40)
    seed = payload.get("seed")

    for name in DFLASH_IGNORED_SAMPLING:
        if payload.get(name) is not None:
            logger.warning(
                f"dflash: ignoring unsupported sampling param '{name}' on model {active.name}"
            )

    try:
        tokenizer = active.get_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {active.name}: {e}")
        raise HTTPException(status_code=503, detail=f"Tokenizer unavailable: {e}")

    try:
        prompt_ids, prompt_blocks = _prompt_layout_from_messages(
            tokenizer,
            messages,
            add_generation_prompt=True,
            model_cfg=model_cfg,
            active=active,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to render chat template: {e}")

    ctx_size = int(model_cfg.get("ctx-size", DEFAULT_CTX_SIZE))
    max_raw_context = model_cfg.get("max-raw-context", model_cfg.get("max_raw_context"))
    if max_raw_context is not None:
        max_raw_context = int(max_raw_context)
        if len(prompt_ids) > max_raw_context:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"raw prompt ({len(prompt_ids)} tokens) exceeds max raw context "
                    f"{max_raw_context}"
                ),
            )

    if len(prompt_ids) + max_tokens > ctx_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt ({len(prompt_ids)} tokens) + max_tokens ({max_tokens}) "
                f"exceeds context size {ctx_size}"
            ),
        )

    prefix_cache = active.prefix_cache
    prefill_config = active.prefill_config
    prefix_boundaries = _prefix_cache_boundaries(prompt_blocks)

    stop_ids, stop_seqs = _dflash_collect_stop_ids(tokenizer, payload.get("stop"), model_cfg)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async def sse_stream():
        daemon = active.dflash_daemon
        if daemon is None or not daemon.is_running():
            yield _sse_error_frames(completion_id, created, "dflash daemon not running")
            return

        async with active.dflash_lock():
            # Long-context compression
            effective_ids = prompt_ids
            effective_blocks = materialize_blocks(prompt_ids, prompt_blocks)
            if prefill_config and prefill_config.enabled:
                try:
                    effective_ids, _, effective_blocks = await maybe_compress(
                        prompt_ids,
                        daemon,
                        prefill_config,
                        blocks=prompt_blocks,
                    )
                except Exception as e:
                    logger.error(f"pflash compression failed: {e}")
                    effective_ids = prompt_ids
                    effective_blocks = materialize_blocks(prompt_ids, prompt_blocks)

            effective_boundaries = _prefix_cache_boundaries(effective_blocks)

            prefix_hit = None
            snap_slot = None
            snap_pos = None
            session_slot = None
            sk = active.session_kv
            swap = getattr(active, "snapshot_swap", None)
            session_key = sk.swap_key(conversation_id) if conversation_id and sk else None
            had_session = False
            # Session KV: check if this conversation has a cached snapshot.
            # If hit, RESTORE from that slot — only prefill the new delta.
            if conversation_id and sk:
                sess = sk.get_session(conversation_id, effective_ids)
                if sess is not None:
                    had_session = True
                    session_slot, session_prefix_len = sess
                    if swap and session_key is not None:
                        swap_hit = swap.get(session_key)
                        if swap_hit is None:
                            logger.warning("session snapshot missing from swap index; evicting session")
                            sk.evict(conversation_id)
                            session_slot = None
                            had_session = False
                        elif swap_hit[1]:
                            session_slot = swap_hit[0]
                            prefix_hit = (session_slot, session_prefix_len)
                        else:
                            session_slot = await asyncio.to_thread(
                                swap.reserve_slot, daemon, session_key
                            )
                            prefix_hit = (session_slot, session_prefix_len)
                    else:
                        prefix_hit = (session_slot, session_prefix_len)
                elif swap and session_key is not None:
                    await asyncio.to_thread(swap.discard, daemon, session_key)
            # Prefix cache: fallback for new conversations or cache-miss sessions.
            if prefix_hit is None:
                pc = prefix_cache
                if pc and not pc.disabled:
                    prefix_hit = pc.lookup(effective_ids, boundaries=effective_boundaries)
            # New conversation: reserve a session slot for an inline prompt snapshot.
            if conversation_id and sk and session_slot is None:
                if swap and session_key is not None:
                    evicted_id = sk.evict_lru_if_full(conversation_id)
                    if evicted_id is not None:
                        await asyncio.to_thread(
                            swap.discard, daemon, sk.swap_key(evicted_id)
                        )
                    session_slot = await asyncio.to_thread(
                        swap.reserve_slot, daemon, session_key
                    )
                else:
                    session_slot = sk.reserve_slot()

            if session_slot is not None:
                snap_slot, snap_pos = session_slot, len(effective_ids)
            else:
                pc = prefix_cache
                if pc and not pc.disabled:
                    prep = pc.prepare_inline_snap(effective_ids, effective_boundaries[0]) if effective_boundaries else None
                    if prep:
                        snap_slot, snap_pos = prep

            cmd_path = await asyncio.to_thread(
                daemon.send_generate_cmd,
                effective_ids,
                max_tokens,
                prefix_hit[0] if prefix_hit else None,
                snap_slot,
                snap_pos,
                temperature,
                top_p,
                top_k,
                seed,
            )

            decoded_prefix = ""
            tokens_emitted = []
            stop_seq_lens = sorted({len(seq) for seq in stop_seqs}, reverse=True)
            t0 = time.monotonic()
            try:
                index = 0
                while True:
                    tok = await asyncio.to_thread(daemon.read_next_token)
                    if tok is None:
                        break
                    if tok in stop_ids:
                        break
                    tokens_emitted.append(tok)
                    stop_hit = False
                    for seq_len in stop_seq_lens:
                        if len(tokens_emitted) < seq_len:
                            continue
                        if tuple(tokens_emitted[-seq_len:]) in stop_seqs:
                            del tokens_emitted[-seq_len:]
                            stop_hit = True
                            break
                    # Incremental decode against the running prefix avoids
                    # BPE/SentencePiece per-token artefacts.
                    new_full = tokenizer.decode(
                        tokens_emitted,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if len(new_full) > len(decoded_prefix):
                        delta = new_full[len(decoded_prefix):]
                        decoded_prefix = new_full
                        frame = _delta_sse(completion_id, created, delta, index)
                        index += 1
                        yield f"data: {json.dumps(frame)}\n\n".encode()
                    if stop_hit:
                        break
                    if len(tokens_emitted) >= max_tokens:
                        break
            finally:
                try:
                    os.unlink(cmd_path)
                except OSError:
                    pass

            elapsed = max(time.monotonic() - t0, 1e-6)
            tps = len(tokens_emitted) / elapsed

            # Session snapshots are captured inline at len(effective_ids)
            # during prefill, before decode advances cache.cur_pos.
            if session_slot is not None:
                if tokens_emitted or had_session:
                    try:
                        sk.update(conversation_id, session_slot, len(effective_ids), effective_ids)
                    except Exception as e:
                        logger.warning(f"session snapshot failed: {e}")
                        sk.evict(conversation_id)
                        if swap and session_key is not None:
                            await asyncio.to_thread(swap.discard, daemon, session_key)
                elif swap and session_key is not None:
                    await asyncio.to_thread(swap.discard, daemon, session_key)

            if snap_slot is not None and snap_pos is not None and snap_slot != session_slot:
                if tokens_emitted:
                    pc.confirm_inline_snap(snap_slot, snap_pos, effective_ids)
                else:
                    pc.abort_inline_snap(snap_slot)

            final = _final_sse(
                completion_id, created,
                len(effective_ids), len(tokens_emitted),
                decoded_prefix, ctx_size,
            )
            final["timings"] = {
                "predicted_n": len(tokens_emitted),
                "predicted_ms": elapsed * 1000.0,
                "predicted_per_second": tps,
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    async def safe_stream():
        try:
            async for chunk in sse_stream():
                yield chunk
        except Exception as e:
            logger.exception(f"dflash generation error: {e}")
            yield _sse_error_frames(completion_id, created, str(e))

    stream = safe_stream()
    stream = plugin_manager.wrap_response_stream(stream, active.name, model_cfg)
    if user_hash:
        stream = _record_response_stream(
            stream, user_hash, conversation_id, active.name,
            model_cfg, payload, gpu_index=active.gpu, record_history=True,
        )

    if want_stream:
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"x-request-id": requested_model},
        )

    body = bytearray()
    async for chunk in stream:
        body.extend(chunk)
    text = _extract_assistant_text(bytes(body))
    usage = _extract_usage(bytes(body)) or {"input_tokens": len(prompt_ids), "output_tokens": 0}
    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        },
        "context_window": ctx_size,
    })


def _sse_error_frames(completion_id, created, message):
    """SSE error payload plus the [DONE] terminator clients expect."""
    err = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "error": {"message": message, "type": "server_error"},
    }
    return (
        f"data: {json.dumps(err)}\n\n".encode()
        + b"data: [DONE]\n\n"
    )


def _delta_sse(completion_id, created, content, index=0):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "",
        "choices": [{
            "index": index,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }]
    }


def _final_sse(completion_id, created, prompt_tokens, completion_tokens, content, ctx_size):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "context_window": ctx_size,
    }


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
