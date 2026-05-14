"""Model backend lifecycle management — ActiveModel and ModelManager."""

import asyncio
import ctypes
import logging
import os
import signal
import subprocess
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

import httpx

from grimoire import config
from grimoire.dflash import DflashDaemon, PflashDaemon, PrefixCache, PrefillConfig, SessionKV, SnapshotSwap
from grimoire.registry import (
    MODELS_DIR,
    registry,
    resolve_path,
    _looks_like_local_path,
    _strip_hf_prefix,
    BACKEND_LLAMA,
    BACKEND_DFLASH,
)

logger = logging.getLogger(__name__)


def _spawn_child_preexec():
    """Detach into a new session so killpg works, then ask the kernel to SIGTERM
    the child if grimoire dies — prevents orphan llama-server processes from
    holding GPU VRAM after a gateway crash."""
    os.setsid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(config.PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


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


def _append_native_dflash_args(cmd, cfg):
    if cfg.get("backend", BACKEND_LLAMA) != BACKEND_LLAMA:
        return
    if cfg.get("speculative-type") != "dflash":
        return

    draft_model = _resolve_config_path(cfg.get("spec-draft-model") or cfg.get("draft"))
    if not draft_model:
        raise FileNotFoundError("Native DFlash requires a GGUF draft model path")
    if not os.path.exists(draft_model):
        raise FileNotFoundError(f"Native DFlash draft model not found at {draft_model}")

    cmd.extend(["--spec-type", "dflash", "--spec-draft-model", draft_model])

    cross_ctx = cfg.get("spec-dflash-cross-ctx")
    if cross_ctx is not None:
        cmd.extend(["--spec-dflash-cross-ctx", str(cross_ctx)])


def _prepend_library_paths(env, paths, exclude_prefixes=()):
    existing = []
    for path in env.get("LD_LIBRARY_PATH", "").split(":"):
        if not path:
            continue
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in exclude_prefixes):
            continue
        existing.append(path)

    merged = []
    for path in [*(paths or []), *existing]:
        if path and path not in merged:
            merged.append(path)
    if merged:
        env["LD_LIBRARY_PATH"] = ":".join(merged)
    else:
        env.pop("LD_LIBRARY_PATH", None)


def build_cmd(cfg, port, alias=None):
    """Build llama-server command from model config."""
    model_path = _resolve_config_path(cfg["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    cmd = [
        config.LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(cfg.get("ctx-size", config.DEFAULT_CTX_SIZE)),
        "--n-gpu-layers", str(cfg.get("n-gpu-layers", config.DEFAULT_N_GPU_LAYERS)),
        "--parallel", str(cfg.get("parallel", 1)),
        "--jinja",
        "--flash-attn", "on",
        "--metrics",
        "--slot-save-path", "/dev/shm/grimoire-slots",
        "--predict", str(cfg.get("predict", config.DEFAULT_PREDICT)),
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
    _append_native_dflash_args(cmd, cfg)

    for bias in cfg.get("logit-bias", []) or []:
        cmd.extend(["--logit-bias", str(bias)])

    for arg in cfg.get("extra-args", []) or []:
        cmd.append(str(arg))

    family = cfg.get("family")
    if family:
        fd = registry.get_family_defaults(family)
        for arg in fd.get("extra-args", []) or []:
            cmd.append(str(arg))

    return cmd


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
        self.status = config.MODEL_STATUS_LOADING
        self.backend_type = cfg.get("backend", BACKEND_LLAMA)

        # DFlash-specific state
        self.dflash_daemon: Optional[DflashDaemon] = None
        self.pflash_daemon: Optional[PflashDaemon] = None
        self.prefix_cache: Optional[PrefixCache] = None
        self.prefill_config: Optional[PrefillConfig] = None
        self.session_kv: Optional[SessionKV] = None
        self.snapshot_swap: Optional[SnapshotSwap] = None
        self.snapshot_staging_slot: int = 7
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
        # Put turboquant ggml libraries first so llama-server doesn't
        # accidentally load dflash's (older) ggml-cuda which lacks the
        # turboquant-specific symbols (e.g. g_innerq_scale_inv_host).
        _prepend_library_paths(
            env,
            [config.TURBOQUANT_LIB_DIR, config.TURBOQUANT_LIB64_DIR],
            exclude_prefixes=(config.DFLASH_HOME,),
        )

        # LD_PRELOAD the park/unpark shim for park models
        if self.cfg.get("park-unpark"):
            existing_pre = env.get("LD_PRELOAD", "")
            shim_path = config.PFLASH_SHIM_PATH
            env["LD_PRELOAD"] = f"{shim_path}:" + existing_pre if existing_pre else shim_path
            logger.info(f"park-unpark enabled, LD_PRELOAD={env['LD_PRELOAD']}")

        logger.info(f"Starting {self.name} (llama) on GPU {self.gpu}, port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, env=env, preexec_fn=_spawn_child_preexec)
        return self.process

    def _park_llama(self):
        """Park llama-server GPU memory via shim FIFO (VMM unmap + host save)."""
        try:
            import os, select
            fd = os.open("/tmp/pflash_shim.ctl", os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"park\n")
            os.close(fd)
            with open("/tmp/pflash_shim.ack", "r") as f:
                poll = select.poll()
                poll.register(f, select.POLLIN)
                if poll.poll(30000):
                    resp = f.read().strip()
                    return resp == "ok"
            logger.warning("park: ack timeout")
            return False
        except Exception as e:
            logger.warning(f"park failed: {e}")
            return False

    def _unpark_llama(self):
        """Unpark llama-server GPU memory via shim FIFO (VMM remap + host restore)."""
        try:
            import os, select
            fd = os.open("/tmp/pflash_shim.ctl", os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"unpark\n")
            os.close(fd)
            with open("/tmp/pflash_shim.ack", "r") as f:
                poll = select.poll()
                poll.register(f, select.POLLIN)
                if poll.poll(30000):
                    resp = f.read().strip()
                    return resp == "ok"
            logger.warning("unpark: ack timeout")
            return False
        except Exception as e:
            logger.warning(f"unpark failed: {e}")
            return False

    def _start_pflash_daemon(self):
        """Start the PFlash compression daemon on the same GPU."""
        drafter_path = resolve_path(self.cfg, "drafter")
        if not drafter_path:
            logger.warning(f"pflash requested but no drafter configured for {self.name}")
            return

        # Pre-build PrefillConfig from model config
        self.prefill_config = PrefillConfig(
            enabled=True,
            threshold=int(self.cfg.get("prefill-threshold", 48000)),
            keep_ratio=float(self.cfg.get("prefill-keep-ratio", 0.05)),
            drafter_path=drafter_path,
            tail_budget=int(self.cfg.get("prefill-tail-budget", 16000)),
        )

        daemon = PflashDaemon(drafter_path=drafter_path, gpu_id=self.gpu)
        daemon.start()
        self.pflash_daemon = daemon
        logger.info(f"Started pflash daemon for {self.name} on GPU {self.gpu}")

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

        use_pflash = self.cfg.get("pflash", True)
        use_dflash = self.cfg.get("dflash", True)

        self.prefill_config = PrefillConfig(
            enabled=bool(use_pflash) and self.cfg.get("prefill-compression", "auto") != "never",
            threshold=self.cfg.get("prefill-threshold", 32000),
            keep_ratio=self.cfg.get("prefill-keep-ratio", 0.05),
            drafter_path=drafter_path,
            tail_budget=self.cfg.get("prefill-tail-budget", 12288),
        )

        snapshot_disk_dir = self.cfg.get(
            "snapshot-disk-dir",
            f"/var/lib/grimoire/snapshot_swap/{self.name}",
        )
        session_cap = max(0, int(self.cfg.get("session-kv-slots", 2)))
        self.session_kv = SessionKV(
            cap=session_cap,
            path=os.path.join(snapshot_disk_dir, "session-kv.json"),
        )

        self.snapshot_staging_slot = int(self.cfg.get("snapshot-staging-slot", 7))
        self.snapshot_swap = SnapshotSwap(
            ram_dir=self.cfg.get("snapshot-ram-dir", "/dev/shm/grimoire-snapshots"),
            disk_dir=snapshot_disk_dir,
            ram_budget_gb=self.cfg.get("snapshot-ram-budget-gb", 20.0),
            disk_budget_gb=self.cfg.get("snapshot-disk-budget-gb", 100.0),
            disk_ttl_hours=self.cfg.get("snapshot-disk-ttl-hours", 24.0),
        )

        self.dflash_daemon = DflashDaemon(
            target_path=target_path,
            draft_path=draft_path if use_dflash else None,
            max_ctx=self.cfg.get("ctx-size", 16384),
            budget=self.cfg.get("budget", 22),
            gpu_id=self.gpu,
            pflash=bool(use_pflash),
            dflash=bool(use_dflash),
            prefill_threshold=self.prefill_config.threshold,
            prefill_keep_ratio=self.prefill_config.keep_ratio,
            kv_k_type=self.cfg.get("cache-type-k", "q8_0"),
            kv_v_type=self.cfg.get("cache-type-v", "q8_0"),
            fa_window=self.cfg.get("fa-window", 2048),
        )
        self.dflash_daemon.spawn(timeout=self.cfg.get("startup-timeout", config.DEFAULT_STARTUP_TIMEOUT))
        self.process = self.dflash_daemon.proc

    def dflash_lock(self) -> asyncio.Lock:
        """Return the per-model lock that serializes daemon I/O."""
        if self._dflash_lock is None:
            self._dflash_lock = asyncio.Lock()
        return self._dflash_lock

    async def wait_ready(self, timeout=config.DEFAULT_STARTUP_TIMEOUT):
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
        self._stop_pflash_daemon()

    def _stop_pflash_daemon(self):
        if self.pflash_daemon:
            try:
                self.pflash_daemon.stop()
            except Exception:
                pass
            self.pflash_daemon = None
        self.prefill_config = None

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
        source = _strip_hf_prefix(resolve_path(self.cfg, "tokenizer") if _looks_like_local_path(spec) else spec)
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

            # Start pflash daemon BEFORE the backend to get contiguous VRAM
            logger.warning(f"pflash: cfg.pflash={cfg.get('pflash')}, is_dflash={is_dflash}")
            if cfg.get("pflash") and not is_dflash:
                try:
                    await asyncio.to_thread(active._start_pflash_daemon)
                    logger.warning(f"pflash: daemon started for {model_name}")
                except Exception as e:
                    logger.warning(f"pflash: daemon failed for {model_name}: {e}")

            await asyncio.to_thread(active.start)
            try:
                startup_timeout = cfg.get("startup-timeout", config.DEFAULT_STARTUP_TIMEOUT)
                try:
                    startup_timeout = float(startup_timeout)
                except (TypeError, ValueError):
                    startup_timeout = config.DEFAULT_STARTUP_TIMEOUT
                await active.wait_ready(timeout=startup_timeout)
            except Exception:
                active.status = config.MODEL_STATUS_FAILED
                await asyncio.to_thread(active.stop)
                self.active.pop(model_name, None)
                raise

            active.status = config.MODEL_STATUS_LOADED
            logger.info(f"Started {model_name} on GPU {gpu}, port {port}")
            return active

    def get_status(self, model_name):
        """Return router-mode status for a registry entry."""
        active = self.active.get(model_name)
        if not active:
            return config.MODEL_STATUS_UNLOADED
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
