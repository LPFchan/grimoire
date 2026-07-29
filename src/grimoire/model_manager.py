"""Model backend lifecycle management — ActiveModel and ModelManager."""

import asyncio
import ctypes
import logging
import math
import os
import signal
import subprocess
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from grimoire import config
from grimoire.dflash import PflashDaemon, PrefillConfig
from grimoire.registry import (
    MODELS_DIR,
    registry,
    resolve_path,
    _looks_like_local_path,
    _strip_hf_prefix,
    BACKEND_LLAMA,
)
from grimoire.proxy.routes_table import publish as _publish_route_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuPlacement:
    """Ordered physical CUDA devices assigned to one backend process."""

    device_ids: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "device_ids", tuple(self.device_ids))
        if not self.device_ids:
            raise ValueError("GPU placement must contain at least one device")
        if any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in self.device_ids):
            raise ValueError("GPU placement IDs must be non-negative integers")
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("GPU placement IDs must be unique")

    @property
    def primary(self):
        return self.device_ids[0]

    @property
    def is_multi(self):
        return len(self.device_ids) > 1


_SELECTOR_FLAGS = {
    "--split-mode": "split_mode",
    "-sm": "split_mode",
    "--tensor-split": "tensor_split",
    "-ts": "tensor_split",
    "--main-gpu": "main_gpu",
    "-mg": "main_gpu",
}
_UNSAFE_MULTI_DEVICE_FLAGS = {
    "--device",
    "-dev",
    "--device-draft",
    "-devd",
}
_SELECTOR_ENV_VARS = (
    "LLAMA_ARG_SPLIT_MODE",
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_MAIN_GPU",
)


def effective_extra_args(cfg, family_defaults_getter=None):
    """Return model and family extra arguments in command-line order."""
    args = [str(arg) for arg in (cfg.get("extra-args", []) or [])]
    family = cfg.get("family")
    if family:
        getter = family_defaults_getter or registry.get_family_defaults
        fd = getter(family)
        args.extend(str(arg) for arg in (fd.get("extra-args", []) or []))
    return args


def validate_multi_gpu_selectors(cfg, placement, family_defaults_getter=None):
    """Validate llama.cpp selectors against CUDA's placement-local numbering."""
    if not placement.is_multi:
        return

    args = effective_extra_args(cfg, family_defaults_getter=family_defaults_getter)
    selectors = {}
    index = 0
    while index < len(args):
        token = args[index]
        flag, separator, inline_value = token.partition("=")
        if flag in _UNSAFE_MULTI_DEVICE_FLAGS:
            raise ValueError(f"'{flag}' is not supported with 'gpu-ids'")
        selector = _SELECTOR_FLAGS.get(flag)
        if selector is None:
            index += 1
            continue
        if selector in selectors:
            raise ValueError(f"Duplicate or conflicting GPU selector '{flag}'")
        if separator:
            value = inline_value
        else:
            index += 1
            if index >= len(args) or args[index].startswith("-"):
                raise ValueError(f"GPU selector '{flag}' requires a value")
            value = args[index]
        if not value:
            raise ValueError(f"GPU selector '{flag}' requires a value")
        selectors[selector] = value
        index += 1

    split_mode = selectors.get("split_mode")
    if split_mode is not None:
        if split_mode not in {"layer", "row", "tensor"}:
            raise ValueError(f"Invalid multi-GPU split mode '{split_mode}'")

    tensor_split = selectors.get("tensor_split")
    if tensor_split is not None:
        pieces = tensor_split.split(",")
        if len(pieces) != len(placement.device_ids):
            raise ValueError(
                f"--tensor-split has {len(pieces)} values for "
                f"{len(placement.device_ids)} visible GPUs"
            )
        try:
            proportions = [float(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError("--tensor-split values must be numeric") from exc
        if any(not math.isfinite(value) or value < 0 for value in proportions) or not any(proportions):
            raise ValueError("--tensor-split values must be finite non-negative proportions with a positive total")

    main_gpu = selectors.get("main_gpu")
    if main_gpu is not None:
        try:
            logical_gpu = int(main_gpu)
        except ValueError as exc:
            raise ValueError("--main-gpu must be a logical GPU integer") from exc
        if str(logical_gpu) != main_gpu.strip() or not 0 <= logical_gpu < len(placement.device_ids):
            raise ValueError(
                f"--main-gpu {main_gpu!r} is outside logical visible-device range "
                f"0-{len(placement.device_ids) - 1}"
            )


def configure_gpu_environment(env, cfg, placement):
    """Apply validated ordered CUDA visibility for one placement."""
    validate_multi_gpu_selectors(cfg, placement)
    if placement.is_multi:
        for key in _SELECTOR_ENV_VARS:
            env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in placement.device_ids)
    return env


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

    max_slots = cfg.get("spec-dflash-max-slots")
    if max_slots is not None:
        cmd.extend(["--spec-dflash-max-slots", str(max_slots)])

    draft_n_max = cfg.get("spec-draft-n-max")
    if draft_n_max is not None:
        cmd.extend(["--spec-draft-n-max", str(draft_n_max)])

    branch_budget = cfg.get("spec-branch-budget")
    if branch_budget is not None:
        cmd.extend(["--spec-branch-budget", str(branch_budget)])

    draft_topk = cfg.get("spec-draft-top-k")
    if draft_topk is not None:
        cmd.extend(["--spec-draft-top-k", str(draft_topk)])

    draft_temp = cfg.get("spec-draft-temp")
    if draft_temp is not None:
        cmd.extend(["--spec-draft-temp", str(draft_temp)])


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
    ]

    capabilities = cfg.get("capabilities", [])
    if "completion" in capabilities:
        cmd.append("--jinja")

    cmd.extend([
        "--flash-attn", "on" if not cfg.get("cpu-only") else "off",
        "--metrics",
        "--slot-save-path", "/dev/shm/grimoire-slots",
        "--predict", str(cfg.get("predict", config.DEFAULT_PREDICT)),
    ])

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

    spec_type = cfg.get("speculative-type")
    if spec_type in ("nextn", "mtp"):
        # TheTom's fork (and upstream #22673) use draft-mtp for --spec-type.
        # For nextn (Qwen), MTP layers are in the same GGUF — the server
        # auto-detects them, so we do NOT pass -md (which would load a full
        # second model copy and double VRAM).
        # For mtp (Gemma4), the assistant is a separate GGUF file passed via -md.
        cmd.extend(["--spec-type", "draft-mtp"])
        if spec_type == "mtp":
            mtp_head = _resolve_config_path(cfg.get("mtp-head"))
            if mtp_head and os.path.exists(mtp_head):
                cmd.extend(["-md", mtp_head])
        # Remove spec-type from extra-args to avoid duplication
        cfg["extra-args"] = [a for a in cfg.get("extra-args", []) if a != "--spec-type" and a not in ("nextn", "mtp", "dflash", "draft-mtp", "draft-nextn")]

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
        if isinstance(gpu, GpuPlacement):
            self.placement = gpu
        elif gpu is None:
            self.placement = None
        else:
            self.placement = GpuPlacement((gpu,))
        self.gpu = self.placement.primary if self.placement is not None else None
        self.gpus = list(self.placement.device_ids) if self.placement is not None else []
        self.process = None
        self.started = datetime.now(timezone.utc)
        self.backend_model_id = None
        self.status = config.MODEL_STATUS_LOADING
        self.backend_type = cfg.get("backend", BACKEND_LLAMA)

        self.pflash_daemon: Optional[PflashDaemon] = None
        self.prefill_config: Optional[PrefillConfig] = None
        self.kv_cache_store = None
        self.snapshot_staging_slot: int = 7
        self._tokenizer = None
        self._qwen_prompt_block_cache = OrderedDict()

    def start(self):
        """Start the llama-server process."""
        self._start_llama()

    def _start_llama(self):
        """Start the llama-server process."""
        cmd = build_cmd(self.cfg, self.port, alias=self.name)
        env = os.environ.copy()
        if self.placement is not None:
            configure_gpu_environment(env, self.cfg, self.placement)
            # Put turboquant ggml libraries first so llama-server doesn't
            # accidentally load dflash's (older) ggml-cuda which lacks the
            # turboquant-specific symbols (e.g. g_innerq_scale_inv_host).
            _prepend_library_paths(
                env,
                [config.TURBOQUANT_LIB_DIR, config.TURBOQUANT_LIB64_DIR],
                exclude_prefixes=(config.PFLASH_HOME,),
            )
        else:
            # CPU-only: hide GPUs entirely so llama-server's CUDA backend
            # doesn't try to init (and OOM) when VRAM is full.
            env["CUDA_VISIBLE_DEVICES"] = ""

        # LD_PRELOAD the park/unpark shim for park models
        if self.cfg.get("park-unpark"):
            existing_pre = env.get("LD_PRELOAD", "")
            shim_path = config.PFLASH_SHIM_PATH
            env["LD_PRELOAD"] = f"{shim_path}:" + existing_pre if existing_pre else shim_path
            env["PFLASH_SHIM_FIFO_BASE"] = f"/tmp/pflash_shim.{self.name}"
            logger.info(f"park-unpark enabled, LD_PRELOAD={env['LD_PRELOAD']}")

        # Auto-derive GGML_DFLASH_MAX_VERIFY_TOKENS from DFlash tree config
        # so the C++ binary's tape/hidden-GPU path isn't silently capped at 25.
        if self.cfg.get("speculative-type") == "dflash":
            n_max = self.cfg.get("spec-draft-n-max", 16)
            branch_budget = self.cfg.get("spec-branch-budget", 0)
            draft_topk = self.cfg.get("spec-draft-top-k", 1)
            tree_budget = min(n_max + branch_budget, n_max * max(1, draft_topk))
            verify_tokens = tree_budget + 4  # small headroom for root + bonus
            if "GGML_DFLASH_MAX_VERIFY_TOKENS" not in env:
                env["GGML_DFLASH_MAX_VERIFY_TOKENS"] = str(verify_tokens)
                logger.info(
                    "Auto-set GGML_DFLASH_MAX_VERIFY_TOKENS=%d "
                    "(n_max=%d, branch_budget=%d, topk=%d, tree_budget=%d)",
                    verify_tokens, n_max, branch_budget, draft_topk, tree_budget
                )

        device_label = "CPU" if self.placement is None else f"GPU(s) {','.join(map(str, self.gpus))}"
        logger.info(f"Starting {self.name} (llama) on {device_label}, port {self.port}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, env=env, preexec_fn=_spawn_child_preexec)
        return self.process

    def _park_llama(self):
        """Park llama-server GPU memory via shim FIFO (VMM unmap + host save)."""
        try:
            import os, select
            base = f"/tmp/pflash_shim.{self.name}"
            fd = os.open(f"{base}.ctl", os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"park\n")
            os.close(fd)
            with open(f"{base}.ack", "r") as f:
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
            base = f"/tmp/pflash_shim.{self.name}"
            fd = os.open(f"{base}.ctl", os.O_WRONLY | os.O_NONBLOCK)
            os.write(fd, b"unpark\n")
            os.close(fd)
            with open(f"{base}.ack", "r") as f:
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
            raise RuntimeError(f"pflash requested but no drafter configured for {self.name}")

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

    async def wait_ready(self, timeout=config.DEFAULT_STARTUP_TIMEOUT):
        """Wait until the backend is ready."""
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

    def is_running(self):
        """Check if the process is running."""
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
        self.preset_lock = None
        self.preset_allows_manual_control = False
        self.gpu_mask = None  # set[int] or None (None = all GPUs available)

    def _is_gpu_allowed(self, gpu_id):
        return self.gpu_mask is None or gpu_id in self.gpu_mask

    def _allowed_gpus(self):
        if self.gpu_mask is not None:
            return sorted(self.gpu_mask)
        return list(range(self.gpu_count))

    def _build_route_table(self):
        """Map each running model -> its replica backends for the proxy workers.

        A model's replicas are itself plus any running `replica_peers` (sibling
        data-parallel copies on other GPUs), so the proxy can round-robin a single
        model name across GPUs.
        """
        running = {name: a for name, a in self.active.items() if a.is_running()}

        def entry(name):
            a = running[name]
            return {"port": a.port, "backend_model_id": a.cfg.get("alias") or a.backend_model_id or name}

        models = {}
        for name in running:
            replicas = [entry(name)]
            for peer in (registry.get(name) or {}).get("replica_peers", []) or []:
                if peer != name and peer in running:
                    replicas.append(entry(peer))
            models[name] = {"status": "loaded", "replicas": replicas}
        return models

    def _publish_routes(self):
        """Publish the current route table for the stateless proxy workers."""
        try:
            _publish_route_table(self._build_route_table())
        except Exception as e:
            logger.warning(f"route table publish failed: {e}")

    def _incumbents_on(self, gpu):
        """All active models currently placed on a GPU."""
        return [
            (name, active)
            for name, active in self.active.items()
            if gpu in getattr(active, "gpus", [active.gpu])
        ]

    def _allocate_explicit(self, model_name, cfg):
        """Reserve every member of an explicit multi-GPU placement."""
        placement = GpuPlacement(tuple(cfg["gpu-ids"]))
        for gpu in placement.device_ids:
            if gpu >= self.gpu_count:
                raise RuntimeError(f"GPU {gpu} is outside available range")
            if not self._is_gpu_allowed(gpu):
                raise RuntimeError(f"GPU {gpu} is excluded by active GPU mask")

        victims = []
        seen = set()
        for gpu in placement.device_ids:
            for name, active in self._incumbents_on(gpu):
                identity = id(active)
                if identity in seen:
                    continue
                seen.add(identity)
                if registry.is_fixed(name):
                    raise RuntimeError(
                        f"Cannot evict pinned model '{name}' from GPU placement {list(placement.device_ids)}"
                    )
                victims.append((name, active))
        return placement, victims

    def _is_budgeted_incumbent(self, active):
        """A budgeted incumbent declared a footprint and shares spare VRAM; an
        unbudgeted incumbent owns its GPU exclusively."""
        return self._model_budget_mib(active.cfg) is not None

    @staticmethod
    def _model_budget_mib(cfg):
        """Return declared `vram-budget-mib`, or None for unbudgeted (exclusive) models.

        Budgeted models may co-locate on a GPU as long as free VRAM covers the
        budget; unbudgeted models keep the strict one-model-per-GPU behavior.
        """
        budget = cfg.get("vram-budget-mib")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
            return budget
        return None

    def _get_gpu_free_vram_mib(self, gpu_id):
        """Query nvidia-smi for free VRAM (MiB) on a GPU. Returns None on failure."""
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_id}",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    def _allocate_exclusive(self, pinned_gpu):
        """Allocation for an unbudgeted (exclusive) model.

        An exclusive model owns its GPU only with respect to other exclusive
        models; budgeted incumbents (e.g. the always-on small models) declared a
        footprint and are treated as non-blocking co-tenants. An exclusive model
        therefore co-locates on top of budgeted incumbents without evicting them,
        and never queries nvidia-smi (it declares no budget of its own).

        Returns (gpu_id, [(name, ActiveModel), ...] incumbents_to_evict).
        """
        if pinned_gpu is not None:
            evict = []
            for name, active in self._incumbents_on(pinned_gpu):
                if self._is_budgeted_incumbent(active):
                    continue  # budgeted co-tenant shares the GPU; never evicted here
                if registry.is_fixed(name):
                    raise RuntimeError(f"Cannot evict pinned model '{name}' from GPU {pinned_gpu}")
                evict.append((name, active))
            return GpuPlacement((pinned_gpu,)), evict

        # Unpinned tiering, safest-for-an-undeclared-footprint first:
        #   1. a truly-empty GPU;
        #   2. evict the oldest non-pinned exclusive incumbent and take its GPU
        #      wholesale — preserves exclusive-swap semantics so loading a second
        #      large chat model swaps it in rather than cramming it onto a small
        #      model's GPU and OOMing;
        #   3. co-locate on a GPU hosting only budgeted co-tenants — a genuine
        #      last resort, used only when there is no exclusive to evict.
        # Budgeted incumbents are never evicted by an exclusive model.
        empty, budgeted_only = [], []
        for gpu in self._allowed_gpus():
            incumbents = self._incumbents_on(gpu)
            if not incumbents:
                empty.append(gpu)
            elif all(self._is_budgeted_incumbent(a) for _, a in incumbents):
                budgeted_only.append(gpu)
        if empty:
            return GpuPlacement((empty[0],)), []

        victim = None
        for name, active in self.active.items():
            allowed_members = [
                gpu for gpu in getattr(active, "gpus", [active.gpu])
                if self._is_gpu_allowed(gpu)
            ]
            if not allowed_members:
                continue
            if self._is_budgeted_incumbent(active) or registry.is_fixed(name):
                continue
            if victim is None or active.started < victim[1].started:
                victim = (name, active, allowed_members[0])
        if victim is not None:
            return GpuPlacement((victim[2],)), [(victim[0], victim[1])]

        if budgeted_only:
            return GpuPlacement((budgeted_only[0],)), []
        raise RuntimeError("All GPUs occupied by pinned or budgeted models")

    async def _allocate_budgeted(self, pinned_gpu, budget):
        """VRAM-budget allocation (budgeted/colocatable models).

        Returns (gpu_id, incumbents_to_evict). Prefers a GPU where the model fits
        without disturbing anyone; only if none exists does it evict non-pinned
        incumbents (pinned incumbents are never evicted). A post-eviction VRAM
        re-check in start_model guards against eviction not freeing enough.
        """
        candidates = [pinned_gpu] if pinned_gpu is not None else self._allowed_gpus()
        if not candidates:
            raise RuntimeError("No GPUs available for allocation")

        last_error = None
        # Pass 1: co-locate without eviction if free VRAM already covers the budget.
        for gpu in candidates:
            sharded_incumbents = [
                name
                for name, active in self._incumbents_on(gpu)
                if len(getattr(active, "gpus", [active.gpu])) > 1
            ]
            if sharded_incumbents:
                last_error = f"GPU {gpu} is exclusively occupied by sharded model(s): {', '.join(sharded_incumbents)}"
                continue
            free = await asyncio.to_thread(self._get_gpu_free_vram_mib, gpu)
            if free is None:
                last_error = f"nvidia-smi query failed for GPU {gpu}"
                continue
            if free >= budget:
                return GpuPlacement((gpu,)), []
            last_error = f"GPU {gpu} free VRAM {free} MiB < budget {budget} MiB"

        # Pass 2: make room by evicting non-pinned incumbents (oldest first).
        # v1 evicts ALL non-pinned incumbents on the chosen GPU rather than the
        # minimum needed — the manager doesn't track per-model VRAM, so it relies
        # on the post-eviction free-VRAM re-check in start_model to confirm fit.
        for gpu in candidates:
            evictable = sorted(
                ((n, a) for n, a in self.active.items()
                 if gpu in getattr(a, "gpus", [a.gpu]) and not registry.is_fixed(n)),
                key=lambda item: item[1].started,
            )
            if evictable:
                return GpuPlacement((gpu,)), evictable

        raise RuntimeError(last_error or "No suitable GPU found")

    async def _allocate_gpu(self, model_name, cfg):
        """Allocate an ordered placement and return incumbents to evict."""
        pinned_gpu = registry.get_fixed_gpu(model_name)
        if pinned_gpu is not None and pinned_gpu >= self.gpu_count:
            raise RuntimeError(f"Pinned GPU {pinned_gpu} is outside available range")
        if pinned_gpu is not None and not self._is_gpu_allowed(pinned_gpu):
            raise RuntimeError(f"Pinned GPU {pinned_gpu} is excluded by active GPU mask")

        if cfg.get("gpu-ids") is not None:
            return self._allocate_explicit(model_name, cfg)

        budget = self._model_budget_mib(cfg)
        if budget is None:
            return self._allocate_exclusive(pinned_gpu)
        return await self._allocate_budgeted(pinned_gpu, budget)

    def _find_available_port(self, gpu_id):
        """Find an available port for a model on a given GPU."""
        return self._find_available_port_excluding(gpu_id)

    def _find_available_port_excluding(self, gpu_id, ignore_ports=None):
        """Find an available port for a model on a given GPU, excluding known victims."""
        ignored = {port for port in (ignore_ports or set()) if port is not None}
        port = 8001 + gpu_id * 10
        for _ in range(100):
            if port in ignored:
                return port
            if not any(m.port == port for m in self.active.values() if m.port not in ignored):
                return port
            port += 1
        raise RuntimeError("No available ports found")

    CPU_PORT_BASE = 8500

    def _find_cpu_port(self):
        used = {m.port for m in self.active.values() if m.is_running()}
        port = self.CPU_PORT_BASE
        while port in used:
            port += 1
        return port

    @staticmethod
    def _startup_timeout(cfg):
        startup_timeout = cfg.get("startup-timeout", config.DEFAULT_STARTUP_TIMEOUT)
        try:
            return float(startup_timeout)
        except (TypeError, ValueError):
            return float(config.DEFAULT_STARTUP_TIMEOUT)

    async def _start_active_model(self, active: ActiveModel):
        """Start an ActiveModel and wait until it is ready."""
        active.status = config.MODEL_STATUS_LOADING
        logger.warning(f"pflash: cfg.pflash={active.cfg.get('pflash')}")
        if active.cfg.get("pflash"):
            await asyncio.to_thread(active._start_pflash_daemon)
            logger.warning(f"pflash: daemon started for {active.name}")

        await asyncio.to_thread(active.start)
        await active.wait_ready(timeout=self._startup_timeout(active.cfg))
        active.status = config.MODEL_STATUS_LOADED

    async def _stop_active_model(self, name: str, active: ActiveModel, drop_on_success: bool = True):
        """Stop a tracked model, but retain failed cleanup in manager state."""
        try:
            await asyncio.to_thread(active.stop)
        except Exception as e:
            active.status = config.MODEL_STATUS_FAILED
            self.active[name] = active
            raise RuntimeError(f"failed to stop {name}: {e}") from e
        if drop_on_success and self.active.get(name) is active:
            self.active.pop(name, None)

    async def _restore_incumbents(self, incumbents):
        """Best-effort restart of incumbents evicted during failed replacement startup."""
        errors = []
        for name, incumbent in incumbents:
            try:
                await self._start_active_model(incumbent)
            except Exception as e:
                incumbent.status = config.MODEL_STATUS_FAILED
                try:
                    await self._stop_active_model(name, incumbent)
                    errors.append(f"{name}: {e}")
                except Exception as stop_error:
                    errors.append(f"{name}: {e}; {stop_error}")
        return errors

    async def start_model(self, model_name, _preset_bypass=False):
        """Start a model with GPU allocation priority: pinned, free, oldest eviction."""
        resolved_name = registry.resolve(model_name)
        if not resolved_name:
            raise KeyError(f"Model '{model_name}' not found in registry")
        model_name = resolved_name

        # Fast path: an already-running model is the common case for proxied
        # traffic. Return it without taking the global lock so high-RPS requests
        # don't serialize through one async critical section. Safe in asyncio:
        # no await between the dict read and is_running(), so it's atomic; the
        # rare not-running case falls through to the locked start path.
        existing = self.active.get(model_name)
        if existing is not None and existing.is_running():
            return existing

        async with self._lock:
            if self.preset_lock is not None and not _preset_bypass and not self.preset_allows_manual_control:
                raise RuntimeError(
                    f"Preset '{self.preset_lock}' is active. "
                    f"Deactivate the preset before manually starting models."
                )
            if model_name in self.active and self.active[model_name].is_running():
                return self.active[model_name]
            if model_name in self.active:
                del self.active[model_name]

            cfg = registry.get(model_name)
            if not cfg:
                raise KeyError(f"Model '{model_name}' not found in registry")

            valid, reason = registry.validate(model_name, gpu_count=self.gpu_count)
            if not valid:
                raise RuntimeError(reason)

            if cfg.get("cpu-only"):
                if model_name in self.active:
                    if self.active[model_name].is_running():
                        return self.active[model_name]
                    await asyncio.to_thread(self.active[model_name].stop)
                    del self.active[model_name]

                port = self._find_cpu_port()
                active = ActiveModel(model_name, cfg, port, gpu=None)
                self.active[model_name] = active
                await self._start_active_model(active)
                self._publish_routes()
                return active

            budget = self._model_budget_mib(cfg)
            placement, incumbents = await self._allocate_gpu(model_name, cfg)
            gpu = placement.primary

            ignored_ports = {incumbent.port for _, incumbent in incumbents if incumbent.port is not None}
            port = self._find_available_port_excluding(gpu, ignored_ports)
            active = ActiveModel(model_name, cfg, port, placement)

            stopped_incumbents = []
            try:
                for name, incumbent in incumbents:
                    logger.info(
                        f"Evicting {name} from GPU placement "
                        f"{getattr(incumbent, 'gpus', [incumbent.gpu])} for replacement model {model_name}"
                    )
                    await asyncio.to_thread(incumbent.stop)
                    stopped_incumbents.append((name, incumbent))
                # Budgeted co-location relies on freed VRAM; after evicting, let the
                # CUDA contexts tear down, then confirm the budget is actually met
                # before launching. (No eviction => pass 1 already verified the fit.)
                if budget is not None and incumbents:
                    await asyncio.sleep(0.5)
                    free = await asyncio.to_thread(self._get_gpu_free_vram_mib, gpu)
                    if free is None:
                        # Eviction already happened; a transient nvidia-smi failure
                        # shouldn't abort + roll back. Proceed and let the backend
                        # surface an OOM if the budget genuinely wasn't freed.
                        logger.warning(
                            f"Post-eviction VRAM query failed on GPU {gpu}; "
                            f"proceeding optimistically for {model_name}"
                        )
                    elif free < budget:
                        raise RuntimeError(
                            f"Post-eviction VRAM check failed on GPU {gpu}: "
                            f"free={free} MiB, budget={budget} MiB"
                        )
                self.active[model_name] = active
                await self._start_active_model(active)
            except Exception as e:
                active.status = config.MODEL_STATUS_FAILED
                try:
                    await self._stop_active_model(model_name, active)
                except Exception as stop_error:
                    raise RuntimeError(f"{e}; {stop_error}") from e
                rollback_errors = await self._restore_incumbents(stopped_incumbents)
                if rollback_errors:
                    raise RuntimeError(
                        f"{e}; rollback failed for evicted incumbents: {'; '.join(rollback_errors)}"
                    ) from e
                raise

            for name, incumbent in stopped_incumbents:
                if self.active.get(name) is incumbent:
                    self.active.pop(name, None)
            logger.info(f"Started {model_name} on GPU placement {active.gpus}, port {port}")
            self._publish_routes()
            return active

    def get_status(self, model_name):
        """Return router-mode status for a registry entry."""
        active = self.active.get(model_name)
        if not active:
            return config.MODEL_STATUS_UNLOADED
        return active.status

    async def stop_model(self, model_name, _preset_bypass=False):
        """Stop an active model."""
        model_name = registry.resolve(model_name) or model_name
        async with self._lock:
            if self.preset_lock is not None and not _preset_bypass and not self.preset_allows_manual_control:
                raise RuntimeError(
                    f"Preset '{self.preset_lock}' is active. "
                    f"Deactivate the preset before manually stopping models."
                )
            active = self.active.pop(model_name, None)
            if not active:
                return False
            await asyncio.to_thread(active.stop)
            logger.info(f"Stopped {model_name}")
            self._publish_routes()
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
            self._publish_routes()


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
