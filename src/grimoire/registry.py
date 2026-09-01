"""Model registry - CRUD operations for models.json."""

import copy
import json
import logging
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from grimoire import config
from grimoire.chat_template import configured_reasoning_capability

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("GRIMOIRE_MODELS_DIR", "/models")
DEFAULT_REGISTRY_PATH = "/var/lib/grimoire/models.json"
DEFAULT_REGISTRY_SEED_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)
REGISTRY_SEED_PATH = os.environ.get("GRIMOIRE_REGISTRY_SEED_PATH", DEFAULT_REGISTRY_SEED_PATH)

BACKEND_LLAMA = "llama"

_GGUF_MAGIC = 0x46554747
_GGUF_SUPPORTED_VERSIONS = {2, 3}
_GGUF_FIXED_VALUE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}





def _get_backend(cfg: dict) -> str:
    """Get the backend type for a model config. Defaults to llama."""
    return cfg.get("backend", "llama")


def resolve_path(cfg: dict, key: str) -> Optional[str]:
    """Resolve a model config path (file, draft, drafter, mmproj, tokenizer).

    Absolute paths are returned as-is; relative paths are anchored at MODELS_DIR.
    """
    path = cfg.get(key)
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(MODELS_DIR, path)


def _looks_like_local_path(spec: str) -> bool:
    """True if *spec* should be treated as a filesystem path rather than an HF id.

    An ``hf:`` prefix explicitly marks the spec as a Hugging Face repo id.
    Without the prefix, any spec containing a path separator (``/``) or
    starting with ``./``, ``../``, or ``/`` is treated as a local path.
    Bare names without separators (e.g. ``gpt2``) are treated as HF ids.
    """
    if not isinstance(spec, str) or not spec:
        return False
    # Explicit ``hf:`` prefix overrides everything — definitely an HF repo id.
    if spec.startswith("hf:"):
        return False
    if os.path.isabs(spec):
        return True
    if spec.startswith("./") or spec.startswith("../"):
        return True
    # Any spec with a path separator is treated as a local path.
    # If a user wants to reference an HF repo id that happens to contain a
    # ``/`` (e.g. ``Qwen/Qwen3.6-27B``), they must use the ``hf:`` prefix.
    if os.sep in spec:
        return True
    # Bare name (no separators) — assume it's an HF repo id.
    return False

def _strip_hf_prefix(spec: str) -> str:
    """Strip the ``hf:`` prefix from *spec* if present."""
    if isinstance(spec, str) and spec.startswith("hf:"):
        return spec[3:]
    return spec


class ModelRegistry:
    """Model registry backed by JSON file.

    Schema:
    {
      "models": { "alias": { "file": "...", "ctx-size": 262144, ... } },
      "fixed": { "alias": 0 }
    }
    """

    def __init__(self, path=None, seed_path=None):
        self.path = path or REGISTRY_PATH
        self.seed_path = REGISTRY_SEED_PATH if seed_path is None else seed_path
        self._lock = RLock()
        self._stamp = None
        self._loaded_from = None
        self._data = {"models": {}, "fixed": {}}
        self.reload()

    @staticmethod
    def _normalize(data):
        if not isinstance(data, dict):
            data = {}
        models = data.get("models", {})
        fixed = data.get("fixed", {})
        family_defaults = data.get("family_defaults", {})
        if not isinstance(models, dict):
            models = {}
        if not isinstance(fixed, dict):
            fixed = {}
        if not isinstance(family_defaults, dict):
            family_defaults = {}
        return {**data, "models": models, "fixed": fixed, "family_defaults": family_defaults}

    @staticmethod
    def _file_stamp(path):
        """Return mtime_ns of a file, or None if it doesn't exist."""
        try:
            return os.stat(path).st_mtime_ns
        except FileNotFoundError:
            return None

    def _latest_stamp(self):
        """Return the highest mtime of runtime and seed paths."""
        rt = self._file_stamp(self.path)
        se = self._file_stamp(self.seed_path) if self.seed_path else None
        if rt is None and se is None:
            return None
        if rt is None:
            return se
        if se is None:
            return rt
        return max(rt, se)

    def _load(self):
        """Load from the newest source (runtime or seed)."""
        rt_stamp = self._file_stamp(self.path) or 0
        se_stamp = self._file_stamp(self.seed_path) or 0

        if rt_stamp == 0 and se_stamp == 0:
            return {"models": {}, "fixed": {}}

        if rt_stamp >= se_stamp:
            source = self.path
        else:
            source = self.seed_path

        try:
            with open(source) as f:
                data = self._normalize(json.load(f))

            if source == self.seed_path and os.path.exists(self.path):
                logger.info("Registry: seed (%s) is newer than runtime; loading from seed and syncing", self.seed_path)
                self._loaded_from = self.seed_path
                self._write_data(data)
            else:
                self._loaded_from = source

            return data
        except FileNotFoundError:
            return {"models": {}, "fixed": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry: {e}")
            return {"models": {}, "fixed": {}}

    def _maybe_reload(self):
        stamp = self._latest_stamp()
        if stamp != self._stamp:
            self._data = self._load()
            self._stamp = stamp

    def reload(self):
        """Reload the registry from disk and return a snapshot."""
        with self._lock:
            self._data = self._load()
            self._stamp = self._latest_stamp()
            return copy.deepcopy(self._data)

    def _write_data(self, data):
        """Write data dict to the runtime path atomically."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.path)
        self._loaded_from = self.path

    def _save(self):
        self._write_data(self._data)
        self._stamp = self._latest_stamp()
        logger.info(f"Registry saved to {self.path}")

    def snapshot(self):
        """Return a deep copy of the current registry data."""
        with self._lock:
            self._maybe_reload()
            return copy.deepcopy(self._data)

    def get(self, model_name):
        with self._lock:
            self._maybe_reload()
            cfg = self._data.get("models", {}).get(model_name)
            return copy.deepcopy(cfg) if cfg is not None else None

    def list_all(self):
        with self._lock:
            self._maybe_reload()
            return list(self._data.get("models", {}).keys())

    def list_fixed(self):
        with self._lock:
            self._maybe_reload()
            return dict(self._data.get("fixed", {}))

    def get_family_defaults(self, family):
        with self._lock:
            self._maybe_reload()
            family_defaults = self._data.get("family_defaults", {})
            if not isinstance(family_defaults, dict):
                return {}
            return copy.deepcopy(family_defaults.get(family, {}) or {})

    @staticmethod
    def normalize_model_id(model_id):
        """Normalize gateway aliases, backend IDs, paths, and GGUF names for matching."""
        if not model_id:
            return ""
        value = str(model_id).strip()
        value = Path(value).name
        value = re.sub(r"\.gguf$", "", value, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def resolve(self, model_id):
        """Resolve an external model ID to a registry alias using fuzzy core rules."""
        if not model_id:
            return None
        with self._lock:
            self._maybe_reload()
            models = self._data.get("models", {})

            if model_id in models:
                return model_id

            normalized = self.normalize_model_id(model_id)
            if not normalized:
                return None

            for name, cfg in models.items():
                candidates = [
                    name,
                    cfg.get("alias"),
                    cfg.get("file"),
                    Path(str(cfg.get("file", ""))).name,
                ]
                candidates.extend(cfg.get("aliases", []) or [])
                for candidate in candidates:
                    if candidate and self.normalize_model_id(candidate) == normalized:
                        return name

            return None

    def model_metadata(self, model_name):
        """Return public metadata for one registry model."""
        cfg = self.get(model_name)
        if not cfg:
            return None
        capabilities = cfg.get("capabilities", ["completion"])
        if isinstance(capabilities, (list, tuple, set)):
            capability_names = {
                value.lower()
                for value in capabilities
                if isinstance(value, str)
            }
        else:
            capability_names = set()
        family_defaults = self.get_family_defaults(cfg.get("family"))
        return {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "grimoire",
            "context": cfg.get("ctx-size"),
            "output": cfg.get("predict"),
            "family": cfg.get("family"),
            "capabilities": capabilities,
            "input_modalities": [
                "text",
                "image",
            ] if {"multimodal", "vision"} & capability_names else ["text"],
            "reasoning": configured_reasoning_capability(cfg, family_defaults),
            "cost": cfg.get("cost", {"input": 0, "output": 0}),
            "backend": _get_backend(cfg),
            "pinned_gpu": self.get_fixed_gpu(model_name),
            "gpu_ids": copy.deepcopy(cfg.get("gpu-ids")),
        }

    def list_metadata(self):
        """Return public metadata for all registry models."""
        return [self.model_metadata(name) for name in self.list_all()]

    def is_fixed(self, model_name):
        """Check if a model is pinned to a GPU."""
        with self._lock:
            self._maybe_reload()
            return model_name in self._data.get("fixed", {})

    def get_fixed_gpu(self, model_name):
        """Get the pinned GPU ID for a model, or None."""
        with self._lock:
            self._maybe_reload()
            return self._data.get("fixed", {}).get(model_name)

    def add(self, model_name, config):
        with self._lock:
            self._maybe_reload()
            if model_name in self._data.get("models", {}):
                raise ValueError(f"Model '{model_name}' already exists")
            self._data.setdefault("models", {})[model_name] = {
                **config,
                "added": datetime.now(timezone.utc).isoformat()
            }
            self._save()
            return copy.deepcopy(self._data["models"][model_name])

    def update(self, model_name, updates):
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            self._data["models"][model_name].update(updates)
            self._save()
            return copy.deepcopy(self._data["models"][model_name])

    def remove(self, model_name):
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            del self._data["models"][model_name]
            self._data.get("fixed", {}).pop(model_name, None)
            self._save()

    def pin_gpu(self, model_name, gpu_id):
        """Pin a model to a specific GPU."""
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("GPU ID must be a non-negative integer")
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            gpu_ids = self._data["models"][model_name].get("gpu-ids")
            if gpu_ids is not None:
                if (
                    not isinstance(gpu_ids, list)
                    or len(gpu_ids) < 2
                    or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids)
                    or len(set(gpu_ids)) != len(gpu_ids)
                ):
                    raise ValueError("Model has invalid 'gpu-ids'; expected at least two unique non-negative integers")
                if gpu_id != gpu_ids[0]:
                    raise ValueError(f"Pinned GPU must equal primary gpu-ids[0] ({gpu_ids[0]})")
            self._data.setdefault("fixed", {})[model_name] = gpu_id
            self._save()

    def unpin_gpu(self, model_name):
        """Remove GPU pinning for a model."""
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            self._data.get("fixed", {}).pop(model_name, None)
            self._save()

    def swap_fixed(self, new_fixed):
        """Atomically replace the fixed GPU pin map and return the old map."""
        with self._lock:
            self._maybe_reload()
            old = dict(self._data.get("fixed", {}))
            self._data["fixed"] = dict(new_fixed) if new_fixed else {}
            self._save()
            return old

    def validate(self, model_name, gpu_count=None):
        """Check if a model config is valid."""
        cfg = self.get(model_name)
        if not cfg:
            return False, f"Model '{model_name}' not found"

        backend = _get_backend(cfg)
        if backend == "llama":
            if not cfg.get("file"):
                return False, "Missing 'file' field"
            model_path = os.path.join(MODELS_DIR, cfg["file"])
            if not os.path.exists(model_path):
                return False, f"Model file not found at {model_path}"
        else:
            return False, f"Unknown backend '{backend}'"

        fixed_gpu = self.get_fixed_gpu(model_name)
        if fixed_gpu is not None:
            if isinstance(fixed_gpu, bool) or not isinstance(fixed_gpu, int) or fixed_gpu < 0:
                return False, f"Invalid pinned GPU ID for '{model_name}': {fixed_gpu}"
            if gpu_count is not None and fixed_gpu >= gpu_count:
                return False, f"Pinned GPU {fixed_gpu} is outside available range 0-{gpu_count - 1}"

        gpu_ids = cfg.get("gpu-ids")
        if gpu_ids is not None:
            if not isinstance(gpu_ids, list) or len(gpu_ids) < 2:
                return False, f"'gpu-ids' for '{model_name}' must contain at least two GPU IDs"
            if any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpu_ids):
                return False, f"'gpu-ids' for '{model_name}' must contain non-negative integers"
            if len(set(gpu_ids)) != len(gpu_ids):
                return False, f"'gpu-ids' for '{model_name}' must not contain duplicates"
            if gpu_count is not None and any(gpu >= gpu_count for gpu in gpu_ids):
                return False, f"'gpu-ids' for '{model_name}' contains a GPU outside range 0-{gpu_count - 1}"
            incompatible = []
            if cfg.get("cpu-only"):
                incompatible.append("cpu-only")
            if cfg.get("vram-budget-mib") is not None:
                incompatible.append("vram-budget-mib")
            if incompatible:
                return False, f"'gpu-ids' for '{model_name}' is incompatible with {', '.join(incompatible)}"
            if fixed_gpu is not None and fixed_gpu != gpu_ids[0]:
                return False, (
                    f"Pinned GPU {fixed_gpu} for '{model_name}' must equal primary "
                    f"gpu-ids[0] ({gpu_ids[0]})"
                )
            try:
                from grimoire.model_manager import GpuPlacement, validate_multi_gpu_selectors
                validate_multi_gpu_selectors(
                    cfg,
                    GpuPlacement(tuple(gpu_ids)),
                    family_defaults_getter=self.get_family_defaults,
                )
            except ValueError as exc:
                return False, str(exc)

        # A malformed vram-budget-mib (string/float/<=0) would otherwise be
        # silently coerced to None by the allocator and revert the model to
        # exclusive one-per-GPU — surface it as a config error instead.
        budget = cfg.get("vram-budget-mib")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
            return False, f"'vram-budget-mib' for '{model_name}' must be a positive integer, got {budget!r}"

        return True, "OK"


registry = ModelRegistry()
