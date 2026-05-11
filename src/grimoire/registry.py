"""Model registry - CRUD operations for models.json."""

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("GRIMOIRE_MODELS_DIR", "/models")
DEFAULT_REGISTRY_PATH = "/var/lib/grimoire/models.json"
DEFAULT_REGISTRY_SEED_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)
REGISTRY_SEED_PATH = os.environ.get("GRIMOIRE_REGISTRY_SEED_PATH", DEFAULT_REGISTRY_SEED_PATH)

BACKEND_LLAMA = "llama"
BACKEND_DFLASH = "dflash"


def _get_backend(cfg: dict) -> str:
    """Get the backend type for a model config. Defaults to llama."""
    return cfg.get("backend", BACKEND_LLAMA)


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
    """True if `spec` should be treated as a filesystem path rather than an HF id.

    Hugging Face repo ids look like `org/repo` (single `/`, no leading dot, not
    absolute). Anything that's absolute, starts with `./` or `../`, or has more
    than one path separator is treated as a local path.
    """
    if not isinstance(spec, str) or not spec:
        return False
    if os.path.isabs(spec):
        return True
    if spec.startswith("./") or spec.startswith("../"):
        return True
    return spec.count(os.sep) >= 2


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
        self._data = {"models": {}, "fixed": {}}
        self.reload()

    @staticmethod
    def _normalize(data):
        if not isinstance(data, dict):
            data = {}
        models = data.get("models", {})
        fixed = data.get("fixed", {})
        if not isinstance(models, dict):
            models = {}
        if not isinstance(fixed, dict):
            fixed = {}
        return {**data, "models": models, "fixed": fixed}

    def _load(self):
        path = self.path
        if not os.path.exists(path) and self.seed_path and os.path.exists(self.seed_path):
            path = self.seed_path
        try:
            with open(path) as f:
                return self._normalize(json.load(f))
        except FileNotFoundError:
            return {"models": {}, "fixed": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry: {e}")
            return {"models": {}, "fixed": {}}

    def _stat_stamp(self):
        try:
            return os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            return None

    def _maybe_reload(self):
        stamp = self._stat_stamp()
        if stamp != self._stamp:
            self._data = self._load()
            self._stamp = stamp

    def reload(self):
        """Reload the registry from disk and return a snapshot."""
        with self._lock:
            self._data = self._load()
            self._stamp = self._stat_stamp()
            return copy.deepcopy(self._data)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.path)
        self._stamp = self._stat_stamp()
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

            matches = []
            for name, cfg in models.items():
                candidates = [name, cfg.get("file"), Path(str(cfg.get("file", ""))).name]
                candidates.extend(cfg.get("aliases", []) or [])
                candidate_norms = [self.normalize_model_id(c) for c in candidates if c]
                if any(normalized in c or c in normalized for c in candidate_norms):
                    matches.append(name)

            if len(matches) == 1:
                return matches[0]
            return None

    def model_metadata(self, model_name):
        """Return public metadata for one registry model."""
        cfg = self.get(model_name)
        if not cfg:
            return None
        return {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "grimoire",
            "context": cfg.get("ctx-size"),
            "output": cfg.get("predict"),
            "family": cfg.get("family"),
            "capabilities": cfg.get("capabilities", ["completion"]),
            "cost": cfg.get("cost", {"input": 0, "output": 0}),
            "backend": _get_backend(cfg),
            "pinned_gpu": self.get_fixed_gpu(model_name),
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
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("GPU ID must be a non-negative integer")
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
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

    def validate(self, model_name, gpu_count=None):
        """Check if a model config is valid."""
        cfg = self.get(model_name)
        if not cfg:
            return False, f"Model '{model_name}' not found"

        backend = _get_backend(cfg)
        if backend == BACKEND_LLAMA:
            if not cfg.get("file"):
                return False, "Missing 'file' field"
            model_path = os.path.join(MODELS_DIR, cfg["file"])
            if not os.path.exists(model_path):
                return False, f"Model file not found at {model_path}"
        elif backend == BACKEND_DFLASH:
            target = resolve_path(cfg, "target")
            if not target or not os.path.exists(target):
                return False, f"Target model not found at {target}"
            draft = resolve_path(cfg, "draft")
            if not draft or not os.path.exists(draft):
                return False, f"Draft model not found at {draft}"
            drafter = resolve_path(cfg, "drafter")
            if drafter and not os.path.exists(drafter):
                return False, f"Drafter model not found at {drafter}"
            tokenizer = cfg.get("tokenizer")
            if not tokenizer:
                return False, (
                    "Missing 'tokenizer' field: dflash backends need an explicit "
                    "tokenizer (HF repo id or local path); runtime download is not safe"
                )
            if _looks_like_local_path(tokenizer):
                tok_path = resolve_path(cfg, "tokenizer")
                if tok_path and not os.path.exists(tok_path):
                    return False, f"Tokenizer path not found at {tok_path}"
        else:
            return False, f"Unknown backend '{backend}'"

        fixed_gpu = self.get_fixed_gpu(model_name)
        if fixed_gpu is not None:
            if not isinstance(fixed_gpu, int) or fixed_gpu < 0:
                return False, f"Invalid pinned GPU ID for '{model_name}': {fixed_gpu}"
            if gpu_count is not None and fixed_gpu >= gpu_count:
                return False, f"Pinned GPU {fixed_gpu} is outside available range 0-{gpu_count - 1}"

        return True, "OK"


registry = ModelRegistry()
