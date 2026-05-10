"""Model registry - CRUD operations for models.json."""

import copy
import json
import logging
import os
from datetime import datetime, timezone
from threading import RLock

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("GRIMOIRE_MODELS_DIR", "/models")
DEFAULT_REGISTRY_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)


class ModelRegistry:
    """Model registry backed by JSON file.

    Schema:
    {
      "models": { "alias": { "file": "...", "ctx-size": 262144, ... } },
      "fixed": { "alias": 0 }
    }
    """

    def __init__(self, path=None):
        self.path = path or REGISTRY_PATH
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
        try:
            with open(self.path) as f:
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
        if not cfg.get("file"):
            return False, "Missing 'file' field"

        model_path = os.path.join(MODELS_DIR, cfg["file"])
        if not os.path.exists(model_path):
            return False, f"Model file not found at {model_path}"

        fixed_gpu = self.get_fixed_gpu(model_name)
        if fixed_gpu is not None:
            if not isinstance(fixed_gpu, int) or fixed_gpu < 0:
                return False, f"Invalid pinned GPU ID for '{model_name}': {fixed_gpu}"
            if gpu_count is not None and fixed_gpu >= gpu_count:
                return False, f"Pinned GPU {fixed_gpu} is outside available range 0-{gpu_count - 1}"

        return True, "OK"


registry = ModelRegistry()
