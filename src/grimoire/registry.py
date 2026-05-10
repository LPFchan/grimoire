"""Model registry - CRUD operations for models.json."""

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("GRIMOIRE_MODELS_DIR", "/models")
DEFAULT_REGISTRY_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)


class ModelRegistry:
    """Model registry backed by JSON file.

    Schema:
    {
      "models": { "alias": { "file": "...", "ctx-size": 262144, ... } },
      "fixed": { "alias": 0 }   # pinned GPU assignments
    }
    """

    def __init__(self, path=None):
        self.path = path or REGISTRY_PATH
        self._data = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {"models": {}, "fixed": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry: {e}")
            return {"models": {}, "fixed": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Registry saved to {self.path}")

    def get(self, model_name):
        return self._data.get("models", {}).get(model_name)

    def list_all(self):
        return list(self._data.get("models", {}).keys())

    def is_fixed(self, model_name):
        """Check if a model is pinned to a GPU."""
        return model_name in self._data.get("fixed", {})

    def get_fixed_gpu(self, model_name):
        """Get the pinned GPU ID for a model, or None."""
        return self._data.get("fixed", {}).get(model_name)

    def add(self, model_name, config):
        if model_name in self._data.get("models", {}):
            raise ValueError(f"Model '{model_name}' already exists")
        self._data.setdefault("models", {})[model_name] = {
            **config,
            "added": datetime.now(timezone.utc).isoformat()
        }
        self._save()
        return self._data["models"][model_name]

    def update(self, model_name, updates):
        if model_name not in self._data.get("models", {}):
            raise KeyError(f"Model '{model_name}' not found")
        self._data["models"][model_name].update(updates)
        self._save()
        return self._data["models"][model_name]

    def remove(self, model_name):
        if model_name not in self._data.get("models", {}):
            raise KeyError(f"Model '{model_name}' not found")
        del self._data["models"][model_name]
        # Also remove from fixed if pinned
        self._data.get("fixed", {}).pop(model_name, None)
        self._save()

    def pin_gpu(self, model_name, gpu_id):
        """Pin a model to a specific GPU."""
        if model_name not in self._data.get("models", {}):
            raise KeyError(f"Model '{model_name}' not found")
        self._data.setdefault("fixed", {})[model_name] = gpu_id
        self._save()

    def unpin_gpu(self, model_name):
        """Remove GPU pinning for a model."""
        if model_name in self._data.get("fixed", {}):
            del self._data["fixed"][model_name]
            self._save()

    def validate(self, model_name):
        """Check if a model config is valid."""
        cfg = self.get(model_name)
        if not cfg:
            return False, f"Model '{model_name}' not found"
        if not cfg.get("file"):
            return False, "Missing 'file' field"
        model_path = os.path.join(MODELS_DIR, cfg["file"])
        if not os.path.exists(model_path):
            return False, f"Model file not found at {model_path}"
        return True, "OK"


registry = ModelRegistry()
