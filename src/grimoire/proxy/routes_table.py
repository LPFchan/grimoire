"""Shared backend route table for the multi-process gateway.

The manager (single process owning ModelManager) publishes the active model ->
backend mapping to a small tmpfs file. The stateless proxy workers read it to
forward requests, round-robining across a model's replica backends. This is the
shared state that lets N proxy workers scale past the single-process ceiling
without each managing model lifecycle.

File shape (`/dev/shm/grimoire/routes.json`):

    {
      "version": 7,
      "models": {
        "eastself-embedder-0.6B": {
          "status": "loaded",
          "replicas": [
            {"port": 8011, "backend_model_id": "eastself-embedder-0.6B"},
            {"port": 8001, "backend_model_id": "eastself-embedder-0.6B"}
          ]
        }
      }
    }
"""

import json
import os
import threading
import time

DEFAULT_PATH = os.environ.get("GRIMOIRE_ROUTES_PATH", "/dev/shm/grimoire/routes.json")


def publish(models: dict, path: str = DEFAULT_PATH) -> None:
    """Atomically write the route table. Called by the manager on model changes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"version": int(time.time() * 1000), "models": models}
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


class RouteTableReader:
    """mtime-cached reader for the proxy workers (one instance per worker)."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._mtime = 0.0
        self._models: dict = {}

    def _maybe_reload(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime == self._mtime:
            return
        with self._lock:
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._models = data.get("models", {})
                self._mtime = mtime
            except (OSError, ValueError):
                pass

    def replicas(self, model: str) -> list[dict]:
        """Return the list of {port, backend_model_id} replicas for a model.

        Empty list if the model is unknown or not loaded yet.
        """
        self._maybe_reload()
        entry = self._models.get(model)
        if not entry or entry.get("status") != "loaded":
            return []
        return entry.get("replicas", [])
