"""Stock llama.cpp webui router-mode contract tests for the grimoire gateway."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-usage.sqlite3"))
os.environ.setdefault("GRIMOIRE_REGISTRY_SEED_PATH", str(ROOT / "etc" / "models.json"))
os.environ.setdefault("GRIMOIRE_REGISTRY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-registry.json"))

from fastapi.testclient import TestClient

import grimoire.entrypoint as entrypoint


class FakeActive:
    def __init__(self, name, gpu=0, port=8001, status=entrypoint.MODEL_STATUS_LOADED):
        self.name = name
        self.gpu = gpu
        self.port = port
        self.status = status

    def is_running(self):
        return self.status == entrypoint.MODEL_STATUS_LOADED


class RouterModeContractTests(unittest.TestCase):
    def setUp(self):
        self._old_api = entrypoint.API_KEY
        self._old_admin = entrypoint.ADMIN_TOKEN
        entrypoint.API_KEY = "test-key"
        entrypoint.ADMIN_TOKEN = "test-key"
        entrypoint.manager.active.clear()
        self.client = TestClient(entrypoint.app)
        self.auth = {"Authorization": "Bearer test-key"}

    def tearDown(self):
        entrypoint.API_KEY = self._old_api
        entrypoint.ADMIN_TOKEN = self._old_admin
        entrypoint.manager.active.clear()

    def test_props_root_returns_router_role(self):
        response = self.client.get("/props", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["role"], "router")
        self.assertIn("default_generation_settings", data)
        self.assertIn("params", data["default_generation_settings"])
        self.assertEqual(data["modalities"], {"vision": False, "audio": False})

    def test_props_unknown_model_returns_404(self):
        response = self.client.get("/props", params={"model": "nonexistent-model"}, headers=self.auth)
        self.assertEqual(response.status_code, 404)

    def test_props_known_model_with_autoload_false_returns_synthetic(self):
        registry_aliases = entrypoint.registry.list_all()
        if not registry_aliases:
            self.skipTest("registry seed empty")
        name = registry_aliases[0]
        response = self.client.get(
            "/props",
            params={"model": name, "autoload": "false"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["role"], "router")
        cfg = entrypoint.registry.get(name)
        self.assertEqual(data["default_generation_settings"]["n_ctx"], cfg["ctx-size"])

    def test_v1_models_includes_status_field_for_webui(self):
        response = self.client.get("/v1/models", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        for entry in data["data"]:
            self.assertIn("status", entry, "webui needs status.value on every model")
            self.assertIn(entry["status"]["value"], {
                entrypoint.MODEL_STATUS_LOADED,
                entrypoint.MODEL_STATUS_LOADING,
                entrypoint.MODEL_STATUS_UNLOADED,
                entrypoint.MODEL_STATUS_FAILED,
            })

    def test_v1_models_marks_active_model_as_loaded(self):
        registry_aliases = entrypoint.registry.list_all()
        if not registry_aliases:
            self.skipTest("registry seed empty")
        name = registry_aliases[0]
        entrypoint.manager.active[name] = FakeActive(name)
        response = self.client.get("/v1/models", headers=self.auth)
        loaded = [e for e in response.json()["data"] if e["id"] == name][0]
        self.assertEqual(loaded["status"]["value"], entrypoint.MODEL_STATUS_LOADED)
        self.assertTrue(loaded["active"])

    def test_models_load_calls_switch_with_payload_model(self):
        called = {}

        async def fake_switch(model_name, request):
            called["name"] = model_name
            return {"status": "started", "model": model_name, "gpu": 0, "port": 8001}

        with patch.object(entrypoint, "switch_model", fake_switch):
            response = self.client.post(
                "/models/load",
                json={"model": "gemma-4-31B"},
                headers=self.auth,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(called["name"], "gemma-4-31B")

    def test_models_load_rejects_missing_model_field(self):
        response = self.client.post("/models/load", json={}, headers=self.auth)
        self.assertEqual(response.status_code, 400)

    def test_models_unload_calls_stop_with_payload_model(self):
        called = {}

        async def fake_stop(model_name, request):
            called["name"] = model_name
            return {"status": "stopped", "model": model_name}

        with patch.object(entrypoint, "stop_model_endpoint", fake_stop):
            response = self.client.post(
                "/models/unload",
                json={"model": "gemma-4-31B"},
                headers=self.auth,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(called["name"], "gemma-4-31B")

    def test_router_endpoints_require_auth(self):
        for path, method, body in [
            ("/props", "get", None),
            ("/v1/models", "get", None),
            ("/models/load", "post", {"model": "x"}),
            ("/models/unload", "post", {"model": "x"}),
        ]:
            response = self.client.request(method, path, json=body)
            self.assertEqual(response.status_code, 401, f"{method} {path} should require auth")


if __name__ == "__main__":
    unittest.main()
