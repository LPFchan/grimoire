"""Stock llama.cpp webui router-mode contract tests for the grimoire gateway."""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-usage.sqlite3"))
os.environ.setdefault("GRIMOIRE_REGISTRY_SEED_PATH", str(ROOT / "etc" / "models.json"))
os.environ.setdefault("GRIMOIRE_REGISTRY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-router-registry.json"))

from fastapi.testclient import TestClient

import grimoire.config as config
import grimoire.entrypoint as entrypoint
import grimoire.routes.models as models_routes


class FakeActive:
    def __init__(self, name, gpu=0, gpus=None, port=8001, status=entrypoint.MODEL_STATUS_LOADED):
        self.name = name
        self.gpu = gpu
        self.gpus = list(gpus) if gpus is not None else [gpu]
        self.port = port
        self.status = status
        self.started = datetime.now(timezone.utc)

    def is_running(self):
        return self.status == entrypoint.MODEL_STATUS_LOADED


class FakeLoraResponse:
    status_code = 200
    text = '{"success":true}'


class FakeLoraClient:
    last_post = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        type(self).last_post = (url, json)
        return FakeLoraResponse()


class RouterModeContractTests(unittest.TestCase):
    def setUp(self):
        self._old_api = config.API_KEY
        self._old_admin = config.ADMIN_TOKEN
        config.API_KEY = "test-key"
        config.ADMIN_TOKEN = "test-key"
        entrypoint.manager.active.clear()
        self.client = TestClient(entrypoint.app)
        self.auth = {"Authorization": "Bearer test-key"}

    def tearDown(self):
        config.API_KEY = self._old_api
        config.ADMIN_TOKEN = self._old_admin
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

    def test_v1_models_includes_registry_capability_metadata(self):
        response = self.client.get("/v1/models", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        for entry in response.json()["data"]:
            self.assertIn("input_modalities", entry)
            capabilities = entry.get("capabilities", [])
            expected = ["text", "image"] if {"multimodal", "vision"} & set(capabilities) else ["text"]
            self.assertEqual(entry["input_modalities"], expected)
            self.assertIn("reasoning", entry)
            self.assertIsInstance(entry["reasoning"], dict)

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

    def test_status_and_switch_preserve_primary_gpu_and_add_full_placement(self):
        registry_aliases = entrypoint.registry.list_all()
        if not registry_aliases:
            self.skipTest("registry seed empty")
        name = registry_aliases[0]
        entrypoint.manager.active[name] = FakeActive(name, gpu=1, gpus=[1, 0])

        status_response = self.client.get("/status", headers=self.auth)
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["active"][name]["gpu"], 1)
        self.assertEqual(status_response.json()["active"][name]["gpus"], [1, 0])

        switch_response = self.client.post(f"/switch/{name}", headers=self.auth)
        self.assertEqual(switch_response.status_code, 200)
        self.assertEqual(switch_response.json()["gpu"], 1)
        self.assertEqual(switch_response.json()["gpus"], [1, 0])

    def test_models_load_calls_switch_with_payload_model(self):
        called = {}

        async def fake_switch(model_name, request):
            called["name"] = model_name
            return {"status": "started", "model": model_name, "gpu": 0, "port": 8001}

        with patch.object(models_routes, "switch_model", fake_switch):
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

        with patch.object(models_routes, "stop_model_endpoint", fake_stop):
            response = self.client.post(
                "/models/unload",
                json={"model": "gemma-4-31B"},
                headers=self.auth,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(called["name"], "gemma-4-31B")

    def test_runtime_clone_route_validates_and_returns_normalized_state(self):
        name = "runtime-test-model"
        metadata = {
            "gpu": None, "gpus": [], "requested_gpu": 0, "requested_gpus": [0, 1],
            "placement_source": "runtime", "pinned": False, "pinned_gpu": None,
            "pin_source": None, "runtime_override": {"gpu_ids": [0, 1]},
        }
        with patch.object(entrypoint.manager, "clone_model", new=AsyncMock()) as clone, \
                patch.object(entrypoint.manager, "override_metadata", return_value=metadata), \
                patch.object(entrypoint.registry, "resolve", return_value=name):
            response = self.client.post(
                f"/models/{name}/clone", json={"gpu_ids": [0, 1], "tensor_split": [1, 1]}, headers=self.auth,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requested_gpus"], [0, 1])
        clone.assert_awaited_once_with(name, [0, 1], [1, 1])

        for payload in ({}, {"gpu_ids": [0]}, {"gpu_ids": [0, 0]},
                        {"gpu_ids": [0, 1], "tensor_split": [True, 1]}):
            response = self.client.post(f"/models/{name}/clone", json=payload, headers=self.auth)
            self.assertEqual(response.status_code, 400, payload)

    def test_alpha_get_returns_active_adapter_scales(self):
        registry_aliases = entrypoint.registry.list_all()
        if not registry_aliases:
            self.skipTest("registry seed empty")
        name = registry_aliases[0]
        entrypoint.manager.active[name] = FakeActive(name)

        async def fake_adapters(_active):
            return [{"id": 0, "scale": 0.8}]

        with patch.object(models_routes, "_backend_lora_adapters", fake_adapters):
            response = self.client.get("/alpha", headers=self.auth)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], name)
        self.assertEqual(response.json()["adapters"][0]["scale"], 0.8)

    def test_alpha_post_updates_active_backend_lora_scale(self):
        registry_aliases = entrypoint.registry.list_all()
        if not registry_aliases:
            self.skipTest("registry seed empty")
        name = registry_aliases[0]
        entrypoint.manager.active[name] = FakeActive(name, port=8123)
        FakeLoraClient.last_post = None

        async def fake_adapters(_active):
            return [{"id": 0, "scale": 0.6}]

        with patch.object(models_routes, "_backend_lora_adapters", fake_adapters), \
                patch.object(models_routes.httpx, "AsyncClient", FakeLoraClient):
            response = self.client.post("/alpha", json=0.6, headers=self.auth)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["alpha"], 0.6)
        self.assertEqual(
            FakeLoraClient.last_post,
            ("http://127.0.0.1:8123/lora-adapters", [{"id": 0, "scale": 0.6}]),
        )

    def test_router_endpoints_require_auth(self):
        for path, method, body in [
            ("/props", "get", None),
            ("/v1/models", "get", None),
            ("/models/load", "post", {"model": "x"}),
            ("/models/unload", "post", {"model": "x"}),
            ("/models/x/clone", "post", {"gpu_ids": [0, 1]}),
            ("/models/x/declone", "post", None),
            ("/models/x/pin", "post", {"gpu": 0}),
            ("/models/x/unpin", "post", None),
            ("/alpha", "get", None),
            ("/alpha", "post", 0.6),
        ]:
            response = self.client.request(method, path, json=body)
            self.assertEqual(response.status_code, 401, f"{method} {path} should require auth")


if __name__ == "__main__":
    unittest.main()
