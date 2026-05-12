import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

from fastapi import HTTPException

import grimoire.entrypoint as entrypoint
from grimoire.history import HistoryStore, identity_hash
from grimoire.registry import ModelRegistry


class FakeRequest:
    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


class DropInBlockerTests(unittest.TestCase):
    def test_required_legacy_model_aliases_are_registered(self):
        data = json.loads((ROOT / "etc" / "models.json").read_text())
        aliases = set(data["models"])
        self.assertTrue({
            "gemma-4-31B",
            "qwen-3.6-27B",
            "huihui-qwen3.5-27B",
            "huihui-gemma-4-31B",
            "qwopus-3.6-27B",
        }.issubset(aliases))

    def test_auth_fails_closed_without_api_key(self):
        old_api_key = entrypoint.API_KEY
        old_allow_anonymous = entrypoint.ALLOW_ANONYMOUS
        try:
            entrypoint.API_KEY = ""
            entrypoint.ALLOW_ANONYMOUS = False
            with self.assertRaises(HTTPException) as cm:
                entrypoint.require_api(FakeRequest())
            self.assertEqual(cm.exception.status_code, 503)
        finally:
            entrypoint.API_KEY = old_api_key
            entrypoint.ALLOW_ANONYMOUS = old_allow_anonymous

    def test_anonymous_mode_requires_explicit_opt_in(self):
        old_api_key = entrypoint.API_KEY
        old_allow_anonymous = entrypoint.ALLOW_ANONYMOUS
        try:
            entrypoint.API_KEY = ""
            entrypoint.ALLOW_ANONYMOUS = True
            token, user_hash = entrypoint.require_api(FakeRequest())
            self.assertEqual(token, "anonymous")
            self.assertEqual(user_hash, identity_hash("anonymous"))
        finally:
            entrypoint.API_KEY = old_api_key
            entrypoint.ALLOW_ANONYMOUS = old_allow_anonymous

    def test_bearer_auth_uses_legacy_gateway_key(self):
        old_api_key = entrypoint.API_KEY
        try:
            entrypoint.API_KEY = "legacy-key"
            token, user_hash = entrypoint.require_api(FakeRequest(headers={"authorization": "Bearer legacy-key"}))
            self.assertEqual(token, "legacy-key")
            self.assertEqual(user_hash, identity_hash("legacy-key"))
        finally:
            entrypoint.API_KEY = old_api_key

    def test_login_template_renders_literal_css_braces(self):
        html = entrypoint._render_login_html("")
        self.assertIn("body{margin:0", html)
        self.assertNotIn("{error}", html)

    def test_build_cmd_binds_backend_to_loopback(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            cmd = entrypoint.build_cmd({"file": model_file.name}, port=8001)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_proxy_headers_strip_credentials_and_hop_by_hop_headers(self):
        headers = entrypoint._backend_request_headers({
            "authorization": "Bearer secret",
            "x-grimoire-token": "secret",
            "cookie": "gw_session=secret",
            "host": "chat.lost.plus",
            "content-length": "123",
            "content-type": "application/json",
        })
        self.assertEqual(headers, {"content-type": "application/json"})

    def test_registry_reads_seed_but_saves_to_state_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state" / "models.json"
            seed_path = Path(tmp) / "seed.json"
            seed_path.write_text(json.dumps({"models": {"seed-model": {"file": "seed.gguf"}}, "fixed": {}}))

            registry = ModelRegistry(path=str(state_path), seed_path=str(seed_path))
            self.assertEqual(registry.list_all(), ["seed-model"])
            self.assertFalse(state_path.exists())

            registry.add("new-model", {"file": "new.gguf"})
            saved = json.loads(state_path.read_text())
            self.assertIn("seed-model", saved["models"])
            self.assertIn("new-model", saved["models"])

    def test_stop_model_resolves_alias_before_stopping(self):
        class FakeRegistry:
            def resolve(self, name):
                return "canonical" if name == "alias" else name

        class FakeActive:
            def __init__(self):
                self.stopped = False

            def is_running(self):
                return True

            def stop(self):
                self.stopped = True

        old_registry = entrypoint.registry
        try:
            entrypoint.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            active = FakeActive()
            manager.active["canonical"] = active
            stopped = asyncio.run(manager.stop_model("alias"))
            self.assertTrue(stopped)
            self.assertTrue(active.stopped)
            self.assertNotIn("canonical", manager.active)
        finally:
            entrypoint.registry = old_registry

    def test_invalid_history_id_is_ignored_without_orphan_creation(self):
        class FakeHistoryStore:
            def get_conversation(self, user_hash, conversation_id):
                raise KeyError(conversation_id)

            def conversation_exists(self, user_hash, conversation_id):
                return False

            def create_conversation(self, *args, **kwargs):
                raise AssertionError("invalid conversation IDs must not create orphan conversations")

        old_history_store = entrypoint.history_store
        try:
            entrypoint.history_store = FakeHistoryStore()
            self.assertIsNone(entrypoint._validated_history_conversation_id("user", "missing"))
        finally:
            entrypoint.history_store = old_history_store

    def test_usage_is_recorded_from_tail_beyond_history_capture_limit(self):
        class FakeUsageStore:
            def __init__(self):
                self.records = []

            def record(self, *args, **kwargs):
                self.records.append((args, kwargs))

        async def stream():
            yield b"x" * 128 + b"\n\n"
            yield b'data: {"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'

        async def consume(async_iter):
            return [chunk async for chunk in async_iter]

        old_usage_store = entrypoint.usage_store
        old_history_capture = entrypoint.MAX_HISTORY_CAPTURE_BYTES
        old_usage_capture = entrypoint.MAX_USAGE_CAPTURE_BYTES
        fake_usage = FakeUsageStore()
        try:
            entrypoint.usage_store = fake_usage
            entrypoint.MAX_HISTORY_CAPTURE_BYTES = 1
            entrypoint.MAX_USAGE_CAPTURE_BYTES = 1024
            chunks = asyncio.run(consume(entrypoint._record_response_stream(
                stream(),
                user_hash="user",
                conversation_id=None,
                model_name="model",
                model_cfg={"cost": {}},
                payload={},
                record_history=False,
            )))
            self.assertEqual(len(chunks), 2)
            self.assertEqual(fake_usage.records[0][0][2:4], (3, 4))
        finally:
            entrypoint.usage_store = old_usage_store
            entrypoint.MAX_HISTORY_CAPTURE_BYTES = old_history_capture
            entrypoint.MAX_USAGE_CAPTURE_BYTES = old_usage_capture

    def test_history_delete_cascades_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "history.sqlite3")
            store = HistoryStore(path)
            conversation = store.create_conversation(
                "user",
                title="chat",
                messages=[{"role": "user", "content": "hello"}],
            )
            store.delete_conversation("user", conversation["id"])

            with sqlite3.connect(path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            self.assertEqual(count, 0)

    def test_deployment_uses_persistent_registry_path_and_dockerignore(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("GRIMOIRE_REGISTRY_PATH=/var/lib/grimoire/models.json", dockerfile)
        self.assertIn("GRIMOIRE_REGISTRY_SEED_PATH=/etc/grimoire/models.json", dockerfile)

        dockerignore = (ROOT / ".dockerignore").read_text()
        self.assertIn("build/", dockerignore)
        self.assertIn("*.egg-info/", dockerignore)

    def test_webui_history_patch_is_well_formed(self):
        patch_path = ROOT / "patches" / "grimoire-webui-history.patch"
        self.assertTrue(patch_path.exists(), "webui history patch file is missing")
        content = patch_path.read_text()
        self.assertIn("diff --git", content)
        self.assertIn("tools/server/webui/src/lib/services/database.service.ts", content)
        self.assertIn("apiFetch", content)
        # The webui stage selectively applies grimoire-webui-* patches
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("/src/patches/grimoire-webui-*.patch", dockerfile)
        self.assertIn("grimoire-webui-*", dockerfile)

    def test_dflash_runtime_uses_test_dflash_binary(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("-DDFLASH27B_TESTS=ON", dockerfile)
        self.assertIn("--target test_dflash", dockerfile)
        self.assertIn("/app/.cache/dflash-build/build/test_dflash /opt/dflash/dflash", dockerfile)
        self.assertNotIn("--target pflash_daemon", dockerfile)
        self.assertNotIn("/app/.cache/dflash-build/build/pflash_daemon /opt/dflash/dflash", dockerfile)


if __name__ == "__main__":
    unittest.main()
