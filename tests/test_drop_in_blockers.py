import asyncio
from datetime import datetime, timezone
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

from fastapi import HTTPException

import grimoire.config as config
import grimoire.entrypoint as entrypoint
import grimoire.model_manager as mm_module
import grimoire.routes.models as models_routes
from grimoire.history import HistoryStore, identity_hash
from grimoire.registry import ModelRegistry


class FakeRequest:
    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


class DropInBlockerTests(unittest.TestCase):
    def test_pinned_aliases_exist_in_the_registry(self):
        """A GPU pin must name a model the registry can actually serve.

        The `fixed` block pins the always-on embedder and reranker to a GPU. A
        pin naming an alias that no longer exists strands silently: nothing
        errors at load, the model simply never starts, and the endpoints it
        backs return 503 with no obvious cause.
        """
        data = json.loads((ROOT / "etc" / "models.json").read_text())
        models = data["models"]
        pinned = data.get("fixed", {})

        self.assertTrue(pinned, "no GPU pins configured; always-on models are unpinned")
        for alias in pinned:
            self.assertIn(alias, models, f"GPU pin names an unknown model: {alias}")

    def test_auth_fails_closed_without_api_key(self):
        old_api_key = config.API_KEY
        old_allow_anonymous = config.ALLOW_ANONYMOUS
        try:
            config.API_KEY = ""
            config.ALLOW_ANONYMOUS = False
            with self.assertRaises(HTTPException) as cm:
                entrypoint.require_api(FakeRequest())
            self.assertEqual(cm.exception.status_code, 503)
        finally:
            config.API_KEY = old_api_key
            config.ALLOW_ANONYMOUS = old_allow_anonymous

    def test_anonymous_mode_requires_explicit_opt_in(self):
        old_api_key = config.API_KEY
        old_allow_anonymous = config.ALLOW_ANONYMOUS
        try:
            config.API_KEY = ""
            config.ALLOW_ANONYMOUS = True
            token, user_hash = entrypoint.require_api(FakeRequest())
            self.assertEqual(token, "anonymous")
            self.assertEqual(user_hash, identity_hash("anonymous"))
        finally:
            config.API_KEY = old_api_key
            config.ALLOW_ANONYMOUS = old_allow_anonymous

    def test_bearer_auth_uses_legacy_gateway_key(self):
        old_api_key = config.API_KEY
        try:
            config.API_KEY = "legacy-key"
            token, user_hash = entrypoint.require_api(FakeRequest(headers={"authorization": "Bearer legacy-key"}))
            self.assertEqual(token, "legacy-key")
            self.assertEqual(user_hash, identity_hash("legacy-key"))
        finally:
            config.API_KEY = old_api_key

    def test_login_template_renders_literal_css_braces(self):
        html = entrypoint._render_login_html("")
        self.assertIn("body{margin:0", html)
        self.assertNotIn("{error}", html)

    def test_build_cmd_binds_backend_to_loopback(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            cmd = entrypoint.build_cmd({"file": model_file.name}, port=8001)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_build_cmd_emits_native_dflash_canary_flags(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, tempfile.NamedTemporaryFile(suffix=".gguf") as draft_file:
            cmd = entrypoint.build_cmd(
                {
                    "file": model_file.name,
                    "draft": draft_file.name,
                    "speculative-type": "dflash",
                    "spec-dflash-cross-ctx": 1024,
                },
                port=8001,
            )
        self.assertIn("--spec-type", cmd)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "dflash")
        self.assertIn("--spec-draft-model", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-model") + 1], draft_file.name)
        self.assertIn("--spec-dflash-cross-ctx", cmd)
        self.assertEqual(cmd[cmd.index("--spec-dflash-cross-ctx") + 1], "1024")

    def test_build_cmd_merges_template_kwargs_and_keeps_reasoning_per_request(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, patch.object(
            mm_module.registry,
            "get_family_defaults",
            return_value={
                "extra-args": ["--chat-template-kwargs", '{"preserve_thinking":true}']
            },
        ):
            cfg = {
                "file": model_file.name,
                "family": "qwen",
                "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"low"}'],
            }
            original = json.loads(json.dumps(cfg))
            cmd = entrypoint.build_cmd(cfg, port=8001)

        self.assertEqual(cmd.count("--chat-template-kwargs"), 1)
        kwargs = json.loads(cmd[cmd.index("--chat-template-kwargs") + 1])
        self.assertEqual(kwargs, {"preserve_thinking": True})
        self.assertEqual(cfg, original)

    def test_build_cmd_uses_family_chat_template_with_model_override(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, \
             tempfile.NamedTemporaryFile(suffix=".jinja") as family_template, \
             tempfile.NamedTemporaryFile(suffix=".jinja") as model_template, \
             patch.object(
                 mm_module.registry,
                 "get_family_defaults",
                 return_value={"chat-template-file": family_template.name},
             ):
            inherited = entrypoint.build_cmd(
                {"file": model_file.name, "family": "gemma4"},
                port=8001,
            )
            overridden = entrypoint.build_cmd(
                {
                    "file": model_file.name,
                    "family": "gemma4",
                    "chat-template-file": model_template.name,
                },
                port=8001,
            )

        self.assertEqual(
            inherited[inherited.index("--chat-template-file") + 1],
            family_template.name,
        )
        self.assertEqual(
            overridden[overridden.index("--chat-template-file") + 1],
            model_template.name,
        )

    def test_start_model_reuses_command_equivalent_reasoning_alias(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            configs = {
                "qwen-xhigh": {
                    "file": model_file.name,
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"xhigh"}'],
                },
                "qwen-low": {
                    "file": model_file.name,
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"low"}'],
                },
            }

            class FakeRegistry:
                def resolve(self, name):
                    return name if name in configs else None

                def get(self, name):
                    value = configs.get(name)
                    return json.loads(json.dumps(value)) if value else None

                def get_family_defaults(self, family):
                    return {}

                def get_fixed_gpu(self, name):
                    return None

            class FakeActive:
                name = "qwen-xhigh"
                cfg = configs["qwen-xhigh"]
                status = config.MODEL_STATUS_LOADED

                def is_running(self):
                    return True

            old_registry = mm_module.registry
            try:
                mm_module.registry = FakeRegistry()
                manager = entrypoint.ModelManager(gpu_count=1)
                active = FakeActive()
                manager.active[active.name] = active

                reused = asyncio.run(manager.start_model("qwen-low"))

                self.assertIs(reused, active)
                self.assertIs(manager.get_active("qwen-low"), active)
                self.assertEqual(manager.get_status("qwen-low"), config.MODEL_STATUS_LOADED)
                self.assertEqual(list(manager.active), ["qwen-xhigh"])
            finally:
                mm_module.registry = old_registry

    def test_runtime_signature_ignores_muse_reasoning_strength(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, patch.object(
            mm_module.registry, "get_family_defaults", return_value={}
        ):
            signatures = {
                mm_module.runtime_command_signature(
                    {
                        "file": model_file.name,
                        "extra-args": [
                            "--chat-template-kwargs",
                            json.dumps({"reasoning_strength": strength}),
                        ],
                    }
                )
                for strength in ("low", "medium", "high", "xhigh")
            }

        self.assertEqual(len(signatures), 1)

    def test_command_equivalence_does_not_merge_unrelated_aliases_or_replicas(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            configs = {
                "plain-a": {"file": model_file.name},
                "plain-b": {"file": model_file.name},
                "reasoning-a": {
                    "file": model_file.name,
                    "replica_peers": ["reasoning-a-copy"],
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"low"}'],
                },
                "reasoning-b": {
                    "file": model_file.name,
                    "replica_peers": ["reasoning-b-copy"],
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"high"}'],
                },
            }

            class FakeRegistry:
                def get(self, name):
                    return configs.get(name)

                def get_family_defaults(self, family):
                    return {}

                def get_fixed_gpu(self, name):
                    return None

            class FakeActive:
                def __init__(self, name):
                    self.name = name
                    self.cfg = configs[name]

                def is_running(self):
                    return True

            old_registry = mm_module.registry
            try:
                mm_module.registry = FakeRegistry()
                manager = entrypoint.ModelManager(gpu_count=2)
                manager.active["plain-a"] = FakeActive("plain-a")
                self.assertEqual(manager._compatible_active_entry("plain-b"), (None, None))

                manager.active = {"reasoning-a": FakeActive("reasoning-a")}
                self.assertEqual(manager._compatible_active_entry("reasoning-b"), (None, None))

                sharded = dict(configs["reasoning-b"], **{"gpu-ids": [0, 1]})
                configs["reasoning-sharded"] = sharded
                self.assertEqual(manager._compatible_active_entry("reasoning-sharded"), (None, None))
            finally:
                mm_module.registry = old_registry

    def test_explicit_request_only_template_kwargs_allow_alias_reuse(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            configs = {
                "low": {
                    "file": model_file.name,
                    "chat-template-kwargs": {"reasoning_effort": "low"},
                },
                "high": {
                    "file": model_file.name,
                    "chat-template-kwargs": {"reasoning_effort": "high"},
                },
            }

            class FakeRegistry:
                def get(self, name):
                    return configs.get(name)

                def get_family_defaults(self, family):
                    return {}

                def get_fixed_gpu(self, name):
                    return None

            class FakeActive:
                name = "low"
                cfg = configs[name]

                def is_running(self):
                    return True

            with patch.object(mm_module, "registry", FakeRegistry()):
                manager = entrypoint.ModelManager(gpu_count=1)
                active = FakeActive()
                manager.active[active.name] = active
                self.assertEqual(manager._compatible_active_entry("high"), ("low", active))

    def test_concurrent_cold_start_returns_the_same_healthy_backend(self):
        cfg = {"file": "model.gguf"}

        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return cfg

            def get_family_defaults(self, family):
                return {}

            def get_fixed_gpu(self, name):
                return None

            def validate(self, name, gpu_count=None):
                return True, "OK"

        class FakeActive:
            def __init__(self, name, cfg, port, gpu):
                self.name = name
                self.cfg = cfg
                self.port = port
                self.gpu = gpu.primary
                self.gpus = gpu.device_ids
                self.running = False
                self.stop_calls = 0

            def is_running(self):
                return self.running

            def stop(self):
                self.stop_calls += 1
                self.running = False

        async def scenario():
            manager = entrypoint.ModelManager(gpu_count=1)
            started = asyncio.Event()
            release = asyncio.Event()
            launches = []

            async def fake_start(active):
                launches.append(active)
                started.set()
                await release.wait()
                active.running = True

            with patch.object(mm_module, "registry", FakeRegistry()), \
                 patch.object(mm_module, "ActiveModel", FakeActive), \
                 patch.object(
                     manager,
                     "_allocate_gpu",
                     return_value=(mm_module.GpuPlacement((0,)), []),
                 ), \
                 patch.object(manager, "_find_available_port_excluding", return_value=8001), \
                 patch.object(manager, "_start_active_model", side_effect=fake_start):
                first = asyncio.create_task(manager.start_model("model"))
                await started.wait()
                second = asyncio.create_task(manager.start_model("model"))
                await asyncio.sleep(0)
                release.set()
                results = await asyncio.gather(first, second)

            return results, launches

        results, launches = asyncio.run(scenario())
        self.assertIs(results[0], results[1])
        self.assertTrue(results[0].is_running())
        self.assertEqual(results[0].stop_calls, 0)
        self.assertEqual(len(launches), 1)

    def test_webui_cache_id_does_not_enable_legacy_history_recording(self):
        request = FakeRequest(headers={"x-grimoire-kv-conversation-id": "conv-cache"})
        payload = {"messages": []}

        history_id = entrypoint._history_conversation_id(request, payload)

        self.assertIsNone(history_id)
        self.assertEqual(entrypoint._kv_conversation_id(request, history_id), "conv-cache")

    def test_legacy_history_id_still_doubles_as_cache_id(self):
        request = FakeRequest()
        payload = {"conversation_id": "conv-history"}

        history_id = entrypoint._history_conversation_id(request, payload)

        self.assertEqual(history_id, "conv-history")
        self.assertEqual(entrypoint._kv_conversation_id(request, history_id), "conv-history")

    def test_pinning_shared_alias_does_not_stop_process_owner(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file:
            configs = {
                "reasoning-base": {
                    "file": model_file.name,
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"high"}'],
                },
                "reasoning-low": {
                    "file": model_file.name,
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"low"}'],
                },
            }

            class FakeRegistry:
                def resolve(self, name):
                    return name if name in configs else None

                def get(self, name):
                    return configs.get(name)

                def get_family_defaults(self, family):
                    return {}

                def get_fixed_gpu(self, name):
                    return None

                def is_fixed(self, name):
                    return False

            class FakeActive:
                name = "reasoning-base"
                cfg = configs[name]
                gpu = 0
                gpus = (0,)
                status = config.MODEL_STATUS_LOADED

                def __init__(self):
                    self.running = True
                    self.stop_calls = 0

                def is_running(self):
                    return self.running

                def stop(self):
                    self.stop_calls += 1
                    self.running = False

            with patch.object(mm_module, "registry", FakeRegistry()):
                manager = entrypoint.ModelManager(gpu_count=2)
                owner = FakeActive()
                manager.active[owner.name] = owner
                self.assertIs(manager.get_active("reasoning-low"), owner)

                active, metadata = asyncio.run(manager.pin_model("reasoning-low", 1))

                self.assertIsNone(active)
                self.assertIsNone(metadata["gpu"])
                self.assertEqual(metadata["pinned_gpu"], 1)
                self.assertEqual(owner.stop_calls, 0)
                self.assertTrue(owner.is_running())
                self.assertEqual(list(manager.active), ["reasoning-base"])

                owner.running = False
                self.assertEqual(manager._incumbents_on(0), [])

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

    def test_module_launch_keeps_single_manager_instance(self):
        env = os.environ.copy()
        pythonpath = str(ROOT / "src")
        if env.get("PYTHONPATH"):
            pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, runpy; import uvicorn; "
                "uvicorn.run = lambda *args, **kwargs: None; "
                "mod = runpy.run_module('grimoire.entrypoint', run_name='__main__', alter_sys=True); "
                "from grimoire.routes.models import _get_manager; "
                "print(json.dumps({'same_manager': _get_manager() is mod['manager']}))",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["same_manager"])

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

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            active = FakeActive()
            manager.active["canonical"] = active
            stopped = asyncio.run(manager.stop_model("alias"))
            self.assertTrue(stopped)
            self.assertTrue(active.stopped)
            self.assertNotIn("canonical", manager.active)
        finally:
            mm_module.registry = old_registry

    def test_start_model_rolls_back_oldest_eviction_when_replacement_fails(self):
        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return {"file": "replacement.gguf"}

            def validate(self, name, gpu_count=None):
                return True, "OK"

            def get_fixed_gpu(self, name):
                return None

            def is_fixed(self, name):
                return False

        class FakeIncumbent:
            def __init__(self):
                self.name = "incumbent"
                self.cfg = {"file": "incumbent.gguf"}
                self.gpu = 0
                self.port = 8001
                self.backend_type = "llama"
                self.started = datetime.now(timezone.utc)
                self.stop_calls = 0
                self.restarted = 0
                self.status = None

            def stop(self):
                self.stop_calls += 1

            def is_running(self):
                return True

        async def fake_start_active_model(active):
            if active.name == "replacement":
                raise RuntimeError("replacement boom")
            active.restarted += 1

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            incumbent = FakeIncumbent()
            manager.active[incumbent.name] = incumbent

            with patch.object(manager, "_start_active_model", side_effect=fake_start_active_model):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(manager.start_model("replacement"))

            self.assertIn("replacement boom", str(cm.exception))
            self.assertIs(manager.active.get("incumbent"), incumbent)
            self.assertNotIn("replacement", manager.active)
            self.assertEqual(incumbent.stop_calls, 1)
            self.assertEqual(incumbent.restarted, 1)
        finally:
            mm_module.registry = old_registry

    def test_start_model_rolls_back_pinned_gpu_eviction_when_replacement_fails(self):
        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return {"file": "replacement.gguf"}

            def validate(self, name, gpu_count=None):
                return True, "OK"

            def get_fixed_gpu(self, name):
                return 0 if name == "replacement" else None

            def is_fixed(self, name):
                return False

        class FakeIncumbent:
            def __init__(self):
                self.name = "incumbent"
                self.cfg = {"file": "incumbent.gguf"}
                self.gpu = 0
                self.port = 8001
                self.backend_type = "llama"
                self.started = datetime.now(timezone.utc)
                self.stop_calls = 0
                self.restarted = 0
                self.status = None

            def stop(self):
                self.stop_calls += 1

            def is_running(self):
                return True

        async def fake_start_active_model(active):
            if active.name == "replacement":
                raise RuntimeError("replacement boom")
            active.restarted += 1

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            incumbent = FakeIncumbent()
            manager.active[incumbent.name] = incumbent

            with patch.object(manager, "_start_active_model", side_effect=fake_start_active_model):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(manager.start_model("replacement"))

            self.assertIn("replacement boom", str(cm.exception))
            self.assertIs(manager.active.get("incumbent"), incumbent)
            self.assertNotIn("replacement", manager.active)
            self.assertEqual(incumbent.stop_calls, 1)
            self.assertEqual(incumbent.restarted, 1)
        finally:
            mm_module.registry = old_registry

    def test_failed_replacement_stop_stays_tracked_as_failed(self):
        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return {"file": "replacement.gguf"}

            def validate(self, name, gpu_count=None):
                return True, "OK"

            def get_fixed_gpu(self, name):
                return None

            def is_fixed(self, name):
                return False

        class FakeIncumbent:
            def __init__(self):
                self.name = "incumbent"
                self.cfg = {"file": "incumbent.gguf"}
                self.gpu = 0
                self.port = 8001
                self.backend_type = "llama"
                self.started = datetime.now(timezone.utc)

            def stop(self):
                return None

            def is_running(self):
                return True

        class FakeReplacement:
            def __init__(self, name, cfg, port, gpu):
                self.name = name
                self.cfg = cfg
                self.port = port
                self.gpu = gpu
                self.backend_type = "llama"
                self.status = None
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                raise RuntimeError("stop boom")

            def is_running(self):
                return True

        async def fake_start_active_model(active):
            raise RuntimeError("replacement boom")

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            manager.active["incumbent"] = FakeIncumbent()

            with patch.object(mm_module, "ActiveModel", FakeReplacement), \
                 patch.object(manager, "_start_active_model", side_effect=fake_start_active_model):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(manager.start_model("replacement"))

            self.assertIn("replacement boom", str(cm.exception))
            self.assertIn("failed to stop replacement: stop boom", str(cm.exception))
            failed = manager.active.get("replacement")
            self.assertIsNotNone(failed)
            self.assertEqual(failed.status, config.MODEL_STATUS_FAILED)
            self.assertEqual(failed.stop_calls, 1)
        finally:
            mm_module.registry = old_registry

    def test_failed_incumbent_cleanup_stays_tracked_when_rollback_restart_fails(self):
        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return {"file": "replacement.gguf"}

            def validate(self, name, gpu_count=None):
                return True, "OK"

            def get_fixed_gpu(self, name):
                return None

            def is_fixed(self, name):
                return False

        class FakeIncumbent:
            def __init__(self):
                self.name = "incumbent"
                self.cfg = {"file": "incumbent.gguf"}
                self.gpu = 0
                self.port = 8001
                self.backend_type = "llama"
                self.started = datetime.now(timezone.utc)
                self.stop_calls = 0
                self.restart_calls = 0
                self.status = None

            def stop(self):
                self.stop_calls += 1
                if self.stop_calls >= 2:
                    raise RuntimeError("incumbent stop boom")

            def is_running(self):
                return True

        async def fake_start_active_model(active):
            if active.name == "replacement":
                raise RuntimeError("replacement boom")
            active.restart_calls += 1
            raise RuntimeError("restart boom")

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            incumbent = FakeIncumbent()
            manager.active[incumbent.name] = incumbent

            with patch.object(manager, "_start_active_model", side_effect=fake_start_active_model):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(manager.start_model("replacement"))

            self.assertIn("replacement boom", str(cm.exception))
            self.assertIn("rollback failed for evicted incumbents", str(cm.exception))
            self.assertIn("failed to stop incumbent: incumbent stop boom", str(cm.exception))
            failed = manager.active.get("incumbent")
            self.assertIs(failed, incumbent)
            self.assertEqual(failed.status, config.MODEL_STATUS_FAILED)
            self.assertEqual(failed.restart_calls, 1)
            self.assertEqual(failed.stop_calls, 2)
        finally:
            mm_module.registry = old_registry

    def test_normal_llama_start_keeps_opt_dflash_out_of_library_path(self):
        captured = {}

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(cmd, env=None, preexec_fn=None):
            captured["cmd"] = cmd
            captured["env"] = dict(env or {})
            captured["preexec_fn"] = preexec_fn
            return FakeProc()

        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, patch.object(mm_module.subprocess, "Popen", side_effect=fake_popen):
            active = mm_module.ActiveModel("qwen-3.6-27B", {"file": model_file.name}, port=8001, gpu=0)
            active._start_llama()

        ld_library_path = captured["env"].get("LD_LIBRARY_PATH", "")
        self.assertIn(mm_module.config.TURBOQUANT_LIB_DIR, ld_library_path)
        self.assertNotIn(mm_module.config.PFLASH_HOME, ld_library_path)
        self.assertNotIn("LD_PRELOAD", captured["env"])

    def test_park_model_still_uses_shim_without_global_opt_dflash_path(self):
        captured = {}

        class FakeProc:
            pid = 12345

            def poll(self):
                return None

        def fake_popen(cmd, env=None, preexec_fn=None):
            captured["env"] = dict(env or {})
            return FakeProc()

        with tempfile.NamedTemporaryFile(suffix=".gguf") as model_file, patch.object(mm_module.subprocess, "Popen", side_effect=fake_popen):
            active = mm_module.ActiveModel(
                "pflash-park-qwen3.6-27B",
                {"file": model_file.name, "park-unpark": True},
                port=8001,
                gpu=0,
            )
            active._start_llama()

        ld_library_path = captured["env"].get("LD_LIBRARY_PATH", "")
        self.assertIn(mm_module.config.TURBOQUANT_LIB_DIR, ld_library_path)
        self.assertNotIn(mm_module.config.PFLASH_HOME, ld_library_path)
        self.assertEqual(captured["env"].get("LD_PRELOAD"), mm_module.config.PFLASH_SHIM_PATH)
        self.assertEqual(captured["env"].get("PFLASH_SHIM_FIFO_BASE"), "/tmp/pflash_shim.pflash-park-qwen3.6-27B")

    def test_pflash_shim_listener_uses_fifo_open_pattern_compatible_with_python_client(self):
        shim = (ROOT / "src" / "grimoire" / "dflash" / "pflash_shim.c").read_text()
        self.assertIn('static char ctl_path[256] = "/tmp/pflash_shim.ctl";', shim)
        self.assertIn('static char ack_path[256] = "/tmp/pflash_shim.ack";', shim)
        self.assertIn('const char *base = getenv("PFLASH_SHIM_FIFO_BASE");', shim)
        self.assertIn('snprintf(ctl_path, sizeof(ctl_path), "%s.ctl", base);', shim)
        self.assertIn('snprintf(ack_path, sizeof(ack_path), "%s.ack", base);', shim)
        self.assertIn('cf = open(ctl_path, O_RDWR);', shim)
        self.assertIn('af = open(ack_path, O_WRONLY);', shim)
        self.assertIn('close(af);', shim)
        self.assertNotIn('af = open(ack_path, O_WRONLY | O_NONBLOCK);', shim)

    def test_llama_proxy_scopes_slot_files_and_serializes_slot_zero_per_model(self):
        llama_proxy = (ROOT / "src" / "grimoire" / "proxy" / "llama.py").read_text()
        self.assertIn('lock = getattr(active, "_pflash_slot_lock", None)', llama_proxy)
        self.assertIn('await slot_guard.acquire()', llama_proxy)
        self.assertIn('slot_guard.release()', llama_proxy)
        self.assertIn('store.kv_filename(hash_bytes)', llama_proxy)

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

    def test_cache_warm_does_not_record_usage(self):
        class FakeUsageStore:
            def __init__(self):
                self.records = []

            def record(self, *args, **kwargs):
                self.records.append((args, kwargs))

        async def stream():
            yield b'data: {"usage":{"prompt_tokens":30,"completion_tokens":0}}\n\n'

        async def consume(async_iter):
            return [chunk async for chunk in async_iter]

        old_usage_store = entrypoint.usage_store
        fake_usage = FakeUsageStore()
        try:
            entrypoint.usage_store = fake_usage
            asyncio.run(consume(entrypoint._record_response_stream(
                stream(),
                user_hash="user",
                conversation_id=None,
                model_name="model",
                model_cfg={"cost": {}},
                payload={},
                record_history=False,
                record_usage=False,
            )))
            self.assertEqual(fake_usage.records, [])
        finally:
            entrypoint.usage_store = old_usage_store

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

    def test_projector_and_multimodal_capability_agree(self):
        """A projector and a multimodal capability have to come as a pair.

        An mmproj on a model that does not declare multimodal loads a projector
        into VRAM that nothing will ever use. A multimodal declaration without
        one advertises image input the backend cannot serve, so requests fail
        only once an image is actually sent.
        """
        data = json.loads((ROOT / "etc" / "models.json").read_text())
        for name, cfg in data["models"].items():
            declares_images = bool({"multimodal", "vision"} & set(cfg.get("capabilities") or []))
            carries_projector = bool(cfg.get("mmproj"))
            self.assertEqual(
                carries_projector,
                declares_images,
                f"{name}: mmproj={carries_projector} but multimodal={declares_images}",
            )

    def test_ingested_model_preserves_source_mmproj(self):
        class FakeRegistry:
            def __init__(self):
                self.added = None
            def get(self, name):
                if name != "gemma-vision":
                    raise AssertionError(name)
                return {
                    "file": "gguf/base.gguf",
                    "mmproj": "gguf/gemma4-mmproj-BF16.gguf",
                    "capabilities": ["completion", "multimodal"],
                    "ctx-size": 120000,
                    "cache-type-k": "turbo4",
                    "cache-type-v": "turbo4",
                }
            def add(self, alias, config):
                self.added = (alias, config)
        fake_registry = FakeRegistry()
        old_registry = models_routes.registry
        try:
            models_routes.registry = fake_registry
            models_routes._ingest_tasks["task-mmproj"] = {
                "alias": "derived-vision",
                "filename": "derived.gguf",
                "load_settings_from": "gemma-vision",
            }
            models_routes._register_model("task-mmproj")
        finally:
            models_routes.registry = old_registry
            models_routes._ingest_tasks.pop("task-mmproj", None)
        alias, config = fake_registry.added
        self.assertEqual(alias, "derived-vision")
        self.assertEqual(config["file"], "gguf/derived.gguf")
        self.assertEqual(config["mmproj"], "gguf/gemma4-mmproj-BF16.gguf")
        self.assertEqual(config["capabilities"], ["completion", "multimodal"])
    def test_llama_side_pflash_startup_fails_closed_when_daemon_boot_fails(self):
        class FakeRegistry:
            def resolve(self, name):
                return name

            def get(self, name):
                return {
                    "file": "target.gguf",
                    "pflash": True,
                    "drafter": "drafter.gguf",
                }

            def validate(self, name, gpu_count=None):
                return True, "OK"

            def get_fixed_gpu(self, name):
                return None

            def is_fixed(self, name):
                return False

        old_registry = mm_module.registry
        try:
            mm_module.registry = FakeRegistry()
            manager = entrypoint.ModelManager(gpu_count=1)
            with patch.object(mm_module, "resolve_path", side_effect=lambda cfg, key: f"/tmp/{key}.gguf"), \
                 patch.object(mm_module.ActiveModel, "_start_pflash_daemon", side_effect=RuntimeError("daemon boom")), \
                 patch.object(mm_module.ActiveModel, "start") as start_backend:
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(manager.start_model("pflash-qwen3.6-27B"))
            self.assertIn("daemon boom", str(cm.exception))
            self.assertFalse(start_backend.called)
            self.assertEqual(manager.active, {})
        finally:
            mm_module.registry = old_registry

    def test_llama_registry_validation_requires_pflash_drafter_and_park_shim(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "models.json"
            model_path = Path(tmp) / "target.gguf"
            model_path.write_text("x")
            registry = ModelRegistry(path=str(state_path), seed_path=None)
            registry.add(
                "pflash-test",
                {
                    "file": model_path.name,
                    "pflash": True,
                    "drafter": "missing-drafter.gguf",
                },
            )

            import grimoire.registry as registry_mod
            old_models_dir = registry_mod.MODELS_DIR
            old_shim = config.PFLASH_SHIM_PATH
            try:
                registry_mod.MODELS_DIR = tmp
                valid, reason = registry.validate("pflash-test")
                self.assertFalse(valid)
                self.assertIn("PFlash drafter", reason)

                drafter = Path(tmp) / "drafter.gguf"
                drafter.write_text("x")
                registry.update("pflash-test", {"drafter": drafter.name, "park-unpark": True})
                config.PFLASH_SHIM_PATH = str(Path(tmp) / "missing-shim.so")
                valid, reason = registry.validate("pflash-test")
                self.assertFalse(valid)
                self.assertIn("park-unpark shim", reason)
            finally:
                registry_mod.MODELS_DIR = old_models_dir
                config.PFLASH_SHIM_PATH = old_shim

    def test_pflash_build_stage_builds_only_pflash_daemon(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertNotIn("-DDFLASH27B_TESTS=ON", dockerfile)
        self.assertIn("--target pflash_daemon", dockerfile)
        self.assertNotIn("test_dflash", dockerfile)
        self.assertIn("/opt/pflash/pflash_daemon", dockerfile)
        self.assertIn("/opt/pflash/pflash_shim.so", dockerfile)

    def test_preserved_dflash_binaries_and_lib_dir_are_individually_configurable(self):
        config_src = (ROOT / "src" / "grimoire" / "config.py").read_text()
        daemon_src = (ROOT / "src" / "grimoire" / "dflash" / "daemon.py").read_text()

        self.assertIn('PFLASH_LIB_DIR = os.environ.get("GRIMOIRE_PFLASH_LIB_DIR", PFLASH_HOME)', config_src)
        self.assertIn('PFLASH_DAEMON_BIN = os.environ.get("GRIMOIRE_PFLASH_DAEMON_BIN", os.path.join(PFLASH_HOME, "pflash_daemon"))', config_src)
        self.assertIn('PFLASH_SHIM_PATH = os.environ.get("GRIMOIRE_PFLASH_SHIM_PATH", os.path.join(PFLASH_HOME, "pflash_shim.so"))', config_src)
        self.assertIn('config.PFLASH_DAEMON_BIN', daemon_src)
        self.assertIn('config.PFLASH_LIB_DIR', daemon_src)


if __name__ == "__main__":
    unittest.main()
