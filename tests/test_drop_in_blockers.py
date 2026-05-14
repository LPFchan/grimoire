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
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))

from fastapi import HTTPException

import grimoire.config as config
import grimoire.entrypoint as entrypoint
import grimoire.model_manager as mm_module
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
        self.assertNotIn(mm_module.config.DFLASH_HOME, ld_library_path)
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
        self.assertNotIn(mm_module.config.DFLASH_HOME, ld_library_path)
        self.assertEqual(captured["env"].get("LD_PRELOAD"), mm_module.config.PFLASH_SHIM_PATH)

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

    def test_models_json_contains_dormant_native_dflash_canary(self):
        data = json.loads((ROOT / "etc" / "models.json").read_text())
        cfg = data["models"]["dflash-native-qwen3.6-27B-canary"]
        self.assertEqual(cfg["speculative-type"], "dflash")
        self.assertEqual(cfg["spec-dflash-cross-ctx"], 1024)
        self.assertEqual(cfg["draft"], "gguf/Qwen3-0.6B-BF16.gguf")

    def test_phase0_checklist_includes_behavior_matrix_and_inventory(self):
        checklist = (ROOT / "MIGRATION_EXECUTION_CHECKLIST.md").read_text()
        self.assertIn("## Phase 0 Status Artifacts", checklist)
        self.assertIn("### Required-vs-Retired Behavior Matrix", checklist)
        self.assertIn("### Served Model Inventory And Harness Mapping", checklist)
        self.assertIn("dflash-native-qwen3.6-27B-canary", checklist)
        self.assertIn("Stale alias `dflash-pflash-qwen-27B` is not a valid current registry entry", checklist)

    def test_harness_defaults_match_current_registry_aliases(self):
        e2e = (ROOT / "tests" / "test_e2e_smoke.py").read_text()
        stress = (ROOT / "tests" / "test_stress_dflash.py").read_text()
        pflash = (ROOT / "tests" / "test_pflash_pipeline.py").read_text()
        tune_ctx = (ROOT / "tests" / "tune_ctx.py").read_text()
        pflash_ctx_tune = (ROOT / "tests" / "test_pflash_ctx_tune.py").read_text()

        self.assertIn('DFLASH_SMOKE_MODEL = os.environ.get("GRIMOIRE_DFLASH_SMOKE_MODEL", "dflash-pflash-qwen3.6-27B")', e2e)
        self.assertIn('LLAMA_SMOKE_MODEL = os.environ.get("GRIMOIRE_LLAMA_SMOKE_MODEL", "qwen-3.6-27B")', e2e)
        self.assertIn('MODEL = DFLASH_SMOKE_MODEL', e2e)
        self.assertIn('MODEL = LLAMA_SMOKE_MODEL', e2e)
        self.assertIn('MODEL = os.environ.get("STRESS_MODEL", "dflash-pflash-qwen3.6-27B")', stress)
        self.assertIn('MODEL = os.environ.get("MODEL", "pflash-qwen3.6-27B")', pflash)
        self.assertIn('MODEL = "pflash-qwen3.6-27B"', tune_ctx)
        self.assertIn('MODEL = "pflash-qwen3.6-27B"', pflash_ctx_tune)
        self.assertNotIn("dflash-pflash-qwen-27B", e2e)
        self.assertNotIn("dflash-pflash-qwen-27B", stress)
        self.assertNotIn("dflash-pflash-qwen-27B", pflash)
        self.assertNotIn("pflash-qwen-27B", tune_ctx)
        self.assertNotIn("pflash-qwen-27B", pflash_ctx_tune)

    def test_text_only_served_pflash_models_have_no_mmproj(self):
        data = json.loads((ROOT / "etc" / "models.json").read_text())
        for name in ("dflash-pflash-qwen3.6-27B", "pflash-qwen3.6-27B", "pflash-park-qwen3.6-27B", "dflash-native-qwen3.6-27B-canary"):
            cfg = data["models"][name]
            self.assertEqual(cfg.get("capabilities"), ["completion"], name)
            self.assertNotIn("mmproj", cfg, name)

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

    def test_native_dflash_patch_is_copied_and_build_patches_reapply_on_fresh_clone(self):
        patch_path = ROOT / "patches" / "spec-dflash-contract.patch"
        self.assertTrue(patch_path.exists(), "native dflash contract patch file is missing")
        content = patch_path.read_text()
        self.assertIn("--spec-draft-model", content)
        self.assertIn("--spec-dflash-cross-ctx", content)
        self.assertIn('COMMON_SPECULATIVE_TYPE_DFLASH', content)
        self.assertIn('LLM_ARCH_DFLASH_DRAFT', content)
        self.assertIn('"dflash-draft"', content)
        self.assertIn('LLM_KV_DFLASH_BLOCK_SIZE', content)
        self.assertIn('LLAMA_API int32_t llama_model_dflash_block_size', content)
        self.assertIn('LLAMA_API int32_t llama_model_dflash_target_layer_ids', content)
        self.assertIn('LLAMA_API void llama_model_share_tensors', content)
        self.assertIn('LLAMA_API void llama_set_dflash_n_slots(struct llama_context * ctx, int n);', content)
        self.assertIn('LLAMA_API void llama_set_dflash_verify_logits(struct llama_context * ctx, bool enabled, int top_k);', content)
        self.assertIn('LLAMA_API void llama_set_dflash_consume_reduced(struct llama_context * ctx, bool enabled);', content)
        self.assertIn('int32_t dflash_n_slots;', content)
        self.assertIn('int32_t dflash_cross_ctx;', content)
        self.assertIn('LLAMA_DFLASH_MAX_SLOTS    = 8', content)
        self.assertIn('cparams.dflash_cross_ctx = params.speculative.dflash_cross_ctx;', content)
        self.assertIn('int dflash_cross_ctx = LLAMA_DFLASH_PER_SLOT_CTX;', content)
        self.assertIn('int dflash_n_slots = 1;', content)
        self.assertIn('bool dflash_verify_logits = false;', content)
        self.assertIn('int  dflash_verify_topk = 1;', content)
        self.assertIn('bool dflash_reduced_consumer_active = false;', content)
        self.assertIn('void set_dflash_verify_logits(bool enabled, int top_k);', content)
        self.assertIn('void set_dflash_consume_reduced(bool enabled);', content)
        self.assertIn('GGML_API struct ggml_tensor * ggml_argmax_ext(', content)
        self.assertIn('GGML_API struct ggml_tensor * ggml_topk_ext(', content)
        self.assertIn('void llama_context::set_dflash_n_slots(int n) {', content)
        self.assertIn('void llama_context::set_dflash_verify_logits(bool enabled, int top_k) {', content)
        self.assertIn('void llama_context::set_dflash_consume_reduced(bool enabled) {', content)
        self.assertIn('const int clamped_top_k = std::max(1, std::min(top_k, 64));', content)
        self.assertIn('cparams.dflash_verify_logits = enabled;', content)
        self.assertIn('cparams.dflash_verify_topk = clamped_top_k;', content)
        self.assertIn('cparams.dflash_reduced_consumer_active = enabled;', content)
        self.assertIn('struct ggml_tensor * ggml_argmax_ext(', content)
        self.assertIn('struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * a->ne[1]);', content)
        self.assertIn('struct ggml_tensor * ggml_topk_ext(', content)
        self.assertIn('GGML_ASSERT(k >= 1 && k <= 64);', content)
        self.assertIn('struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * k * a->ne[1]);', content)
        self.assertIn('ggml_set_op_params_i32(result, 3, (int32_t) k);', content)
        self.assertIn('LLAMA_API int32_t * llama_get_logits_argmax(struct llama_context * ctx);', content)
        self.assertIn('LLAMA_API float *   llama_get_logits_argmax_probs_ith(struct llama_context * ctx, int32_t i);', content)
        self.assertIn('ggml_tensor * t_logits_argmax = nullptr;', content)
        self.assertIn('int32_t * get_logits_argmax_ith(int32_t i);', content)
        self.assertIn('std::vector<int32_t> logits_argmax_buf;', content)
        self.assertIn('std::vector<float>   logits_argmax_prob_buf;', content)
        self.assertIn('int32_t * llama_context::get_logits_argmax() {', content)
        self.assertIn('float * llama_context::get_logits_argmax_probs_ith(int32_t i) {', content)
        self.assertIn('auto * t_argmax = res->t_logits_argmax;', content)
        self.assertIn('logits_argmax_count = n_outputs_prev + n_outputs;', content)
        self.assertIn('const bool dflash_reduced_consumed = t_argmax != nullptr && cparams.dflash_reduced_consumer_active;', content)
        self.assertIn('if (logits.data && t_logits && n_outputs > 0 && !dflash_reduced_consumed && needs_raw_logits(ubatch, sampling.samplers)) {', content)
        self.assertIn('std::swap(logits_argmax_buf[i0*logits_argmax_k + k], logits_argmax_buf[i1*logits_argmax_k + k]);', content)
        self.assertIn('int32_t * llama_get_logits_argmax(llama_context * ctx) {', content)
        self.assertIn('float * llama_get_logits_argmax_probs(llama_context * ctx) {', content)
        self.assertIn('void llama_set_dflash_consume_reduced(llama_context * ctx, bool enabled) {', content)
        self.assertIn('if (cparams.dflash_verify_logits) {', content)
        self.assertIn('const int topk = std::max(1, std::min(cparams.dflash_verify_topk, 64));', content)
        self.assertIn('res->t_logits_argmax = ggml_topk_ext(ctx0, cur, topk, 0.0f, 0);', content)
        self.assertIn('res->t_logits_argmax = ggml_argmax_ext(ctx0, cur, 0.0f, 0);', content)
        self.assertIn('ggml_build_forward_expand(gf, res->t_logits_argmax);', content)
        self.assertIn('bool common_sampler_supports_reduced(struct common_sampler * gsmpl);', content)
        self.assertIn('bool common_sampler_blocks_speculative(const struct common_sampler * gsmpl);', content)
        self.assertIn('std::vector<llama_token> common_sampler_sample_reduced_and_accept_n(', content)
        self.assertIn('return common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_FORCING;', content)
        self.assertIn('if (common_sampler_blocks_speculative(gsmpl)) {', content)
        self.assertIn('static dflash_reduced_verify_plan dflash_select_reduced_verify_plan(', content)
        self.assertIn('static bool dflash_batch_view_is_reduced_verify(', content)
        self.assertIn('static std::vector<llama_token> dflash_sample_reduced_verify(', content)
        self.assertIn('llama_set_dflash_consume_reduced(ctx, dflash_reduce_this_view);', content)
        self.assertIn('accepted = dflash_sample_reduced_verify(', content)
        self.assertIn('GGML_ABORT("DFlash reduced verifier output missing; raw logits were not copied for this decode view\\n");', content)
        self.assertIn('llama_set_dflash_consume_reduced(ctx, false);', content)
        self.assertIn('cparams.dflash_n_slots = std::clamp(params.dflash_n_slots <= 0 ? 1 : params.dflash_n_slots,', content)
        self.assertIn('sched_need_reserve = true;', content)
        self.assertIn('gf_res_prev->reset();', content)
        self.assertIn('void llama_set_dflash_verify_logits(llama_context * ctx, bool enabled, int top_k) {', content)
        self.assertIn('void llama_set_dflash_n_slots(llama_context * ctx, int n) {', content)
        self.assertIn('cparams.dflash_verify_logits == other.cparams.dflash_verify_logits &&', content)
        self.assertIn('cparams.dflash_verify_topk   == other.cparams.dflash_verify_topk &&', content)
        self.assertIn('/*.dflash_cross_ctx            =*/ LLAMA_DFLASH_PER_SLOT_CTX', content)
        self.assertIn('return model->arch == LLM_ARCH_DFLASH_DRAFT ? (int32_t) model->hparams.dflash_block_size : 0;', content)
        self.assertIn('dst->tok_embd = src->tok_embd;', content)
        self.assertIn('if (llama_model_dflash_block_size(model_dft.get()) > 0 &&', content)
        self.assertIn('auto-detected DFlash drafter (block_size=%d)', content)
        self.assertIn('struct llm_build_dflash_draft : public llm_graph_context {', content)
        self.assertIn('llm_build_dflash_draft(const llama_model & model, const llm_graph_params & params);', content)
        self.assertIn('void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id) {', content)
        self.assertIn('llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);', content)
        self.assertIn('llama_set_cross_data_seq(ctx_dft, seq_id, history.data(), n_target_features, cross_len);', content)
        self.assertIn('common_speculative_set_seq_id(spec, 0);', content)
        self.assertIn('common_speculative_set_seq_id(slot.spec.get(), slot.id);', content)
        self.assertIn('llama_model_share_tensors(model_dft.get(), model_tgt);', content)
        self.assertIn('llama_model_share_tensors(model_dft.get(), llama_get_model(ctx));', content)
        self.assertIn('DFlash draft models are not supported by this example; use speculative-simple or server', content)
        self.assertNotIn('DFlash launch contract recognized, but this build does not yet include a native DFlash implementation', content)
        self.assertNotIn('native implementation not yet ported', content)
        self.assertIn('@ModelBase.register("DFlashDraftModel")', content)
        self.assertIn('model_arch = gguf.MODEL_ARCH.DFLASH_DRAFT', content)
        self.assertIn('self.gguf_writer.add_uint32(f"{arch}.dflash.block_size", block_size)', content)
        self.assertIn('self.gguf_writer.add_array(f"{arch}.dflash.target_layer_ids", target_layer_ids)', content)
        self.assertIn('MODEL_ARCH.DFLASH_DRAFT:     "dflash-draft"', content)
        self.assertIn('MODEL_TENSOR.DFLASH_FC:                 "dflash_fc"', content)
        self.assertIn('"fc",                  # dflash drafter', content)

        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("COPY patches/spec-dflash-contract.patch /app/patches/", dockerfile)
        self.assertIn("patch_hash_file=/app/.cache/llama-cpp-build/.patch_hash", dockerfile)
        self.assertIn("need_patches=1", dockerfile)
        self.assertIn("printf '%s' \"$patch_hash\" > \"$patch_hash_file\"", dockerfile)

    def test_webui_patches_reapply_on_fresh_clone_and_patch_change(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("webui_patch_hash_file=/cache/webui-src/.patch_hash", dockerfile)
        self.assertIn("need_webui_patches=1", dockerfile)
        self.assertIn("printf '%s' \"$webui_patch_hash\" > \"$webui_patch_hash_file\"", dockerfile)

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

    def test_dflash_runtime_scopes_opt_dflash_to_preserved_components(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("-DDFLASH27B_TESTS=ON", dockerfile)
        self.assertIn("--target test_dflash", dockerfile)
        self.assertIn("--target pflash_daemon", dockerfile)
        self.assertIn("/app/.cache/dflash-build/build/test_dflash /opt/dflash/dflash", dockerfile)
        self.assertIn("/app/.cache/dflash-build/build/pflash_daemon /opt/dflash/pflash_daemon", dockerfile)
        self.assertIn("LD_LIBRARY_PATH=/opt/grimoire-llama-cpp/lib:/opt/grimoire-llama-cpp/lib64", dockerfile)
        self.assertNotIn("LD_LIBRARY_PATH=/opt/dflash:/opt/grimoire-llama-cpp/lib:/opt/grimoire-llama-cpp/lib64", dockerfile)


if __name__ == "__main__":
    unittest.main()
