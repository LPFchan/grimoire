import asyncio
import sys
import tempfile
import unittest
from fastapi import HTTPException
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import grimoire.proxy.llama as llama_proxy


class _FakeTokenizer:
    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


class _FakeActive:
    def __init__(self):
        self.name = "test-model"
        self.cfg = {
            "family": "qwen",
            "ctx-size": 4096,
        }
        self.backend_type = "llama"
        self.port = 8001
        self.gpu = 0
        self._park_calls = 0
        self._unpark_calls = 0

    def get_tokenizer(self):
        return _FakeTokenizer()

    async def get_backend_model_id(self):
        return self.name

    def _park_llama(self):
        self._park_calls += 1
        return True

    def _unpark_llama(self):
        self._unpark_calls += 1
        return True


class _FakeUpstream:
    status_code = 200
    headers = {"content-type": "application/json"}

    async def aiter_raw(self):
        yield b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}'

    async def aclose(self):
        return None


class _FakeClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.requests = []
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def build_request(self, method, url, headers=None, json=None):
        return {"method": method, "url": url, "headers": headers or {}, "json": json}

    async def send(self, request, stream=False):
        self.requests.append((request, stream))
        return _FakeUpstream()

    async def post(self, url, json=None, timeout=None):
        self.requests.append(({"method": "POST", "url": url, "json": json}, False))
        return type("Resp", (), {"status_code": 200})()

    async def aclose(self):
        return None


class LlamaProxyTests(unittest.TestCase):
    def test_chat_template_defaults_merge_and_request_wins(self):
        payload = {"chat_template_kwargs": {"reasoning_effort": "medium"}}
        cfg = {
            "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":"low"}']
        }
        family = {
            "extra-args": ["--chat-template-kwargs", '{"preserve_thinking":true}']
        }

        result = llama_proxy.apply_chat_template_kwargs(payload, cfg, family)

        self.assertEqual(
            result["chat_template_kwargs"],
            {"preserve_thinking": True, "reasoning_effort": "medium"},
        )

    async def test_shared_process_uses_requested_alias_config(self):
        active = _FakeActive()
        active.name = "muse-glimmer-30b-low"
        active.cfg = {"family": "muse", "ctx-size": 4096}
        active.prefill_config.enabled = False
        requested_cfg = {
            "family": "muse",
            "ctx-size": 8192,
            "extra-args": ["--chat-template-kwargs", '{"reasoning_strength":"high"}'],
        }
        plugin_calls = []

        def before_request(payload, model_name, model_cfg):
            plugin_calls.append((model_name, model_cfg))
            return payload

        with patch.object(llama_proxy.registry, "resolve", return_value="muse-glimmer-30b-high"), \
             patch.object(llama_proxy.registry, "get", return_value=requested_cfg), \
             patch.object(llama_proxy.registry, "get_family_defaults", return_value={}), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=before_request), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=lambda p, *a: p), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda s, *a: s), \
             patch.object(llama_proxy, "get_proxy_client", _FakeClient):
            response = await llama_proxy._proxy_chat(
                "muse-glimmer-30b-high",
                {"messages": [{"role": "user", "content": "ping"}], "stream": False},
                active,
            )
            async for _ in response.body_iterator:
                pass

        request = _FakeClient.instances[-1].requests[0][0]["json"]
        self.assertEqual(request["chat_template_kwargs"], {"reasoning_strength": "high"})
        self.assertEqual(plugin_calls, [("muse-glimmer-30b-high", requested_cfg)])

    async def test_conversation_cache_key_is_scoped_by_process_and_user(self):
        active = _FakeActive()
        active.prefill_config = None

        class FakeStore:
            def __init__(self):
                self.saved = []
                self.restored = []

            async def save_conv(self, client, slot_url, conversation_id):
                self.saved.append(conversation_id)

            async def restore_conv(self, client, slot_url, conversation_id):
                self.restored.append(conversation_id)

        store = FakeStore()
        payload = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }

        with patch.object(llama_proxy, "_kv_store", return_value=store), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda p, *a: p), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=lambda p, *a: p), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda s, *a: s), \
             patch.object(llama_proxy, "get_proxy_client", _FakeClient):
            first = await llama_proxy._proxy_chat(
                active.name, payload, active, user_hash="user-a", conversation_id="conversation"
            )
            async for _ in first.body_iterator:
                pass
            second = await llama_proxy._proxy_chat(
                active.name, payload, active, user_hash="user-b", conversation_id="conversation"
            )
            async for _ in second.body_iterator:
                pass

        first_key = f"{active.name}\0user-a\0conversation"
        second_key = f"{active.name}\0user-b\0conversation"
        self.assertEqual(store.restored, [first_key, second_key])
        self.assertEqual(store.saved, [first_key])

    def test_model_logit_bias_merges_cli_style_defaults(self):
        payload = {"logit_bias": {"262143": 1.0, "5": -2}}
        cfg = {"logit-bias": ["262143+5", "111038-inf", [7, 1.5]]}
        result = llama_proxy._apply_model_logit_bias(payload.copy(), cfg)
        self.assertEqual(result["logit_bias"], {"262143": 1.0, "111038": -100.0, "7": 1.5, "5": -2.0})

    def test_request_logit_bias_overrides_cli_logit_bias(self):
        payload = {"messages": []}
        cfg = {
            "logit-bias": ["262143+5"],
            "request-logit-bias": {262143: 3, 111038: -6},
        }
        result = llama_proxy._apply_model_logit_bias(payload.copy(), cfg)
        self.assertEqual(result["logit_bias"], {"262143": 3.0, "111038": -6.0})

    def test_kv_filename_format(self):
        from grimoire.cache.kv_cache_store import KVCacheStore, KV_PREFIX, KV_SUFFIX
        store = KVCacheStore()
        h = bytes(16)
        name = store.kv_filename(h)
        self.assertTrue(name.startswith(KV_PREFIX))
        self.assertTrue(name.endswith(KV_SUFFIX))
        self.assertEqual(len(name), len(KV_PREFIX) + 16 + len(KV_SUFFIX))
        self.assertNotIn("/", name)

    def test_slot_lock_is_scoped_per_active_model(self):
        active_a = _FakeActive()
        active_b = _FakeActive()
        active_b.name = "other-model"

        self.assertIs(llama_proxy._slot_lock(active_a), llama_proxy._slot_lock(active_a))
        self.assertIsNot(llama_proxy._slot_lock(active_a), llama_proxy._slot_lock(active_b))


if __name__ == "__main__":
    unittest.main()
