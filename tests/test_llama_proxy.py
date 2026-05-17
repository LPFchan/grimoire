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


class _FakePflashDaemon:
    def is_running(self):
        return True


class _FakeActive:
    def __init__(self):
        self.name = "pflash-qwen3.6-27B"
        self.cfg = {
            "family": "qwen",
            "prefill-threshold": 1,
            "park-unpark": True,
            "ctx-size": 4096,
            "drafter": "gguf/Qwen3.5-0.8B-Q8_0.gguf",
        }
        self.backend_type = "llama"
        self.port = 8001
        self.gpu = 0
        self.pflash_daemon = _FakePflashDaemon()
        self.prefill_config = type(
            "PrefillCfg",
            (),
            {"enabled": True, "threshold": 1, "keep_ratio": 0.1, "drafter_path": "/tmp/drafter.gguf", "tail_budget": 128},
        )()
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


class LlamaProxyPflashTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeClient.instances.clear()

    async def test_park_model_unparks_when_compression_raises(self):
        active = _FakeActive()
        payload = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "conversation_id": "conv-1",
        }

        async def fake_before_backend_request(payload, model_name, model_cfg, backend_model_id, client, url, headers):
            return payload

        async def raising_compress(prompt_ids, daemon, config, blocks=None):
            raise RuntimeError("compress failed")

        with patch.object(llama_proxy, "_prompt_layout_from_messages", return_value=([1, 2, 3], [])), \
             patch.object(llama_proxy, "maybe_compress", side_effect=raising_compress), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda payload, *_: payload), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=fake_before_backend_request), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda stream, *_: stream), \
             patch.object(llama_proxy.httpx, "AsyncClient", _FakeClient):
            with self.assertRaises(HTTPException) as cm:
                await llama_proxy._proxy_chat(active.name, payload, active, user_hash=None, conversation_id="conv-1")

        self.assertEqual(active._park_calls, 1)
        self.assertEqual(active._unpark_calls, 1)
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("pflash compression failed", cm.exception.detail)

    async def test_slot_restore_and_save_use_validated_conversation_id(self):
        active = _FakeActive()
        expected_name = llama_proxy._slot_save_key(active, "validated-conv-id")
        payload = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "conversation_id": "../../raw-payload-id",
        }

        async def fake_before_backend_request(payload, model_name, model_cfg, backend_model_id, client, url, headers):
            return payload

        async def fake_compress(prompt_ids, daemon, config, blocks=None):
            return prompt_ids, False, blocks or []

        with patch.object(llama_proxy, "_prompt_layout_from_messages", return_value=([1, 2, 3], [])), \
             patch.object(llama_proxy, "maybe_compress", side_effect=fake_compress), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda payload, *_: payload), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=fake_before_backend_request), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda stream, *_: stream), \
             patch.object(llama_proxy, "_has_saved_kv", return_value=False), \
             patch.object(llama_proxy.httpx, "AsyncClient", _FakeClient):
            response = await llama_proxy._proxy_chat(
                active.name,
                payload,
                active,
                user_hash=None,
                conversation_id="validated-conv-id",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertIn("ok", b"".join(chunks).decode())
        requests = [req for client in _FakeClient.instances for req, _ in client.requests]
        slot_calls = [req for req in requests if req["method"] == "POST" and "/slots/0?action=" in req["url"]]
        self.assertEqual(len(slot_calls), 2)
        self.assertIn("action=restore", slot_calls[0]["url"])
        self.assertIn("action=save", slot_calls[1]["url"])
        self.assertEqual(slot_calls[0]["json"]["filename"], expected_name)
        self.assertEqual(slot_calls[1]["json"]["filename"], expected_name)

    async def test_restore_success_saves_even_without_compression(self):
        active = _FakeActive()
        expected_name = llama_proxy._slot_save_key(active, "conv-1")
        payload = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "conversation_id": "raw-conversation-id",
        }

        async def fake_before_backend_request(payload, model_name, model_cfg, backend_model_id, client, url, headers):
            return payload

        async def fake_compress(prompt_ids, daemon, config, blocks=None):
            return prompt_ids, False, blocks or []

        with patch.object(llama_proxy, "_prompt_layout_from_messages", return_value=([1, 2, 3], [])), \
             patch.object(llama_proxy, "maybe_compress", side_effect=fake_compress), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda payload, *_: payload), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=fake_before_backend_request), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda stream, *_: stream), \
             patch.object(llama_proxy.httpx, "AsyncClient", _FakeClient):
            response = await llama_proxy._proxy_chat(
                active.name,
                payload,
                active,
                user_hash=None,
                conversation_id="conv-1",
            )
            async for _ in response.body_iterator:
                pass

        requests = [req for client in _FakeClient.instances for req, _ in client.requests]
        slot_calls = [req for req in requests if req["method"] == "POST" and "/slots/0?action=" in req["url"]]
        self.assertEqual(len(slot_calls), 2)
        self.assertIn("action=restore", slot_calls[0]["url"])
        self.assertIn("action=save", slot_calls[1]["url"])
        self.assertEqual(slot_calls[0]["json"]["filename"], expected_name)
        self.assertEqual(slot_calls[1]["json"]["filename"], expected_name)

    async def test_first_turn_without_compression_still_saves_model_scoped_kv(self):
        active = _FakeActive()
        active.prefill_config.threshold = 1000
        expected_name = llama_proxy._slot_save_key(active, "conv-2")
        payload = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "conversation_id": "conv-2",
        }

        async def fake_before_backend_request(payload, model_name, model_cfg, backend_model_id, client, url, headers):
            return payload

        with patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda payload, *_: payload), \
             patch.object(llama_proxy, "_prompt_layout_from_messages", return_value=([1], [])), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=fake_before_backend_request), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda stream, *_: stream), \
             patch.object(llama_proxy.httpx, "AsyncClient", _FakeClient):
            response = await llama_proxy._proxy_chat(
                active.name,
                payload,
                active,
                user_hash=None,
                conversation_id="conv-2",
            )
            async for _ in response.body_iterator:
                pass

        requests = [req for client in _FakeClient.instances for req, _ in client.requests]
        slot_calls = [req for req in requests if req["method"] == "POST" and "/slots/0?action=" in req["url"]]
        self.assertEqual(len(slot_calls), 2)
        self.assertIn("action=restore", slot_calls[0]["url"])
        self.assertIn("action=save", slot_calls[1]["url"])
        self.assertEqual(slot_calls[1]["json"]["filename"], expected_name)

    async def test_slot_restore_and_save_are_serialized_per_model(self):
        active = _FakeActive()
        active.prefill_config.threshold = 1000
        request_log = []
        first_stream_entered = asyncio.Event()
        allow_first_finish = asyncio.Event()

        async def fake_before_backend_request(payload, model_name, model_cfg, backend_model_id, client, url, headers):
            return payload

        async def consume(response):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return b"".join(chunks)

        class SerialUpstream:
            status_code = 200
            headers = {"content-type": "application/json"}

            def __init__(self, conversation_id):
                self.conversation_id = conversation_id

            async def aiter_raw(self):
                request_log.append(("stream", self.conversation_id))
                if self.conversation_id == "conv-a":
                    first_stream_entered.set()
                    await allow_first_finish.wait()
                yield b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}'

            async def aclose(self):
                return None

        class SerialClient(_FakeClient):
            async def send(self, request, stream=False):
                self.requests.append((request, stream))
                request_log.append(("send", request["json"]["conversation_id"]))
                return SerialUpstream(request["json"]["conversation_id"])

            async def post(self, url, json=None, timeout=None):
                self.requests.append(({"method": "POST", "url": url, "json": json}, False))
                action = "restore" if "action=restore" in url else "save"
                request_log.append((action, json["filename"]))
                return type("Resp", (), {"status_code": 200})()

        payload_a = {
            "model": active.name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "conversation_id": "conv-a",
        }
        payload_b = {
            "model": active.name,
            "messages": [{"role": "user", "content": "pong"}],
            "stream": False,
            "conversation_id": "conv-b",
        }
        filename_a = llama_proxy._slot_save_key(active, "conv-a")
        filename_b = llama_proxy._slot_save_key(active, "conv-b")

        with patch.object(llama_proxy, "_prompt_layout_from_messages", return_value=([1], [])), \
             patch.object(llama_proxy.plugin_manager, "before_request", side_effect=lambda payload, *_: payload), \
             patch.object(llama_proxy.plugin_manager, "before_backend_request", side_effect=fake_before_backend_request), \
             patch.object(llama_proxy.plugin_manager, "wrap_response_stream", side_effect=lambda stream, *_: stream), \
             patch.object(llama_proxy.httpx, "AsyncClient", SerialClient):
            response_a = await llama_proxy._proxy_chat(
                active.name,
                payload_a,
                active,
                user_hash=None,
                conversation_id="conv-a",
            )
            first_consume = asyncio.create_task(consume(response_a))
            await first_stream_entered.wait()

            response_b_task = asyncio.create_task(
                llama_proxy._proxy_chat(
                    active.name,
                    payload_b,
                    active,
                    user_hash=None,
                    conversation_id="conv-b",
                )
            )
            await asyncio.sleep(0)

            self.assertFalse(response_b_task.done())
            self.assertEqual([event for event in request_log if event[0] == "restore"], [("restore", filename_a)])

            allow_first_finish.set()
            self.assertIn("ok", (await first_consume).decode())

            response_b = await response_b_task
            self.assertIn("ok", (await consume(response_b)).decode())

        self.assertLess(request_log.index(("restore", filename_a)), request_log.index(("save", filename_a)))
        self.assertLess(request_log.index(("save", filename_a)), request_log.index(("restore", filename_b)))
        self.assertLess(request_log.index(("restore", filename_b)), request_log.index(("save", filename_b)))

    def test_kv_filename_sanitizes_unsafe_conversation_ids(self):
        filename = llama_proxy._kv_filename("pflash-qwen3.6-27B", "../../bad/id?name=1")
        self.assertTrue(filename.startswith("pflash-"))
        self.assertIn("-bad_id_name_1-", filename)
        self.assertTrue(filename.endswith(".kv"))
        self.assertNotIn("/", filename)
        self.assertNotIn("..", filename)

    def test_slot_lock_is_scoped_per_active_model(self):
        active_a = _FakeActive()
        active_b = _FakeActive()
        active_b.name = "pflash-park-qwen3.6-27B"

        self.assertIs(llama_proxy._slot_lock(active_a), llama_proxy._slot_lock(active_a))
        self.assertIsNot(llama_proxy._slot_lock(active_a), llama_proxy._slot_lock(active_b))


if __name__ == "__main__":
    unittest.main()
