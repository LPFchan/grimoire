import sys
import tempfile
import unittest
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
    def __init__(self, *args, **kwargs):
        self.requests = []

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
            response = await llama_proxy._proxy_chat(active.name, payload, active, user_hash=None, conversation_id="conv-1")
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        self.assertEqual(active._park_calls, 1)
        self.assertEqual(active._unpark_calls, 1)
        body = b"".join(chunks).decode()
        self.assertIn("ok", body)


if __name__ == "__main__":
    unittest.main()
