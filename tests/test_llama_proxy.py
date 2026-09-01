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
