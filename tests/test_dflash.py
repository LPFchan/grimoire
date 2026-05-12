"""Tests for the dflash backend (registry, prefix cache, helpers, daemon, proxy)."""

import asyncio
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-dflash-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-dflash-usage.sqlite3"))
os.environ.setdefault("GRIMOIRE_REGISTRY_SEED_PATH", str(ROOT / "etc" / "models.json"))
os.environ.setdefault("GRIMOIRE_REGISTRY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-dflash-registry.json"))

from fastapi.testclient import TestClient

import grimoire.entrypoint as entrypoint
from grimoire import registry as registry_mod
from grimoire.dflash.daemon import DflashDaemon
from grimoire.dflash.prefix_cache import PrefixCache
from grimoire.dflash.session_kv import SessionKV
from grimoire.dflash.snapshot_swap import SnapshotSwap
from grimoire.history import HistoryStore, identity_hash


OPENCODE_SESSION_FIXTURES_DIR = Path(
    os.environ.get("GRIMOIRE_OPENCODE_SESSION_FIXTURES", "/home/yeowool/opencode_splits")
)
REAL_QWEN_TOKENIZER_PATH = Path(
    os.environ.get(
        "GRIMOIRE_REAL_QWEN_TOKENIZER",
        "/home/yeowool/models/tokenizers/qwen3.6-27B",
    )
)

_REAL_QWEN_TOKENIZER = None


def _loads_json_blob(blob):
    if not isinstance(blob, str):
        return {}
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


def _get_real_qwen_tokenizer():
    global _REAL_QWEN_TOKENIZER
    if _REAL_QWEN_TOKENIZER is None:
        if not REAL_QWEN_TOKENIZER_PATH.exists():
            raise unittest.SkipTest(f"Missing real Qwen tokenizer: {REAL_QWEN_TOKENIZER_PATH}")
        from transformers import AutoTokenizer

        _REAL_QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            str(REAL_QWEN_TOKENIZER_PATH),
            trust_remote_code=True,
            local_files_only=True,
        )
        # The replay harness only exercises chat-template/tokenization behavior,
        # not model forward passes, so do not warn on long real transcripts.
        _REAL_QWEN_TOKENIZER.model_max_length = max(
            getattr(_REAL_QWEN_TOKENIZER, "model_max_length", 0),
            1_048_576,
        )
    return _REAL_QWEN_TOKENIZER


def _json_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _tool_call_arguments(raw_input):
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None:
        return {}
    return {"input": raw_input}


def _iter_opencode_session_turns(session_doc):
    for raw_msg in session_doc.get("messages", []):
        meta = _loads_json_blob(raw_msg.get("data"))
        role = meta.get("role")
        parts = []
        for raw_part in raw_msg.get("parts", []):
            part = _loads_json_blob(raw_part.get("data"))
            if part:
                parts.append(part)

        if role == "user":
            text = "".join(
                part.get("text", "")
                for part in parts
                if part.get("type") == "text" and isinstance(part.get("text"), str)
            )
            if text:
                yield {
                    "kind": "user",
                    "messages": [{"role": "user", "content": text}],
                }
            continue

        if role != "assistant":
            continue

        for part in parts:
            part_type = part.get("type")
            if part_type == "reasoning" and isinstance(part.get("text"), str) and part.get("text"):
                yield {
                    "kind": "reasoning",
                    "messages": [{
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": part["text"],
                    }],
                }
                continue

            if part_type == "text" and isinstance(part.get("text"), str) and part.get("text"):
                yield {
                    "kind": "assistant_text",
                    "messages": [{"role": "assistant", "content": part["text"]}],
                }
                continue

            if part_type != "tool":
                continue

            tool_name = part.get("tool")
            call_id = part.get("callID")
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            if not isinstance(tool_name, str) or not isinstance(call_id, str):
                continue

            output = _json_text(state.get("output", ""))
            yield {
                "kind": "tool",
                "tool_name": tool_name,
                "call_id": call_id,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": _tool_call_arguments(state.get("input")),
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool": {"name": tool_name},
                        "tool_call_id": call_id,
                        "content": output,
                    },
                ],
            }


def _normalize_opencode_session_messages(session_doc):
    messages = []
    for turn in _iter_opencode_session_turns(session_doc):
        messages.extend(turn["messages"])
    return messages


def _load_opencode_session_messages(filename):
    path = OPENCODE_SESSION_FIXTURES_DIR / filename
    if not path.exists():
        raise unittest.SkipTest(f"Missing OpenCode session fixture: {path}")
    return _normalize_opencode_session_messages(json.loads(path.read_text()))


def _load_opencode_session_turns(filename):
    path = OPENCODE_SESSION_FIXTURES_DIR / filename
    if not path.exists():
        raise unittest.SkipTest(f"Missing OpenCode session fixture: {path}")
    return list(_iter_opencode_session_turns(json.loads(path.read_text())))


def _list_opencode_session_files():
    if not OPENCODE_SESSION_FIXTURES_DIR.exists():
        raise unittest.SkipTest(f"Missing OpenCode session fixture dir: {OPENCODE_SESSION_FIXTURES_DIR}")
    return sorted(OPENCODE_SESSION_FIXTURES_DIR.glob("opencode_ses_*.json"))


class _ReplayRequest:
    def __init__(self, payload, headers=None, cookies=None):
        self._payload = payload
        self.headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        self.cookies = cookies or {}

    async def json(self):
        return self._payload


def _response_json(response):
    body = getattr(response, "body", None)
    if not body:
        return {}
    if isinstance(body, bytes):
        return json.loads(body)
    if isinstance(body, bytearray):
        return json.loads(bytes(body))
    return json.loads(body)


async def _replay_session_file_async(filename):
    token = "test-key"
    user_hash = identity_hash(token)
    tmp_dir = "/dev/shm" if Path("/dev/shm").exists() else None
    tokenizer = _get_real_qwen_tokenizer()
    cfg = {
        "backend": "dflash",
        "ctx-size": 1_048_576,
        "predict": 64,
        "family": "qwen",
    }
    daemon = FakeDflashDaemon([3793, 248046])
    active = FakeActive("dflash-replay", cfg, daemon, tokenizer)
    active.session_kv = SessionKV(cap=2, prefix_cap=2)
    auth = {"authorization": f"Bearer {token}"}

    async def fake_start_model(model_name):
        return active

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        old_store = entrypoint.history_store
        old_api = entrypoint.API_KEY
        old_admin = entrypoint.ADMIN_TOKEN
        entrypoint.history_store = HistoryStore(str(Path(tmp) / "replay-history.sqlite3"))
        entrypoint.API_KEY = token
        entrypoint.ADMIN_TOKEN = token
        entrypoint.manager.active.clear()
        try:
            with patch.object(entrypoint.manager, "start_model", fake_start_model), patch.object(
                entrypoint.registry, "resolve", lambda _model_id: "dflash-replay"
            ), patch.object(entrypoint.usage_store, "record", lambda *args, **kwargs: None), patch.object(
                entrypoint.telemetry_store, "record", lambda *args, **kwargs: None
            ):
                turns = _load_opencode_session_turns(filename)
                if not turns:
                    return {"filename": filename, "turns": 0}

                conversation_id = entrypoint.history_store.create_conversation(
                    user_hash,
                    title=Path(filename).stem,
                    model="dflash-replay",
                    messages=[],
                )["id"]
                transcript = []
                expected_history_count = 0

                for turn in turns:
                    transcript.extend(turn["messages"])
                    daemon._tokens = [3793, 248046]
                    response = await entrypoint.chat_completions(
                        _ReplayRequest(
                            {
                                "model": "dflash-replay",
                                "conversation_id": conversation_id,
                                "messages": transcript,
                                "stream": False,
                                "max_tokens": 8,
                            },
                            headers=auth,
                        )
                    )
                    if response.status_code != 200:
                        raise AssertionError(
                            f"{filename}: {turn['kind']} -> {response.status_code} {_response_json(response)}"
                        )
                    body = _response_json(response)
                    if body["choices"][0]["message"]["content"] != "OK":
                        raise AssertionError(f"{filename}: unexpected assistant text {body}")
                    if body["usage"]["completion_tokens"] != 1:
                        raise AssertionError(f"{filename}: unexpected usage {body['usage']}")

                    if turn["messages"][-1]["role"] != "assistant":
                        expected_history_count += 1
                    expected_history_count += 1

                    count, tail = _history_tail_for_test(entrypoint.history_store, conversation_id, limit=2)
                    if count != expected_history_count:
                        raise AssertionError(
                            f"{filename}: expected {expected_history_count} history rows, got {count}"
                        )
                    if not tail or tail[-1]["role"] != "assistant" or tail[-1]["content"] != "OK":
                        raise AssertionError(f"{filename}: bad assistant tail {tail}")

                    if turn["kind"] == "tool":
                        if len(tail) < 2 or tail[-2]["role"] != "tool":
                            raise AssertionError(f"{filename}: missing tool tail {tail}")
                        if tail[-2]["content"] != turn["messages"][-1]["content"]:
                            raise AssertionError(f"{filename}: wrong tool content tail")

                    if turn["kind"] in {"user", "assistant_text", "tool"}:
                        prompt_ids = daemon.last_cmd_args["prompt_ids"]
                        if not active.session_kv.has_session(conversation_id):
                            raise AssertionError(f"{filename}: missing session kv entry")
                        slot, prefix_len = active.session_kv.get_session(conversation_id, prompt_ids)
                        if slot is None or prefix_len != len(prompt_ids):
                            raise AssertionError(f"{filename}: bad session kv state")
                        if daemon.last_cmd_args["snap_pos"] != len(prompt_ids):
                            raise AssertionError(f"{filename}: bad snap_pos")

                final_count, _ = _history_tail_for_test(entrypoint.history_store, conversation_id, limit=1)
                if final_count <= 0:
                    raise AssertionError(f"{filename}: empty history after replay")
                return {"filename": filename, "turns": len(turns), "history_count": final_count}
        finally:
            entrypoint.history_store = old_store
            entrypoint.API_KEY = old_api
            entrypoint.ADMIN_TOKEN = old_admin
            entrypoint.manager.active.clear()


def _replay_session_file_worker(filename):
    return asyncio.run(_replay_session_file_async(filename))


def _history_tail_for_test(store, conversation_id, limit=2):
    with store._lock, store._connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["n"]
        rows = conn.execute(
            """
            SELECT role, content_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY COALESCE(timestamp_ms, 0) DESC, created_at DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    tail = []
    for row in reversed(rows):
        try:
            content = json.loads(row["content_json"])
        except json.JSONDecodeError:
            content = row["content_json"]
        tail.append({"role": row["role"], "content": content})
    return count, tail


def _expected_protected_tool_ranges(messages, boundaries=None, protected_tools=None, tokenizer=None, prompt_ids=None):
    protected_names = set(protected_tools or {"obsidian_read-note"})
    tool_call_names = {}
    expected = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = fn.get("name")
                if isinstance(tc_id, str) and isinstance(fn_name, str):
                    tool_call_names[tc_id] = fn_name
            continue
        if msg.get("role") != "tool":
            continue
        tool_name = None
        if isinstance(msg.get("tool"), dict):
            tool_name = msg["tool"].get("name")
        elif isinstance(msg.get("tool"), str):
            tool_name = msg.get("tool")
        if tool_name is None:
            tc_id = msg.get("tool_call_id")
            if isinstance(tc_id, str):
                tool_name = tool_call_names.get(tc_id)
        if tool_name not in protected_names:
            continue
        if tokenizer is not None and prompt_ids is not None:
            rendered_before = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=False
            )
            rendered_after = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            start = len(tokenizer.encode(rendered_before, add_special_tokens=False))
            end = len(tokenizer.encode(rendered_after, add_special_tokens=False))
            if not (0 <= start < end <= len(prompt_ids)):
                continue
        else:
            start = boundaries[i - 1] if i > 0 else 0
            end = boundaries[i]
        expected.append((start, end))
    return expected


def _synthetic_prompt_blocks(boundaries, prompt_len, protected_indexes=None, include_generation=True):
    protected_indexes = set(protected_indexes or ())
    blocks = []
    cursor = 0
    for i, end in enumerate(boundaries or []):
        blocks.append(
            entrypoint.PromptBlock(
                block_id=f"message:{i}",
                index=len(blocks),
                start=cursor,
                end=end,
                role="message",
                kind="message",
                message_start=i,
                message_end=i + 1,
                protected=i in protected_indexes,
                metadata={"message_index": i},
            )
        )
        cursor = end

    if include_generation and cursor < prompt_len:
        blocks.append(
            entrypoint.PromptBlock(
                block_id="generation:0",
                index=len(blocks),
                start=cursor,
                end=prompt_len,
                role="assistant",
                kind="generation_prompt",
                message_start=len(boundaries or []),
                message_end=len(boundaries or []),
                protected=True,
                metadata={"generation_prompt": True},
            )
        )
    return blocks


class DflashRegistryValidationTests(unittest.TestCase):
    """Validation guardrails for the dflash backend entries in models.json."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models_dir = Path(self.tmp.name)
        self._orig_models_dir = registry_mod.MODELS_DIR
        registry_mod.MODELS_DIR = str(self.models_dir)

        # Real files on disk for the happy path; tests can remove or shadow as needed.
        (self.models_dir / "target.gguf").write_bytes(b"x")
        (self.models_dir / "draft.safetensors").write_bytes(b"x")
        (self.models_dir / "drafter.gguf").write_bytes(b"x")

        self.registry_path = self.models_dir / "registry.json"
        self.reg = registry_mod.ModelRegistry(path=str(self.registry_path), seed_path=None)

    def tearDown(self):
        registry_mod.MODELS_DIR = self._orig_models_dir

    def _add(self, name, **overrides):
        cfg = {
            "backend": "dflash",
            "target": "target.gguf",
            "draft": "draft.safetensors",
            "drafter": "drafter.gguf",
            "tokenizer": "Qwen/Qwen3.6-27B",
            "ctx-size": 1024,
        }
        cfg.update(overrides)
        self.reg.add(name, cfg)

    def test_valid_dflash_entry_passes(self):
        self._add("ok")
        valid, reason = self.reg.validate("ok")
        self.assertTrue(valid, reason)

    def test_missing_target_fails(self):
        os.unlink(self.models_dir / "target.gguf")
        self._add("notarget")
        valid, reason = self.reg.validate("notarget")
        self.assertFalse(valid)
        self.assertIn("Target model not found", reason)

    def test_missing_draft_fails(self):
        os.unlink(self.models_dir / "draft.safetensors")
        self._add("nodraft")
        valid, reason = self.reg.validate("nodraft")
        self.assertFalse(valid)
        self.assertIn("Draft model not found", reason)

    def test_missing_drafter_optional_unless_set(self):
        os.unlink(self.models_dir / "drafter.gguf")
        # Without a drafter field, validation should still pass.
        self._add("no_drafter_field")
        self.reg.update("no_drafter_field", {"drafter": None})
        valid, reason = self.reg.validate("no_drafter_field")
        self.assertTrue(valid, reason)

    def test_missing_tokenizer_fails_loudly(self):
        self._add("no_tok")
        self.reg.update("no_tok", {"tokenizer": None})
        valid, reason = self.reg.validate("no_tok")
        self.assertFalse(valid)
        self.assertIn("tokenizer", reason.lower())

    def test_unknown_backend_fails(self):
        self.reg.add("weird", {"backend": "tflite", "file": "x.tflite"})
        valid, reason = self.reg.validate("weird")
        self.assertFalse(valid)
        self.assertIn("Unknown backend", reason)


class PrefixCacheBoundaryTests(unittest.TestCase):
    """The lookup() probe must check supplied boundaries, not just the full prompt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = PrefixCache(cap=4, cache_dir=self.tmp.name)

    def _commit(self, prompt, boundary):
        prep = self.cache.prepare_inline_snap(prompt, boundary)
        self.assertIsNotNone(prep)
        slot, pos = prep
        self.cache.confirm_inline_snap(slot, pos, prompt)
        return slot

    def test_full_prompt_hit(self):
        prompt = list(range(20))
        self._commit(prompt, len(prompt))
        hit = self.cache.lookup(prompt)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], len(prompt))

    def test_partial_boundary_hit(self):
        prompt = list(range(30))
        sys_boundary = 10
        # Cache only the system prefix.
        self._commit(prompt[:sys_boundary], sys_boundary)
        # A new conversation that *starts* with the same system prefix.
        new_prompt = prompt[:sys_boundary] + list(range(100, 110))
        hit = self.cache.lookup(new_prompt, boundaries=[sys_boundary])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], sys_boundary)

    def test_lookup_picks_deepest_boundary(self):
        prompt = list(range(40))
        self._commit(prompt[:10], 10)
        self._commit(prompt[:25], 25)
        hit = self.cache.lookup(prompt, boundaries=[10, 25])
        self.assertEqual(hit[1], 25)

    def test_abort_does_not_register_entry(self):
        prompt = list(range(20))
        prep = self.cache.prepare_inline_snap(prompt, 20)
        self.assertIsNotNone(prep)
        slot, _ = prep
        self.cache.abort_inline_snap(slot)
        self.assertIsNone(self.cache.lookup(prompt))

    def test_disabled_cache_is_inert(self):
        cache = PrefixCache(cap=0, cache_dir=self.tmp.name)
        self.assertIsNone(cache.lookup([1, 2, 3]))
        self.assertIsNone(cache.prepare_inline_snap([1, 2, 3], 2))

    def test_invalid_boundary_is_skipped(self):
        prompt = list(range(10))
        self.assertIsNone(self.cache.prepare_inline_snap(prompt, 0))
        self.assertIsNone(self.cache.prepare_inline_snap(prompt, 11))


class LooksLikeLocalPathTests(unittest.TestCase):
    """`_looks_like_local_path` must distinguish HF repo ids from filesystem paths."""

    def _check(self, spec, expected):
        self.assertEqual(registry_mod._looks_like_local_path(spec), expected, spec)

    def test_absolute_path(self):
        self._check("/models/qwen", True)

    def test_explicit_relative_path(self):
        self._check("./models/qwen", True)
        self._check("../tokenizers/qwen", True)

    def test_nested_relative_path(self):
        self._check("models/qwen/tokenizer", True)

    def test_hf_repo_id(self):
        self._check("Qwen/Qwen3.6-27B", False)
        self._check("google/gemma-4-31b-it", False)

    def test_bare_name(self):
        self._check("gpt2", False)

    def test_empty_or_invalid(self):
        self._check("", False)
        self._check(None, False)


class FakeTokenizer:
    """Char-by-token tokenizer that satisfies the slice of the HF API _proxy_dflash uses.

    Each token id is the unicode codepoint of one character, so encode/decode
    round-trip cleanly and incremental decode produces clean per-char deltas.
    """

    eos_token_id = 0
    unk_token_id = 999

    _SPECIAL = {"<|im_end|>": 1}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m.get('role')}]{m.get('content', '')}[/{m.get('role')}]")
        text = "".join(parts)
        if add_generation_prompt:
            text += "[assistant]"
        return text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        skipped = {self.eos_token_id, self._SPECIAL["<|im_end|>"]} if skip_special_tokens else set()
        return "".join(chr(t) for t in tokens if t not in skipped)

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL.get(token, -1)


class FakeDflashDaemon:
    """Stand-in for the dflash daemon binary.

    Yields a pre-baked list of tokens. send_generate_cmd creates a real temp
    file so the production code's `os.unlink` cleanup works without patching.
    """

    def __init__(self, tokens):
        self._tokens = list(tokens)
        self.last_cmd_args = None
        self._running = True
        self.freed_slots = []

    def is_running(self):
        return self._running

    def send_generate_cmd(
        self, prompt_ids, n_gen,
        prefix_cache_slot=None, snap_slot=None, snap_pos=None,
        temperature=0.8, top_p=0.9, top_k=40, seed=None,
    ):
        self.last_cmd_args = {
            "prompt_ids": list(prompt_ids),
            "n_gen": n_gen,
            "prefix_cache_slot": prefix_cache_slot,
            "snap_slot": snap_slot,
            "snap_pos": snap_pos,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        return path

    def read_next_token(self):
        if not self._tokens:
            return None
        return self._tokens.pop(0)

    def free_snapshot(self, slot):
        self.freed_slots.append(slot)


class FakeActive:
    """Lightweight ActiveModel substitute for the dflash proxy path."""

    def __init__(self, name, cfg, daemon, tokenizer):
        self.name = name
        self.cfg = cfg
        self.gpu = 0
        self.port = None
        self.backend_type = entrypoint.BACKEND_DFLASH
        self.dflash_daemon = daemon
        self.prefix_cache = None
        self.prefill_config = None
        self.session_kv = None
        self.snapshot_swap = None
        self._tokenizer = tokenizer
        self._lock = None

    def get_tokenizer(self):
        return self._tokenizer

    def dflash_lock(self):
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def is_running(self):
        return self.dflash_daemon.is_running()


def _parse_sse(body_text):
    """Parse SSE event stream into a list of parsed JSON frames + raw terminators."""
    frames = []
    for chunk in body_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk or not chunk.startswith("data:"):
            continue
        data = chunk[len("data:"):].strip()
        if data == "[DONE]":
            frames.append("[DONE]")
        else:
            try:
                frames.append(json.loads(data))
            except json.JSONDecodeError:
                frames.append(data)
    return frames


class DflashDaemonProtocolTests(unittest.TestCase):
    """Exercise send_generate_cmd / read_next_token without launching the binary."""

    def setUp(self):
        self.daemon = DflashDaemon(target_path="t", draft_path="d")
        # Fake proc with a captured stdin and a benign poll().
        self.captured = io.BytesIO()
        proc = type("Proc", (), {})()
        proc.stdin = self.captured
        proc.poll = lambda: None
        self.daemon.proc = proc
        # Read end of pipe for the streamed tokens.
        self.r_fd, self.w_fd = os.pipe()
        self.daemon.r_pipe = self.r_fd

    def tearDown(self):
        for fd in (self.r_fd, self.w_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def test_send_generate_cmd_writes_command_and_prompt_bin(self):
        path = self.daemon.send_generate_cmd(
            [10, 20, 30], 16,
            temperature=0.7, top_p=0.95, top_k=50, seed=42,
        )
        try:
            sent = self.captured.getvalue().decode()
            self.assertTrue(sent.startswith(path + " 16"), sent)
            self.assertIn("temp=0.7000", sent)
            self.assertIn("top_p=0.9500", sent)
            self.assertIn("top_k=50", sent)
            self.assertIn("seed=42", sent)
            self.assertTrue(sent.endswith("\n"))
            with open(path, "rb") as f:
                prompt_bytes = f.read()
            self.assertEqual(prompt_bytes, struct.pack("<iii", 10, 20, 30))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_send_generate_cmd_includes_restore_and_snap(self):
        path = self.daemon.send_generate_cmd(
            [1, 2], 8, prefix_cache_slot=3, snap_slot=5, snap_pos=2,
        )
        try:
            sent = self.captured.getvalue().decode()
            self.assertIn(f"RESTORE 3 {path} 8", sent)
            self.assertIn("snap=2:5", sent)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_read_next_token_parses_int32_stream(self):
        os.write(self.w_fd, struct.pack("<i", 42))
        os.write(self.w_fd, struct.pack("<i", 99))
        self.assertEqual(self.daemon.read_next_token(), 42)
        self.assertEqual(self.daemon.read_next_token(), 99)

    def test_read_next_token_returns_none_on_sentinel(self):
        os.write(self.w_fd, struct.pack("<i", -1))
        self.assertIsNone(self.daemon.read_next_token())

    def test_read_next_token_returns_none_on_closed_pipe(self):
        os.close(self.w_fd)
        self.w_fd = -1  # already closed; suppress tearDown re-close
        self.assertIsNone(self.daemon.read_next_token())


class DflashHelperTests(unittest.TestCase):
    """Pure-logic helpers used by _proxy_dflash."""

    def setUp(self):
        self.tok = FakeTokenizer()

    def test_collect_stop_ids_includes_eos_and_chat_end(self):
        ids, stop_seqs = entrypoint._dflash_collect_stop_ids(self.tok, None, {})
        self.assertIn(self.tok.eos_token_id, ids)
        self.assertIn(self.tok._SPECIAL["<|im_end|>"], ids)
        self.assertEqual(stop_seqs, [])

    def test_collect_stop_ids_includes_request_stop_string(self):
        ids, stop_seqs = entrypoint._dflash_collect_stop_ids(self.tok, "STOP", {})
        self.assertNotIn(ord("S"), ids)
        self.assertIn(tuple(ord(c) for c in "STOP"), stop_seqs)

    def test_collect_stop_ids_includes_request_stop_list(self):
        ids, stop_seqs = entrypoint._dflash_collect_stop_ids(self.tok, ["A", "B"], {})
        self.assertIn(ord("A"), ids)
        self.assertIn(ord("B"), ids)
        self.assertEqual(stop_seqs, [])

    def test_collect_stop_ids_includes_cfg_stop_strings(self):
        ids, stop_seqs = entrypoint._dflash_collect_stop_ids(self.tok, None, {"stop-strings": ["Z"]})
        self.assertIn(ord("Z"), ids)
        self.assertEqual(stop_seqs, [])

    def test_collect_stop_ids_skips_unknown_specials(self):
        # convert_tokens_to_ids returns -1 for unknown specials; those must NOT enter the set.
        ids, _ = entrypoint._dflash_collect_stop_ids(self.tok, None, {})
        self.assertNotIn(-1, ids)

    def test_collect_stop_ids_keeps_multi_token_cfg_stop_as_sequence(self):
        ids, stop_seqs = entrypoint._dflash_collect_stop_ids(self.tok, None, {"stop-strings": ["END"]})
        self.assertNotIn(ord("E"), ids)
        self.assertIn(tuple(ord(c) for c in "END"), stop_seqs)

    def test_prompt_layout_returns_one_block_per_message_plus_generation(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        prompt_ids, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].message_start, 0)
        self.assertEqual(blocks[1].message_start, 1)
        self.assertEqual(blocks[-1].kind, "generation_prompt")
        sys_prefix = self.tok.encode(self.tok.apply_chat_template(msgs[:1]))
        self.assertEqual(blocks[0].end, len(sys_prefix))
        self.assertEqual(prompt_ids[:blocks[0].end], sys_prefix)
        full_no_gen = self.tok.encode(self.tok.apply_chat_template(msgs))
        self.assertEqual(blocks[1].end, len(full_no_gen))
        self.assertEqual(prompt_ids[:blocks[1].end], full_no_gen)

    def test_prompt_layout_user_only_returns_one_message_block(self):
        msgs = [{"role": "user", "content": "hi"}]
        prompt_ids, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].message_start, 0)
        self.assertEqual(blocks[0].message_end, 1)
        self.assertEqual(blocks[1].kind, "generation_prompt")
        user_prefix = self.tok.encode(self.tok.apply_chat_template(msgs))
        self.assertEqual(blocks[0].end, len(user_prefix))
        self.assertEqual(prompt_ids[:blocks[0].end], user_prefix)

    def test_prompt_layout_walks_multi_turn_conversation(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
        ]
        prompt_ids, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        self.assertEqual(len(blocks), len(msgs) + 1)
        ends = [block.end for block in blocks[:-1]]
        self.assertEqual(ends, sorted(ends))
        self.assertEqual(len(set(ends)), len(ends))
        for i, n in enumerate(ends):
            expected = self.tok.encode(self.tok.apply_chat_template(msgs[: i + 1]))
            self.assertEqual(n, len(expected))
            self.assertEqual(prompt_ids[:n], expected)
        last_user_start = self.tok.encode(self.tok.apply_chat_template(msgs[:-1]))
        self.assertEqual(blocks[-2].start, len(last_user_start))

    def test_prompt_layout_marks_obsidian_read_note_block_protected(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "read this note"},
            {"role": "assistant", "content": "I'll use the obsidian_read-note tool"},
            {
                "role": "tool",
                "tool": {"name": "obsidian_read-note"},
                "content": "this is the note content that should be protected",
            },
            {"role": "assistant", "content": "Here's what the note says..."},
            {"role": "user", "content": "thanks"},
        ]
        _, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        tool_block = next(block for block in blocks if block.kind == "tool")
        self.assertTrue(tool_block.protected)
        self.assertEqual(tool_block.message_start, 3)
        self.assertEqual(tool_block.message_end, 4)

    def test_prompt_layout_groups_adjacent_tool_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read notes"},
            {
                "role": "tool",
                "tool": {"name": "obsidian_read-note"},
                "content": "note A content here",
            },
            {"role": "assistant", "content": "reading next..."},
            {
                "role": "tool",
                "tool": {"name": "obsidian_read-note"},
                "content": "note B content here",
            },
            {"role": "assistant", "content": "done"},
        ]
        _, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        tool_blocks = [block for block in blocks if block.kind == "tool"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertTrue(all(block.protected for block in tool_blocks))
        self.assertLess(tool_blocks[0].end, tool_blocks[1].start)

    def test_prompt_layout_ignores_non_obsidian_tools(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do something"},
            {
                "role": "tool",
                "tool": {"name": "bash"},
                "content": "ls -la",
            },
            {"role": "assistant", "content": "ok"},
        ]
        _, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        tool_block = next(block for block in blocks if block.kind == "tool")
        self.assertFalse(tool_block.protected)

    def test_prompt_layout_protects_obsidian_tool_output_via_tool_call_id(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read note"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "obsidian_read-note", "arguments": {"filename": "x.md"}},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "protected note body",
            },
        ]
        _, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, msgs, add_generation_prompt=True, model_cfg={}, active=None
        )
        tool_block = next(block for block in blocks if block.kind == "tool")
        self.assertTrue(tool_block.protected)
        self.assertEqual(tool_block.message_start, 3)
        self.assertEqual(tool_block.message_end, 4)


class OpenCodeSessionFixtureTests(unittest.TestCase):
    """Real-world OpenCode session dumps should map cleanly onto DFlash helpers."""

    def setUp(self):
        self.tok = FakeTokenizer()

    def test_real_session_obsidian_tool_outputs_become_protected_blocks(self):
        messages = _load_opencode_session_messages(
            "opencode_ses_1eb7_Update_machine_config_with_Obsidian_resume.json"
        )
        prompt_ids, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, messages, add_generation_prompt=True, model_cfg={}, active=None
        )
        self.assertGreaterEqual(len(messages), 6)
        protected = [(block.start, block.end) for block in blocks if block.protected and block.kind == "tool"]
        boundaries = [block.end for block in blocks if block.kind != "generation_prompt"]
        self.assertTrue(protected)
        self.assertEqual(protected, _expected_protected_tool_ranges(messages, boundaries, tokenizer=self.tok, prompt_ids=prompt_ids))

    def test_real_session_without_obsidian_reads_has_no_protected_blocks(self):
        messages = _load_opencode_session_messages(
            "opencode_ses_1e9c_Export_10_latest_opencode_conversations_to_JSON.json"
        )
        _, blocks = entrypoint._prompt_layout_from_messages(
            self.tok, messages, add_generation_prompt=True, model_cfg={}, active=None
        )
        protected = [block for block in blocks if block.protected and block.kind == "tool"]
        self.assertEqual(protected, [])


class MaybeCompressHeadTailTests(unittest.TestCase):
    """maybe_compress must protect head + tail and only compress middle blocks."""

    def setUp(self):
        from grimoire.dflash.prefill import PrefillConfig
        self.PrefillConfig = PrefillConfig

    def _run(self, prompt_ids, blocks, keep_ratio=0.5, threshold=10, tail_budget=600):
        """Drive maybe_compress synchronously with a stub daemon."""
        import asyncio
        from grimoire.dflash.prefill import maybe_compress

        class StubDaemon:
            def __init__(self, ratio):
                self.ratio = ratio
                self.compress_calls = []

            def compress(self, ids, drafter_path, keep_ratio):
                self.compress_calls.append(list(ids))
                keep = max(1, int(len(ids) * self.ratio))
                # Return a fixed sentinel so the test can detect the compressed slice.
                return [-7] * keep

        daemon = StubDaemon(keep_ratio)
        cfg = self.PrefillConfig(
            enabled=True,
            threshold=threshold,
            keep_ratio=keep_ratio,
            drafter_path="/dummy",
            tail_budget=tail_budget,
        )
        compressed, fired, effective_blocks = asyncio.run(
            maybe_compress(prompt_ids, daemon, cfg, blocks=blocks)
        )
        return compressed, fired, daemon, effective_blocks

    def test_head_and_tail_are_preserved_byte_identical(self):
        # 3000-token prompt, 6 boundaries:
        #   msg 0 ends at 50    (system)
        #   msg 1 ends at 200   (first user — together with msg 0 forms the head)
        #   msg 2 ends at 1500  (1300-tok turn — the middle to compress)
        #   msg 3 ends at 2000  (500-tok turn)
        #   msg 4 ends at 2500  (500-tok turn)
        #   msg 5 ends at 2800  (300-tok turn — last user content)
        # Head (compress_start) = boundaries[1] = 200, which protects sys + the
        # first user message (opencode compaction summary).
        # tail_budget=700: walking backwards, msg 5 (300) fits (tail=300), but
        # adding msg 4 (500) → 800 > 700, so break at i=4 with compress_end =
        # boundaries[3] = 2000. Protected tail spans msgs 4 and 5 (overshooting
        # budget by msg 4's 500 tokens — intentional whole-turn protection).
        prompt_ids = list(range(3000))
        boundaries = [50, 200, 1500, 2000, 2500, 2800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        expected_tail_start = 2000
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=1000, tail_budget=700
        )
        self.assertTrue(fired)
        self.assertEqual(compressed[:200], prompt_ids[:200])
        tail_len = len(prompt_ids) - expected_tail_start
        self.assertEqual(compressed[-tail_len:], prompt_ids[expected_tail_start:])
        self.assertEqual(daemon.compress_calls, [prompt_ids[200:1500], prompt_ids[1500:2000]])
        self.assertEqual([block.index for block in effective_blocks if block.compressed], [2, 3])

    def test_all_turns_fit_in_tail_budget_no_compression(self):
        # 4 boundaries totalling 1500 tokens of conversation, tail_budget=2000:
        # the loop's else clause fires, compress_end == compress_start, no compress.
        prompt_ids = list(range(2500))
        boundaries = [500, 1000, 1500, 2000]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=100, tail_budget=2000
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])

    def test_single_huge_last_turn_overshoots_budget(self):
        # Last turn alone is 3000 tokens, tail_budget=500. The first iteration
        # of the backwards walk sees turn_len > budget and breaks immediately,
        # setting compress_end = boundaries[-2]. The huge turn is still protected
        # (intentional overshoot — never truncate mid-turn).
        # boundaries[0]=50 is sys, boundaries[1]=100 ends the first user message
        # and forms compress_start (head protection).
        prompt_ids = list(range(4500))
        boundaries = [50, 100, 600, 1100, 1400, 4400]  # last turn = 4400-1400 = 3000
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=100, tail_budget=500
        )
        self.assertTrue(fired)
        tail_len = len(prompt_ids) - 1400
        self.assertEqual(compressed[-tail_len:], prompt_ids[1400:])
        self.assertEqual(daemon.compress_calls, [prompt_ids[100:600], prompt_ids[600:1100], prompt_ids[1100:1400]])
        self.assertEqual([block.index for block in effective_blocks if block.compressed], [2, 3, 4])

    def test_tail_walk_accumulates_until_break(self):
        # 7 boundaries; sys (boundaries[0]) + first_user (boundaries[1]) form
        # the protected head. tail_budget=900 accommodates the last three
        # post-head turns (200 + 300 + 200 = 700), but the fourth (500) would
        # push to 1200 > 900 → break at i=3, compress_end = boundaries[2] = 1300.
        # Middle = [100:1300] = 1200 tokens, large enough to clear the
        # 1024-min-middle short-circuit.
        prompt_ids = list(range(3000))
        boundaries = [50, 100, 1300, 1800, 2000, 2300, 2500]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=100, tail_budget=900
        )
        self.assertTrue(fired)
        tail_len = len(prompt_ids) - 1300
        self.assertEqual(compressed[-tail_len:], prompt_ids[1300:])
        self.assertEqual(daemon.compress_calls, [prompt_ids[100:1300]])
        self.assertEqual([block.index for block in effective_blocks if block.compressed], [2])

    def test_skip_when_middle_too_small(self):
        # Middle < 1024 → no compression even if threshold is exceeded.
        prompt_ids = list(range(2000))
        boundaries = [100, 500, 800, 1500]  # middle = 800 - 100 = 700, < 1024
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, _, _ = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=500
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)

    def test_single_block_still_skips_due_to_head_protection(self):
        prompt_ids = list(range(3000))
        boundaries = [200]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks, keep_ratio=0.1, threshold=1000
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])

    def test_no_blocks_compresses_whole_prompt(self):
        prompt_ids = list(range(3000))
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks=None, keep_ratio=0.1, threshold=1000
        )
        self.assertTrue(fired)
        self.assertEqual(daemon.compress_calls, [prompt_ids])
        self.assertEqual(len(effective_blocks), 1)
        self.assertTrue(effective_blocks[0].compressed)

    def test_below_threshold_short_circuits(self):
        prompt_ids = list(range(100))
        blocks = _synthetic_prompt_blocks([10, 50, 80], len(prompt_ids), include_generation=True)
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks=blocks, keep_ratio=0.1, threshold=1000
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])


class MaybeCompressProtectedRangesTests(unittest.TestCase):
    """maybe_compress must respect protected blocks and preserve them byte-identical."""

    def setUp(self):
        from grimoire.dflash.prefill import PrefillConfig
        self.PrefillConfig = PrefillConfig

    def _run(self, prompt_ids, blocks, keep_ratio=0.5, threshold=10, tail_budget=600):
        import asyncio
        from grimoire.dflash.prefill import maybe_compress

        class StubDaemon:
            def __init__(self, ratio):
                self.ratio = ratio
                self.compress_calls = []

            def compress(self, ids, drafter_path, keep_ratio):
                self.compress_calls.append(list(ids))
                keep = max(1, int(len(ids) * self.ratio))
                return [-7] * keep

        daemon = StubDaemon(keep_ratio)
        cfg = self.PrefillConfig(
            enabled=True,
            threshold=threshold,
            keep_ratio=keep_ratio,
            drafter_path="/dummy",
            tail_budget=tail_budget,
        )
        compressed, fired, effective_blocks = asyncio.run(
            maybe_compress(prompt_ids, daemon, cfg, blocks=blocks)
        )
        return compressed, fired, daemon, effective_blocks

    def test_protected_range_in_middle_is_preserved(self):
        # Prompt: 5000 tokens
        # Protect one middle block while leaving compressible blocks on both sides.
        prompt_ids = list(range(5000))
        boundaries = [50, 200, 3000, 3500, 4200, 4500, 4800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), protected_indexes={3}, include_generation=True)
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks=blocks,
            keep_ratio=0.1, threshold=2481, tail_budget=300
        )
        self.assertTrue(fired)
        self.assertEqual(compressed[:200], prompt_ids[:200])
        tail_len = 5000 - 4200
        self.assertEqual(compressed[-tail_len:], prompt_ids[4200:])
        gap1_compressed_len = 280
        protected_offset = 200 + gap1_compressed_len
        self.assertEqual(
            compressed[protected_offset:protected_offset + 500],
            prompt_ids[3000:3500],
            "protected block must be byte-identical in compressed output",
        )
        self.assertEqual(len(daemon.compress_calls), 2)
        self.assertEqual(daemon.compress_calls[0], prompt_ids[200:3000])
        self.assertEqual(daemon.compress_calls[1], prompt_ids[3500:4200])
        self.assertEqual([block.index for block in effective_blocks if block.compressed], [2, 4])

    def test_protected_head_block_skips_that_block(self):
        prompt_ids = list(range(5000))
        boundaries = [50, 200, 2500, 3000, 3500, 4500, 4800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), protected_indexes={2}, include_generation=True)
        blocks[1] = entrypoint.PromptBlock(
            block_id=blocks[1].block_id,
            index=blocks[1].index,
            start=blocks[1].start,
            end=blocks[1].end,
            role=blocks[1].role,
            kind=blocks[1].kind,
            message_start=blocks[1].message_start,
            message_end=blocks[1].message_end,
            protected=True,
            metadata=blocks[1].metadata,
        )
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks=blocks,
            keep_ratio=0.1, threshold=3201, tail_budget=300
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])

    def test_protected_tail_block_is_left_uncompressed(self):
        prompt_ids = list(range(5000))
        boundaries = [50, 200, 2500, 3000, 3500, 4500, 4800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), protected_indexes={4}, include_generation=True)
        compressed, fired, daemon, effective_blocks = self._run(
            prompt_ids, blocks=blocks,
            keep_ratio=0.1, threshold=2481, tail_budget=300
        )
        self.assertTrue(fired)
        self.assertEqual(compressed[:200], prompt_ids[:200])
        self.assertEqual(compressed[-2000:], prompt_ids[3000:])
        self.assertEqual(len(daemon.compress_calls), 2)
        self.assertEqual(daemon.compress_calls[0], prompt_ids[200:2500])
        self.assertEqual(daemon.compress_calls[1], prompt_ids[2500:3000])
        self.assertFalse(effective_blocks[4].compressed)

    def test_protected_middle_block_consumes_all_compressible_skips(self):
        prompt_ids = list(range(2000))
        boundaries = [50, 200, 1800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), protected_indexes={1}, include_generation=True)
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks=blocks,
            keep_ratio=0.1, threshold=500, tail_budget=200
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])

    def test_small_block_below_min_is_skipped(self):
        prompt_ids = list(range(2000))
        boundaries = [50, 200, 1800]
        blocks = _synthetic_prompt_blocks(boundaries, len(prompt_ids), include_generation=True)
        compressed, fired, daemon, _ = self._run(
            prompt_ids, blocks=blocks,
            keep_ratio=0.1, threshold=500, tail_budget=200
        )
        self.assertFalse(fired)
        self.assertEqual(compressed, prompt_ids)
        self.assertEqual(daemon.compress_calls, [])


class SessionKVTests(unittest.TestCase):
    """SessionKV maps conversation_id -> (slot, prefix_len, prefix_hash) for cross-turn KV reuse."""

    # Long enough to cover every prefix_len used by these tests.
    PROMPT = list(range(256))

    def test_get_session_returns_none_for_unknown_conv(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        self.assertIsNone(sk.get_session("nope", []))

    def test_reserve_slot_uses_offset_for_cold_cache(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        # First reservation hands out the offset slot (= prefix_cap).
        self.assertEqual(sk.reserve_slot(), 2)

    def test_reserve_then_update_keeps_consecutive_slots_distinct(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        slot_a = sk.reserve_slot()
        sk.update("a", slot_a, 100, self.PROMPT)
        slot_b = sk.reserve_slot()
        sk.update("b", slot_b, 200, self.PROMPT)
        self.assertNotEqual(slot_a, slot_b)
        self.assertEqual({slot_a, slot_b}, {2, 3})

    def test_reserve_after_eviction_avoids_mru_collision(self):
        # Regression: reserve_slot used to compute idx=len(sessions) after
        # popping the LRU, which gave the SAME slot as the surviving MRU
        # session. Two conversations ended up mapped to one daemon slot and
        # the survivor's snapshot was clobbered on the next snapshot.
        sk = SessionKV(cap=2, prefix_cap=2)
        slot_a = sk.reserve_slot()
        sk.update("a", slot_a, 100, self.PROMPT)
        slot_b = sk.reserve_slot()
        sk.update("b", slot_b, 200, self.PROMPT)
        # 'a' is LRU; reserving for 'c' must evict 'a' and reuse its slot.
        slot_c = sk.reserve_slot()
        self.assertEqual(slot_c, slot_a, "evicted slot should be reused")
        self.assertNotEqual(slot_c, slot_b, "must not collide with the MRU survivor")

    def test_get_session_promotes_lru(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        sk.update("a", 2, 10, self.PROMPT)
        sk.update("b", 3, 20, self.PROMPT)
        # Touching 'a' promotes it; 'b' is now LRU and gets evicted on next reserve.
        sk.get_session("a", self.PROMPT)
        slot_c = sk.reserve_slot()
        sk.update("c", slot_c, 30, self.PROMPT)
        self.assertIsNone(sk.get_session("b", self.PROMPT))
        self.assertIsNotNone(sk.get_session("a", self.PROMPT))

    def test_evict_is_idempotent(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        sk.update("a", 2, 10, self.PROMPT)
        sk.evict("a")
        sk.evict("a")  # Must not raise.
        self.assertIsNone(sk.get_session("a", self.PROMPT))

    def test_clear_drops_all_state(self):
        sk = SessionKV(cap=2, prefix_cap=2)
        sk.update("a", 2, 10, self.PROMPT)
        sk.update("b", 3, 20, self.PROMPT)
        sk.clear()
        self.assertIsNone(sk.get_session("a", self.PROMPT))
        self.assertIsNone(sk.get_session("b", self.PROMPT))
        # After clear, reservations restart from offset.
        self.assertEqual(sk.reserve_slot(), 2)

    def test_cap_zero_disables_reservation(self):
        sk = SessionKV(cap=0, prefix_cap=2)
        self.assertIsNone(sk.reserve_slot())

    def test_reserve_slot_does_not_wrap_into_prefix_slots(self):
        sk = SessionKV(cap=10, prefix_cap=7)
        self.assertEqual(sk.reserve_slot(), 7)
        sk.update("a", 7, 10, self.PROMPT)
        self.assertEqual(sk.reserve_slot(), 7)
        self.assertNotIn(0, {entry[0] for entry in sk.sessions.values()})

    def test_get_session_accepts_extended_prompt(self):
        # Continuation: the next-turn prompt is the cached prefix plus a delta.
        sk = SessionKV(cap=2, prefix_cap=2)
        cached = list(range(100))
        sk.update("a", 2, 50, cached)
        extended = cached + list(range(100, 150))
        self.assertEqual(sk.get_session("a", extended), (2, 50))

    def test_get_session_evicts_on_hash_mismatch(self):
        # In-place edit: same conversation_id, but tokens within the cached
        # prefix changed. Must NOT restore stale KV.
        sk = SessionKV(cap=2, prefix_cap=2)
        original = list(range(100))
        sk.update("a", 2, 50, original)
        edited = original[:10] + list(range(1000, 1040)) + original[50:]
        self.assertIsNone(sk.get_session("a", edited))
        # The mismatch evicted the entry — even the original prompt now misses.
        self.assertIsNone(sk.get_session("a", original))

    def test_get_session_evicts_when_prompt_shorter_than_prefix(self):
        # Truncation: caller retried with a shorter prompt than the cached
        # prefix. The snapshot can't possibly match; evict.
        sk = SessionKV(cap=2, prefix_cap=2)
        sk.update("a", 2, 100, list(range(200)))
        self.assertIsNone(sk.get_session("a", list(range(50))))


class SnapshotSwapTests(unittest.TestCase):
    """SnapshotSwap keeps disk-resident snapshots distinct and slot-scoped."""

    class Daemon:
        def __init__(self):
            self.saved = []
            self.loaded = []
            self.freed = []

        def save_snapshot(self, slot, path):
            self.saved.append((slot, path))
            Path(path).write_text(f"slot={slot}\n")

        def load_snapshot(self, slot, path):
            self.loaded.append((slot, path))

        def free_snapshot(self, slot):
            self.freed.append(slot)

    def test_reserve_uses_allowed_slot_range_and_swaps_by_key_path(self):
        with tempfile.TemporaryDirectory() as td:
            daemon = self.Daemon()
            swap = SnapshotSwap(td, max_vram_slots=1, slot_offset=2, slot_count=2)
            key_a = b"a" * 16
            key_b = b"b" * 16

            self.assertEqual(swap.reserve_slot(daemon, key_a), 2)
            self.assertEqual(swap.reserve_slot(daemon, key_b), 2)
            self.assertEqual(swap.get(key_a), (None, False))
            self.assertEqual(swap.get(key_b), (2, True))

            a_path = swap.disk[key_a]
            self.assertIn(key_a.hex()[:16], Path(a_path).name)
            self.assertTrue(Path(a_path).exists())
            self.assertTrue(all(slot == 2 for slot, _ in daemon.saved))

            self.assertEqual(swap.reserve_slot(daemon, key_a), 2)
            self.assertEqual(swap.get(key_a), (2, True))
            self.assertEqual(swap.get(key_b), (None, False))
            self.assertEqual(daemon.loaded[-1], (2, a_path))
            self.assertNotEqual(a_path, swap.disk[key_b])

    def test_reserve_existing_vram_key_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            daemon = self.Daemon()
            swap = SnapshotSwap(td, max_vram_slots=1, slot_offset=2, slot_count=1)
            key = b"a" * 16
            self.assertEqual(swap.reserve_slot(daemon, key), 2)
            self.assertEqual(swap.reserve_slot(daemon, key), 2)
            self.assertEqual(daemon.saved, [])
            self.assertEqual(swap.get(key), (2, True))

    def test_discard_removes_vram_or_disk_without_saving(self):
        with tempfile.TemporaryDirectory() as td:
            daemon = self.Daemon()
            swap = SnapshotSwap(td, max_vram_slots=1, slot_offset=2, slot_count=1)
            key_a = b"a" * 16
            key_b = b"b" * 16

            swap.reserve_slot(daemon, key_a)
            swap.discard(daemon, key_a)
            self.assertIsNone(swap.get(key_a))
            self.assertEqual(daemon.freed, [2])

            swap.reserve_slot(daemon, key_a)
            swap.reserve_slot(daemon, key_b)
            path = Path(swap.disk[key_a])
            self.assertTrue(path.exists())
            swap.discard(daemon, key_a)
            self.assertIsNone(swap.get(key_a))
            self.assertFalse(path.exists())


class PrefixCachePersistenceTests(unittest.TestCase):
    def test_load_clamps_slots_to_current_cap(self):
        with tempfile.TemporaryDirectory() as td:
            cache = PrefixCache(cap=4, cache_dir=td)
            key = cache.hash_prefix([1, 2, 3])
            Path(td, "index.json").write_text(json.dumps({
                "entries": [{"key_hex": key.hex(), "slot": 3}],
                "next_slot": 3,
            }))

            smaller = PrefixCache(cap=2, cache_dir=td)
            smaller.load()
            self.assertEqual(list(smaller.entries.values()), [])
            self.assertEqual(smaller.next_slot, 1)

    def test_cleanup_clears_saved_index_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            cache = PrefixCache(cap=2, cache_dir=td)
            prompt = [1, 2, 3, 4]
            slot, boundary = cache.prepare_inline_snap(prompt, 2)
            cache.confirm_inline_snap(slot, boundary, prompt)
            cache.save()
            cache.cleanup(None)
            self.assertFalse(Path(td, "index.json").exists())


class DflashProxyIntegrationTests(unittest.TestCase):
    """End-to-end /v1/chat/completions through _proxy_dflash with a fake daemon."""

    def setUp(self):
        self._old_api = entrypoint.API_KEY
        self._old_admin = entrypoint.ADMIN_TOKEN
        entrypoint.API_KEY = "test-key"
        entrypoint.ADMIN_TOKEN = "test-key"
        entrypoint.manager.active.clear()
        self.client = TestClient(entrypoint.app)
        self.auth = {"authorization": "Bearer test-key"}

    def tearDown(self):
        entrypoint.API_KEY = self._old_api
        entrypoint.ADMIN_TOKEN = self._old_admin
        entrypoint.manager.active.clear()

    def _install_fake_active(self, tokens, cfg_overrides=None):
        cfg = {"backend": "dflash", "ctx-size": 1024, "predict": 64}
        if cfg_overrides:
            cfg.update(cfg_overrides)
        daemon = FakeDflashDaemon(tokens)
        active = FakeActive("dflash-test", cfg, daemon, FakeTokenizer())

        async def fake_start_model(model_name):
            return active

        # registry.resolve has to recognise the alias; add it to the live registry.
        if "dflash-test" not in entrypoint.registry.list_all():
            entrypoint.registry.add("dflash-test", cfg)
        self.addCleanup(self._remove_alias, "dflash-test")
        self._patch_start = patch.object(entrypoint.manager, "start_model", fake_start_model)
        self._patch_start.start()
        self.addCleanup(self._patch_start.stop)
        return active, daemon

    def _remove_alias(self, name):
        try:
            entrypoint.registry.remove(name)
        except KeyError:
            pass

    def test_streaming_emits_per_token_deltas_in_order(self):
        # Emit "hi!" then EOS — EOS lands in stop_ids and must not appear as a delta.
        active, _ = self._install_fake_active([ord("h"), ord("i"), ord("!"), 0])
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "dflash-test", "messages": [{"role": "user", "content": "ping"}], "stream": True},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        frames = _parse_sse(response.text)
        deltas = [f["choices"][0]["delta"].get("content", "")
                  for f in frames
                  if isinstance(f, dict) and "choices" in f and f["choices"][0].get("delta", {}).get("content")]
        self.assertEqual(deltas, ["h", "i", "!"], frames)
        self.assertEqual(frames[-1], "[DONE]")
        # Final usage frame carries token counts + timings.
        final = next(f for f in reversed(frames)
                     if isinstance(f, dict) and f.get("usage"))
        self.assertEqual(final["usage"]["completion_tokens"], 3)
        self.assertIn("timings", final)
        self.assertGreater(final["timings"]["predicted_per_second"], 0)

    def test_context_overflow_returns_400_before_daemon_call(self):
        active, daemon = self._install_fake_active([], cfg_overrides={"ctx-size": 4, "predict": 64})
        long_text = "x" * 100
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "dflash-test", "messages": [{"role": "user", "content": long_text}], "stream": True},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 400)
        # max-raw-ceiling defaults to ctx-size, so it now fires before any daemon work.
        self.assertIn("exceeds max raw ceiling", response.json()["detail"])
        # Daemon must not have been touched.
        self.assertIsNone(daemon.last_cmd_args)

    def test_effective_context_limit_allows_post_pflash_prompt(self):
        active, daemon = self._install_fake_active(
            [ord("o"), ord("k"), 0],
            cfg_overrides={
                "ctx-size": 10,
                "predict": 64,
                "max-effective-context": 4,
                "max-raw-ceiling": 4096,
            },
        )
        active.prefill_config = type(
            "PrefillCfg",
            (),
            {"enabled": True, "threshold": 1, "keep_ratio": 0.5, "drafter_path": "/dummy", "tail_budget": 512},
        )()

        async def fake_maybe_compress(prompt_ids, daemon, config, blocks=None):
            effective_ids = [1, 2, 3, 4]
            return effective_ids, True, entrypoint.materialize_blocks(effective_ids)

        with patch.object(entrypoint, "maybe_compress", fake_maybe_compress):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "dflash-test",
                    "messages": [{"role": "user", "content": "x" * 100}],
                    "max_tokens": 1,
                    "stream": False,
                },
                headers=self.auth,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(daemon.last_cmd_args["prompt_ids"], [1, 2, 3, 4])

    def test_effective_context_overflow_returns_400_after_pflash(self):
        active, daemon = self._install_fake_active(
            [],
            cfg_overrides={
                "ctx-size": 10,
                "predict": 64,
                "max-effective-context": 4,
                "max-raw-ceiling": 4096,
            },
        )
        active.prefill_config = type(
            "PrefillCfg",
            (),
            {"enabled": True, "threshold": 1, "keep_ratio": 0.5, "drafter_path": "/dummy", "tail_budget": 512},
        )()

        async def fake_maybe_compress(prompt_ids, daemon, config, blocks=None):
            effective_ids = [1, 2, 3, 4, 5]
            return effective_ids, True, entrypoint.materialize_blocks(effective_ids)

        with patch.object(entrypoint, "maybe_compress", fake_maybe_compress):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "dflash-test",
                    "messages": [{"role": "user", "content": "x" * 100}],
                    "max_tokens": 1,
                    "stream": True,
                },
                headers=self.auth,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("exceeds max effective context", response.json()["detail"])
        self.assertIsNone(daemon.last_cmd_args)

    def test_error_path_yields_done_terminator(self):
        active, daemon = self._install_fake_active([])
        daemon._running = False  # Force the "daemon not running" branch.
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "dflash-test", "messages": [{"role": "user", "content": "x"}], "stream": True},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("[DONE]", response.text)
        frames = _parse_sse(response.text)
        err_frame = next(f for f in frames if isinstance(f, dict) and f.get("error"))
        self.assertIn("not running", err_frame["error"]["message"])

    def test_non_streaming_returns_json_with_full_text(self):
        self._install_fake_active([ord("o"), ord("k"), 0])
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "dflash-test", "messages": [{"role": "user", "content": "ping"}], "stream": False},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    def test_session_kv_uses_inline_prompt_snapshot_and_restores_next_turn(self):
        active, daemon = self._install_fake_active([ord("x"), 0])
        active.session_kv = SessionKV(cap=2, prefix_cap=2)
        hist = self.client.post(
            "/history",
            json={"title": "session test", "model": "dflash-test", "messages": []},
            headers=self.auth,
        )
        self.assertEqual(hist.status_code, 200)
        conv_id = hist.json()["id"]

        first = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "dflash-test",
                "conversation_id": conv_id,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
            headers=self.auth,
        )
        self.assertEqual(first.status_code, 200)
        first_args = daemon.last_cmd_args
        self.assertEqual(first_args["snap_slot"], 2)
        self.assertEqual(first_args["snap_pos"], len(first_args["prompt_ids"]))
        self.assertEqual(
            active.session_kv.get_session(conv_id, first_args["prompt_ids"]),
            (2, len(first_args["prompt_ids"])),
        )

        daemon._tokens = [ord("y"), 0]
        second = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "dflash-test",
                "conversation_id": conv_id,
                "messages": [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "x"},
                    {"role": "user", "content": "again"},
                ],
                "stream": True,
            },
            headers=self.auth,
        )
        self.assertEqual(second.status_code, 200)
        second_args = daemon.last_cmd_args
        self.assertEqual(second_args["prefix_cache_slot"], 2)
        self.assertEqual(second_args["snap_slot"], 2)
        self.assertEqual(second_args["snap_pos"], len(second_args["prompt_ids"]))

    def test_proxy_v1_short_circuits_dflash_to_501(self):
        self._install_fake_active([])
        response = self.client.post(
            "/v1/completions",
            json={"model": "dflash-test", "prompt": "x"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 501)
        self.assertIn("not supported on dflash", response.json()["detail"])

    def test_non_streaming_assistant_tail_does_not_duplicate_previous_user_in_history(self):
        self._install_fake_active([ord("o"), ord("k"), 0])
        hist = self.client.post(
            "/history",
            json={"title": "assistant tail history", "model": "dflash-test", "messages": []},
            headers=self.auth,
        )
        self.assertEqual(hist.status_code, 200)
        conv_id = hist.json()["id"]

        payload = {
            "model": "dflash-test",
            "conversation_id": conv_id,
            "messages": [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "prior assistant"},
            ],
            "stream": False,
        }
        response = self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)

        conv = self.client.get(f"/history/{conv_id}", headers=self.auth).json()
        self.assertEqual([m["role"] for m in conv["messages"]], ["assistant"])
        self.assertEqual(conv["messages"][0]["content"], "ok")

    def test_real_session_passes_obsidian_protected_blocks_to_compression(self):
        active, _ = self._install_fake_active(
            [ord("o"), ord("k"), 0],
            cfg_overrides={"ctx-size": 200000, "predict": 64},
        )
        messages = _load_opencode_session_messages(
            "opencode_ses_1eb7_Update_machine_config_with_Obsidian_resume.json"
        )
        active.prefill_config = type(
            "PrefillCfg",
            (),
            {"enabled": True, "threshold": 1, "keep_ratio": 0.5, "drafter_path": "/dummy", "tail_budget": 512},
        )()

        calls = {}

        async def fake_maybe_compress(prompt_ids, daemon, config, blocks=None):
            calls["blocks"] = list(blocks or [])
            return prompt_ids, False, entrypoint.materialize_blocks(prompt_ids, blocks)

        with patch.object(entrypoint, "maybe_compress", fake_maybe_compress):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "dflash-test", "messages": messages, "stream": False},
                headers=self.auth,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("blocks", calls)
        protected = [(block.start, block.end) for block in calls["blocks"] if block.protected and block.kind == "tool"]
        boundaries = [block.end for block in calls["blocks"] if block.kind != "generation_prompt"]
        self.assertTrue(protected)
        self.assertEqual(protected, _expected_protected_tool_ranges(messages, boundaries))


class OpenCodeSessionReplayTests(unittest.TestCase):
    """Stress replay real OpenCode sessions across history + session-KV using the real Qwen tokenizer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "replay-history.sqlite3")
        self._old_store = entrypoint.history_store
        entrypoint.history_store = HistoryStore(self.db_path)
        self._old_api = entrypoint.API_KEY
        self._old_admin = entrypoint.ADMIN_TOKEN
        entrypoint.API_KEY = "test-key"
        entrypoint.ADMIN_TOKEN = "test-key"
        entrypoint.manager.active.clear()
        self.user_hash = identity_hash("test-key")

    def tearDown(self):
        entrypoint.history_store = self._old_store
        entrypoint.API_KEY = self._old_api
        entrypoint.ADMIN_TOKEN = self._old_admin
        entrypoint.manager.active.clear()
        self.tmp.cleanup()

    def _install_real_tokenizer_active(self, tokens, cfg_overrides=None):
        tokenizer = _get_real_qwen_tokenizer()
        cfg = {
            "backend": "dflash",
            "ctx-size": 1_048_576,
            "predict": 64,
            "family": "qwen",
        }
        if cfg_overrides:
            cfg.update(cfg_overrides)
        daemon = FakeDflashDaemon(tokens)
        active = FakeActive("dflash-replay", cfg, daemon, tokenizer)
        active.session_kv = SessionKV(cap=2, prefix_cap=2)

        async def fake_start_model(model_name):
            return active

        if "dflash-replay" not in entrypoint.registry.list_all():
            entrypoint.registry.add("dflash-replay", cfg)
        self.addCleanup(self._remove_alias, "dflash-replay")
        self._patch_start = patch.object(entrypoint.manager, "start_model", fake_start_model)
        self._patch_start.start()
        self.addCleanup(self._patch_start.stop)
        return active, daemon

    def _remove_alias(self, name):
        try:
            entrypoint.registry.remove(name)
        except KeyError:
            pass

    def _create_conversation(self, title):
        return entrypoint.history_store.create_conversation(
            self.user_hash,
            title=title,
            model="dflash-replay",
            messages=[],
        )["id"]

    def _history_messages(self, conversation_id):
        return entrypoint.history_store.get_conversation_tree(self.user_hash, conversation_id)["messages"]

    def _history_tail(self, conversation_id, limit=2):
        return _history_tail_for_test(entrypoint.history_store, conversation_id, limit=limit)

    def _assert_history_tail(self, conversation_id, expected_count, response_text):
        count, tail = self._history_tail(conversation_id, limit=1)
        self.assertEqual(count, expected_count)
        self.assertEqual(tail[-1]["role"], "assistant")
        self.assertEqual(tail[-1]["content"], response_text)

    def test_replay_all_session_files_across_every_turn(self):
        session_files = _list_opencode_session_files()
        self.assertTrue(session_files)
        max_workers = min(len(session_files), 3)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_replay_session_file_worker, path.name): path.name
                for path in session_files
            }
            completed = 0
            total_turns = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                total_turns += result["turns"]
                self.assertGreaterEqual(result["turns"], 0, result["filename"])

        self.assertEqual(completed, len(session_files))
        self.assertGreater(total_turns, 0)


if __name__ == "__main__":
    unittest.main()
