"""Tests for KVCacheStore: content-hash KV cache with RAM→disk tiering."""

import hashlib
import json
import os
import struct
import tempfile
import time
from pathlib import Path

from grimoire.dflash.kv_cache_store import KVCacheStore, KV_PREFIX, KV_SUFFIX


def _make_ids(n: int, seed: int = 0) -> list:
    return [(seed + i) % 32000 for i in range(n)]


class TestHashPrefix:
    def test_deterministic(self):
        store = KVCacheStore()
        ids = _make_ids(100)
        assert store.hash_prefix(ids) == store.hash_prefix(ids)

    def test_different_inputs_different_hashes(self):
        store = KVCacheStore()
        assert store.hash_prefix(_make_ids(100, 0)) != store.hash_prefix(_make_ids(100, 1))

    def test_different_lengths_different_hashes(self):
        store = KVCacheStore()
        assert store.hash_prefix(_make_ids(50)) != store.hash_prefix(_make_ids(100))

    def test_16_bytes(self):
        store = KVCacheStore()
        h = store.hash_prefix(_make_ids(100))
        assert len(h) == 16

    def test_includes_kv_type(self):
        a = KVCacheStore(kv_k_type="q8_0", kv_v_type="q8_0")
        b = KVCacheStore(kv_k_type="turbo4", kv_v_type="turbo4")
        assert a.hash_prefix(_make_ids(100)) != b.hash_prefix(_make_ids(100))

    def test_includes_fa_window(self):
        a = KVCacheStore(fa_window=2048)
        b = KVCacheStore(fa_window=4096)
        assert a.hash_prefix(_make_ids(100)) != b.hash_prefix(_make_ids(100))


class TestKvFilename:
    def test_format(self):
        store = KVCacheStore()
        h = bytes(16)
        name = store.kv_filename(h)
        assert name.startswith(KV_PREFIX)
        assert name.endswith(KV_SUFFIX)
        assert len(name) == len(KV_PREFIX) + 16 + len(KV_SUFFIX)

    def test_paths(self):
        store = KVCacheStore(ram_dir="/tmp/ram", disk_dir="/tmp/disk")
        h = b"\x01" * 16
        rp = store.ram_path(h)
        dp = store.disk_path(h)
        assert str(rp).startswith("/tmp/ram")
        assert str(dp).startswith("/tmp/disk")
        assert rp.name == dp.name

    def test_disk_path_fallback(self):
        store = KVCacheStore(ram_dir="/tmp/ram")
        h = b"\x01" * 16
        dp = store.disk_path(h)
        assert str(dp).startswith("/tmp/ram")


class TestLookup:
    def test_ram_hit(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d, cap=4)
            h = b"\xaa" * 16
            p = store.ram_path(h)
            p.write_text("data")
            store.register(h)
            result = store.lookup(h)
            assert result is not None
            assert result.name == p.name

    def test_ram_miss(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d)
            h = b"\xbb" * 16
            assert store.lookup(h) is None

    def test_disk_fallback(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk, cap=4)
            h = b"\xcc" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("disk data")
            result = store.lookup(h)
            assert result is not None
            assert result.name == dp.name

    def test_bumps_lru_on_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d, cap=2)
            h1, h2, h3 = b"\x01" * 16, b"\x02" * 16, b"\x03" * 16
            for h in (h1, h2):
                p = store.ram_path(h)
                p.write_text("data")
                store.register(h)
            assert store.lookup(h1) is not None
            store.ram_path(h3).write_text("x")
            store.register(h3)
            assert store.lookup(h1) is not None
            assert store.lookup(h2) is None


class TestRegister:
    def test_registers_ram_path(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d, cap=4)
            h = b"\xdd" * 16
            p = store.ram_path(h)
            p.write_text("data")
            store.register(h)
            assert h in store.ram

    def test_ignores_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d)
            h = b"\xee" * 16
            store.register(h)
            assert h not in store.ram

    def test_evicts_lru_when_over_cap(self):
        with tempfile.TemporaryDirectory() as d:
            store = KVCacheStore(ram_dir=d, cap=2)
            h1, h2, h3 = b"\x10" * 16, b"\x20" * 16, b"\x30" * 16
            for h in (h1, h2):
                store.ram_path(h).write_text("x")
                store.register(h)
            store.ram_path(h3).write_text("x")
            store.register(h3)
            assert store.lookup(h1) is None
            assert store.lookup(h3) is not None


class TestPromoteToRam:
    def test_copies_disk_to_ram(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xff" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("disk content")
            rp = store.promote_to_ram(h, dp)
            assert rp is not None
            assert rp.exists()
            assert rp.read_text() == "disk content"

    def test_registers_in_ram_index(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xee" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("x")
            store.promote_to_ram(h, dp)
            assert h in store.ram


class TestManifest:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xab" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("persistent")
            store.disk[h] = dp
            store._save_manifest()

            store2 = KVCacheStore(ram_dir=ram, disk_dir=disk)
            assert h in store2.disk

    def test_load_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xac" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("x")
            store.disk[h] = dp
            store._save_manifest()
            dp.unlink()

            store2 = KVCacheStore(ram_dir=ram, disk_dir=disk)
            assert h not in store2.disk
            assert not store2.disk_path(h).exists()


class TestDiskCleanup:
    def test_removes_stray_files(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            stray = Path(disk) / f"{KV_PREFIX}deadbeef{KV_SUFFIX}"
            stray.write_text("stray")
            store._cleanup_disk()
            assert not stray.exists()

    def test_keeps_tracked_files(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xad" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("tracked")
            store.disk[h] = dp
            store._cleanup_disk()
            assert dp.exists()

    def test_evicts_lru_when_over_budget(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk, disk_budget_gb=0.0001)
            h1 = b"\xae" * 16
            h2 = b"\xaf" * 16
            dp1 = store.disk_path(h1)
            dp1.parent.mkdir(parents=True, exist_ok=True)
            dp1.write_text("x" * 200000)
            store.disk[h1] = dp1
            dp2 = store.disk_path(h2)
            dp2.write_text("x" * 200000)
            store.disk[h2] = dp2
            store._cleanup_disk()
            assert len(store.disk) <= 1


class TestIntegration:
    def test_coding_agent_scenario(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk, cap=4)
            sysprompt = _make_ids(5000, 0)
            user_msg = _make_ids(100, 1)

            h1 = store.hash_prefix(sysprompt)
            assert store.lookup(h1) is None

            fp = store.ram_path(h1)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"x" * 1024)
            store.register(h1)

            assert store.lookup(h1) is not None

            h2 = store.hash_prefix(sysprompt + user_msg)
            assert h2 != h1
            assert store.lookup(h2) is None

    def test_two_conversations_same_prompt(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk, cap=4)
            sysprompt = _make_ids(5000, 0)

            h = store.hash_prefix(sysprompt)
            fp = store.ram_path(h)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"x" * 1024)
            store.register(h)

            assert store.lookup(h) is not None

            conv2_path = store.lookup(h)
            assert conv2_path is not None
            assert conv2_path.name == fp.name

    def test_disk_survives_restart(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xba" * 16
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("x")
            store.disk[h] = dp
            store._save_manifest()

            store2 = KVCacheStore(ram_dir=ram, disk_dir=disk)
            result = store2.lookup(h)
            assert result is not None
            assert result.name == dp.name


class TestDiscard:
    def test_removes_from_ram_and_disk(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xca" * 16
            rp = store.ram_path(h)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text("ram")
            store.ram[h] = rp
            dp = store.disk_path(h)
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text("disk")
            store.disk[h] = dp

            store.discard(h)
            assert h not in store.ram
            assert h not in store.disk
            assert not rp.exists()
            assert not dp.exists()


class TestClear:
    def test_empties_directories(self):
        with tempfile.TemporaryDirectory() as ram, tempfile.TemporaryDirectory() as disk:
            store = KVCacheStore(ram_dir=ram, disk_dir=disk)
            h = b"\xda" * 16
            store.ram_path(h).parent.mkdir(parents=True, exist_ok=True)
            store.ram_path(h).write_text("x")
            store.disk_path(h).parent.mkdir(parents=True, exist_ok=True)
            store.disk_path(h).write_text("x")
            store.ram[h] = store.ram_path(h)
            store.disk[h] = store.disk_path(h)

            store.clear()
            assert len(store.ram) == 0
            assert len(store.disk) == 0
            assert store.ram_dir.exists()
