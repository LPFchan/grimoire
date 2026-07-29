"""GPU allocation tests for ModelManager: exclusive vs budgeted co-location."""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GRIMOIRE_REGISTRY_SEED_PATH", str(ROOT / "etc" / "models.json"))
os.environ.setdefault("GRIMOIRE_REGISTRY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-alloc-registry.json"))

import grimoire.model_manager as mm
from grimoire.model_manager import ModelManager


class FakeActive:
    def __init__(self, name, gpu, port, started=None, cfg=None):
        self.name = name
        self.gpu = gpu
        self.port = port
        self.started = started or datetime.now(timezone.utc)
        # An incumbent is classified budgeted/exclusive by its stored cfg.
        self.cfg = cfg if cfg is not None else {}

    def is_running(self):
        return True

    def stop(self):
        pass


BUDGETED = {"vram-budget-mib": 1500}


class FakeRegistry:
    """Minimal stand-in for the registry's pin lookups."""

    def __init__(self, fixed=None):
        self._fixed = fixed or {}

    def is_fixed(self, name):
        return name in self._fixed

    def get_fixed_gpu(self, name):
        return self._fixed.get(name)


def run(coro):
    return asyncio.run(coro)


class AllocatorTests(unittest.TestCase):
    def setUp(self):
        self.mgr = ModelManager(gpu_count=2)

    def _patch_registry(self, fixed=None):
        return patch.object(mm, "registry", FakeRegistry(fixed))

    def _patch_vram(self, per_gpu):
        return patch.object(self.mgr, "_get_gpu_free_vram_mib", side_effect=lambda g: per_gpu.get(g))

    # ---- unbudgeted (exclusive, backward-compatible) ----

    def test_exclusive_unpinned_prefers_free_gpu(self):
        self.mgr.active = {"a": FakeActive("a", gpu=0, port=8001)}
        with self._patch_registry():
            gpu, evict = run(self.mgr._allocate_gpu("new", {}))
        self.assertEqual(gpu.primary, 1)
        self.assertEqual(evict, [])

    def test_exclusive_pinned_refuses_pinned_exclusive_incumbent(self):
        # A pinned *unbudgeted* incumbent still blocks (exclusive owns the GPU).
        self.mgr.active = {"pin": FakeActive("pin", gpu=1, port=8011)}
        with self._patch_registry(fixed={"pin": 1, "x": 1}):
            with self.assertRaises(RuntimeError) as ctx:
                run(self.mgr._allocate_gpu("x", {}))
        self.assertIn("Cannot evict pinned model", str(ctx.exception))

    def test_exclusive_pinned_colocates_over_budgeted_pinned_incumbents(self):
        # Auto co-locate: an unbudgeted model pinned to GPU 1 co-locates on top of
        # the pinned budgeted always-on pair without evicting them or erroring.
        self.mgr.active = {
            "emb": FakeActive("emb", gpu=1, port=8011, cfg=BUDGETED),
            "rer": FakeActive("rer", gpu=1, port=8012, cfg=BUDGETED),
        }
        with self._patch_registry(fixed={"emb": 1, "rer": 1, "chat": 1}):
            gpu, evict = run(self.mgr._allocate_gpu("chat", {}))
        self.assertEqual(gpu.primary, 1)
        self.assertEqual(evict, [])

    def test_exclusive_unpinned_prefers_empty_over_budgeted_only(self):
        # GPU 1 has only budgeted co-tenants; GPU 0 is truly empty -> pick GPU 0.
        self.mgr.active = {"emb": FakeActive("emb", gpu=1, port=8011, cfg=BUDGETED)}
        with self._patch_registry(fixed={"emb": 1}):
            gpu, evict = run(self.mgr._allocate_gpu("chat", {}))
        self.assertEqual(gpu.primary, 0)
        self.assertEqual(evict, [])

    def test_exclusive_unpinned_swaps_exclusive_rather_than_cramming_budgeted_gpu(self):
        # Regression guard: GPU 0 runs chat A (exclusive); GPU 1 holds the pinned
        # budgeted pair. Loading chat B (unpinned, exclusive) must SWAP it onto
        # GPU 0 (evict A) rather than cram it onto GPU 1's spare VRAM and OOM.
        self.mgr.active = {
            "chatA": FakeActive("chatA", gpu=0, port=8001),
            "embA": FakeActive("embA", gpu=1, port=8011, cfg=BUDGETED),
            "embB": FakeActive("embB", gpu=1, port=8012, cfg=BUDGETED),
        }
        with self._patch_registry(fixed={"embA": 1, "embB": 1}):
            gpu, evict = run(self.mgr._allocate_gpu("chatB", {}))
        self.assertEqual(gpu.primary, 0)
        self.assertEqual([n for n, _ in evict], ["chatA"])

    def test_exclusive_unpinned_colocates_on_budgeted_only_as_last_resort(self):
        # No empty GPU and no exclusive to evict (both GPUs are budgeted-only):
        # co-locating over budgeted co-tenants is the last resort.
        self.mgr.active = {
            "embA": FakeActive("embA", gpu=0, port=8001, cfg=BUDGETED),
            "embB": FakeActive("embB", gpu=1, port=8011, cfg=BUDGETED),
        }
        with self._patch_registry(fixed={"embA": 0, "embB": 1}):
            gpu, evict = run(self.mgr._allocate_gpu("chat", {}))
        self.assertEqual(gpu.primary, 0)
        self.assertEqual(evict, [])

    def test_exclusive_unpinned_evicts_oldest_exclusive_when_all_full(self):
        # Both GPUs host a non-pinned exclusive incumbent -> evict the oldest.
        old = FakeActive("old", gpu=0, port=8001, started=datetime(2020, 1, 1, tzinfo=timezone.utc))
        new = FakeActive("new", gpu=1, port=8011, started=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.mgr.active = {"old": old, "new": new}
        with self._patch_registry():
            gpu, evict = run(self.mgr._allocate_gpu("x", {}))
        self.assertEqual(gpu.primary, 0)
        self.assertEqual([n for n, _ in evict], ["old"])

    # ---- budgeted (co-location) ----

    def test_budgeted_pinned_colocates_when_vram_fits(self):
        self.mgr.active = {"emb": FakeActive("emb", gpu=1, port=8011)}
        cfg = {"vram-budget-mib": 1500}
        with self._patch_registry(fixed={"emb": 1, "rer": 1}), self._patch_vram({1: 22000}):
            gpu, evict = run(self.mgr._allocate_gpu("rer", cfg))
        self.assertEqual(gpu.primary, 1)
        self.assertEqual(evict, [])

    def test_budgeted_pinned_refuses_when_only_pinned_incumbents(self):
        self.mgr.active = {
            "emb": FakeActive("emb", gpu=1, port=8011),
            "rer": FakeActive("rer", gpu=1, port=8012),
        }
        cfg = {"vram-budget-mib": 20500}
        with self._patch_registry(fixed={"emb": 1, "rer": 1, "chat": 1}), self._patch_vram({1: 18000}):
            with self.assertRaises(RuntimeError) as ctx:
                run(self.mgr._allocate_gpu("chat", cfg))
        self.assertIn("< budget", str(ctx.exception))

    def test_budgeted_pinned_evicts_nonpinned_when_needed(self):
        self.mgr.active = {"junk": FakeActive("junk", gpu=1, port=8011)}
        cfg = {"vram-budget-mib": 20500}
        with self._patch_registry(fixed={"chat": 1}), self._patch_vram({1: 4000}):
            gpu, evict = run(self.mgr._allocate_gpu("chat", cfg))
        self.assertEqual(gpu.primary, 1)
        self.assertEqual([n for n, _ in evict], ["junk"])

    def test_budgeted_unpinned_prefers_no_eviction_gpu(self):
        # GPU 0 is tight with an evictable incumbent; GPU 1 fits cleanly.
        self.mgr.active = {"junk": FakeActive("junk", gpu=0, port=8001)}
        cfg = {"vram-budget-mib": 20500}
        with self._patch_registry(), self._patch_vram({0: 4000, 1: 22000}):
            gpu, evict = run(self.mgr._allocate_gpu("chat", cfg))
        self.assertEqual(gpu.primary, 1)
        self.assertEqual(evict, [])

    def test_budgeted_nvidia_smi_failure_refuses_gpu(self):
        cfg = {"vram-budget-mib": 1500}
        with self._patch_registry(fixed={"x": 0}), self._patch_vram({0: None}):
            with self.assertRaises(RuntimeError) as ctx:
                run(self.mgr._allocate_gpu("x", cfg))
        self.assertIn("nvidia-smi query failed", str(ctx.exception))

    def test_start_model_post_eviction_recheck_aborts_and_restores_incumbent(self):
        # Budgeted model pinned to GPU 0 forces eviction of a non-pinned incumbent
        # (Pass 1 short on VRAM). The post-eviction re-check is still short, so the
        # start aborts and the evicted incumbent is restored.
        cfg = {"vram-budget-mib": 20500, "file": "x"}
        junk = FakeActive("junk", gpu=0, port=8001)
        self.mgr.active = {"junk": junk}

        reg = MagicMock()
        reg.resolve.side_effect = lambda n: n
        reg.get.return_value = cfg
        reg.validate.return_value = (True, "OK")
        reg.get_fixed_gpu.side_effect = lambda n: 0 if n == "chat" else None
        reg.is_fixed.side_effect = lambda n: n == "chat"  # junk is evictable

        restored = []
        start_calls = []

        async def fake_start(active):
            start_calls.append(active.name)

        async def fake_restore(incumbents):
            restored.extend(n for n, _ in incumbents)
            return []

        with patch.object(mm, "registry", reg), \
             patch.object(self.mgr, "_get_gpu_free_vram_mib", side_effect=lambda g: 4000), \
             patch.object(self.mgr, "_start_active_model", new=AsyncMock(side_effect=fake_start)), \
             patch.object(self.mgr, "_restore_incumbents", new=AsyncMock(side_effect=fake_restore)), \
             patch("grimoire.model_manager.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(RuntimeError) as ctx:
                run(self.mgr.start_model("chat"))

        self.assertIn("Post-eviction VRAM check failed", str(ctx.exception))
        self.assertEqual(restored, ["junk"])          # evicted incumbent restored
        self.assertNotIn("chat", self.mgr.active)      # failed model not left active
        self.assertEqual(start_calls, [])              # new model never launched

    def test_model_budget_mib_parsing(self):
        self.assertEqual(ModelManager._model_budget_mib({"vram-budget-mib": 1500}), 1500)
        self.assertIsNone(ModelManager._model_budget_mib({}))
        self.assertIsNone(ModelManager._model_budget_mib({"vram-budget-mib": 0}))
        self.assertIsNone(ModelManager._model_budget_mib({"vram-budget-mib": "1500"}))
        # bool is an int subclass; must not read True as budget=1
        self.assertIsNone(ModelManager._model_budget_mib({"vram-budget-mib": True}))


if __name__ == "__main__":
    unittest.main()
