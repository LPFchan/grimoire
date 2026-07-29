"""Ordered multi-GPU placement, validation, lifecycle, and launcher tests."""

import argparse
import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import grimoire.launcher as launcher
import grimoire.model_manager as mm
from grimoire.model_manager import (
    ActiveModel,
    GpuPlacement,
    ModelManager,
    configure_gpu_environment,
    validate_multi_gpu_selectors,
)
from grimoire.proxy.llama import _telemetry_gpu_index
from grimoire.registry import ModelRegistry


def run(coro):
    return asyncio.run(coro)


class FakeActive:
    def __init__(self, name, gpus, port, stop_error=None):
        self.name = name
        self.gpus = list(gpus)
        self.gpu = self.gpus[0]
        self.port = port
        self.cfg = {}
        self.started = datetime.now(timezone.utc)
        self.stop_error = stop_error
        self.stop_calls = 0

    def is_running(self):
        return True

    def stop(self):
        self.stop_calls += 1
        if self.stop_error:
            raise self.stop_error


class PlacementTests(unittest.TestCase):
    def test_active_model_preserves_scalar_and_full_placement(self):
        scalar = ActiveModel("scalar", {}, 8001, 1)
        self.assertEqual(scalar.gpu, 1)
        self.assertEqual(scalar.gpus, [1])

        sharded = ActiveModel("sharded", {}, 8001, GpuPlacement((1, 0)))
        self.assertEqual(sharded.gpu, 1)
        self.assertEqual(sharded.gpus, [1, 0])

        source = [0, 1]
        immutable = GpuPlacement(source)
        source.append(2)
        self.assertEqual(immutable.device_ids, (0, 1))

    def test_ordered_visibility_and_selector_env_sanitization(self):
        env = {
            "CUDA_VISIBLE_DEVICES": "9",
            "LLAMA_ARG_SPLIT_MODE": "none",
            "LLAMA_ARG_TENSOR_SPLIT": "9,1",
            "LLAMA_ARG_MAIN_GPU": "1",
        }
        cfg = {"extra-args": ["--tensor-split", "3,1", "--main-gpu=1"]}
        configure_gpu_environment(env, cfg, GpuPlacement((1, 0)))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,0")
        for key in ("LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_TENSOR_SPLIT", "LLAMA_ARG_MAIN_GPU"):
            self.assertNotIn(key, env)

    def test_selector_validation(self):
        placement = GpuPlacement((0, 1))
        valid = [
            {},
            {"extra-args": ["--split-mode", "layer"]},
            {"extra-args": ["-sm=row", "-ts", "2,1", "-mg", "1"]},
        ]
        for cfg in valid:
            validate_multi_gpu_selectors(cfg, placement)

        invalid = [
            {"extra-args": ["--split-mode", "none"]},
            {"extra-args": ["--tensor-split", "1"]},
            {"extra-args": ["--tensor-split", "1,nan"]},
            {"extra-args": ["--main-gpu", "2"]},
            {"extra-args": ["--split-mode", "layer", "-sm", "row"]},
            {"extra-args": ["--device", "CUDA0"]},
        ]
        for cfg in invalid:
            with self.assertRaises(ValueError, msg=cfg):
                validate_multi_gpu_selectors(cfg, placement)

        with patch.object(mm.registry, "get_family_defaults", return_value={"extra-args": ["--tensor-split", "1"]}):
            with self.assertRaisesRegex(ValueError, "has 1 values"):
                validate_multi_gpu_selectors({"family": "test"}, placement)

    def test_registry_gpu_ids_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model.gguf").touch()
            with patch("grimoire.registry.MODELS_DIR", td):
                registry = ModelRegistry(path=str(root / "registry.json"), seed_path="")
                registry.add("model", {"file": "model.gguf", "gpu-ids": [0, 1]})
                self.assertEqual(registry.validate("model", gpu_count=2), (True, "OK"))
                registry.pin_gpu("model", 0)
                self.assertEqual(registry.get_fixed_gpu("model"), 0)
                with self.assertRaisesRegex(ValueError, "primary"):
                    registry.pin_gpu("model", 1)
                registry._data["models"]["model"]["gpu-ids"] = []
                with self.assertRaisesRegex(ValueError, "invalid 'gpu-ids'"):
                    registry.pin_gpu("model", 0)
                registry._data["models"]["model"]["gpu-ids"] = [0, 1]
                registry._data["fixed"]["model"] = 1
                self.assertFalse(registry.validate("model", gpu_count=2)[0])
                registry._data["fixed"]["model"] = 0

                invalid_values = ([0], [0, 0], [0, True], [0, 2])
                for value in invalid_values:
                    registry.update("model", {"gpu-ids": value})
                    self.assertFalse(registry.validate("model", gpu_count=2)[0], value)

                for field, value in (
                    ("cpu-only", True),
                    ("vram-budget-mib", 1),
                    ("pflash", True),
                    ("park-unpark", True),
                    ("speculative-type", "dflash"),
                ):
                    cfg = {"file": "model.gguf", "gpu-ids": [0, 1], field: value}
                    registry._data["models"]["model"] = cfg
                    self.assertFalse(registry.validate("model", gpu_count=2)[0], field)

    def test_explicit_allocation_dedupes_incumbents_and_blocks_fixed(self):
        manager = ModelManager(gpu_count=2)
        spanning = FakeActive("spanning", [0, 1], 8001)
        manager.active = {"spanning": spanning}
        registry = MagicMock()
        registry.get_fixed_gpu.return_value = None
        registry.is_fixed.return_value = False
        with patch.object(mm, "registry", registry):
            placement, victims = run(manager._allocate_gpu("new", {"gpu-ids": [0, 1]}))
        self.assertEqual(placement.device_ids, (0, 1))
        self.assertEqual(victims, [("spanning", spanning)])

        registry.is_fixed.return_value = True
        with patch.object(mm, "registry", registry):
            with self.assertRaisesRegex(RuntimeError, "Cannot evict pinned model"):
                run(manager._allocate_gpu("new", {"gpu-ids": [0, 1]}))

    def test_explicit_allocation_honors_full_gpu_mask(self):
        manager = ModelManager(gpu_count=2)
        manager.gpu_mask = {0}
        registry = MagicMock()
        registry.get_fixed_gpu.return_value = None
        with patch.object(mm, "registry", registry):
            with self.assertRaisesRegex(RuntimeError, "excluded by active GPU mask"):
                run(manager._allocate_gpu("new", {"gpu-ids": [0, 1]}))

    def test_scalar_replacement_can_evict_sharded_incumbent_from_allowed_member(self):
        manager = ModelManager(gpu_count=2)
        manager.gpu_mask = {1}
        spanning = FakeActive("spanning", [0, 1], 8001)
        manager.active = {"spanning": spanning}
        registry = MagicMock()
        registry.get_fixed_gpu.return_value = None
        registry.is_fixed.return_value = False
        with patch.object(mm, "registry", registry):
            placement, victims = run(manager._allocate_gpu("new", {}))
        self.assertEqual(placement.device_ids, (1,))
        self.assertEqual(victims, [("spanning", spanning)])

    def test_budgeted_model_cannot_colocate_with_pinned_sharded_incumbent(self):
        manager = ModelManager(gpu_count=2)
        spanning = FakeActive("spanning", [0, 1], 8001)
        manager.active = {"spanning": spanning}
        registry = MagicMock()
        registry.get_fixed_gpu.return_value = 1
        registry.is_fixed.side_effect = lambda name: name == "spanning"
        with patch.object(mm, "registry", registry), \
             patch.object(manager, "_get_gpu_free_vram_mib", return_value=24000):
            with self.assertRaisesRegex(RuntimeError, "exclusively occupied"):
                run(manager._allocate_gpu("budgeted", {"vram-budget-mib": 1000}))

    def test_partial_stop_failure_rolls_back_only_stopped_victims(self):
        manager = ModelManager(gpu_count=2)
        first = FakeActive("first", [0], 8001)
        second = FakeActive("second", [1], 8011, stop_error=RuntimeError("stop failed"))
        manager.active = {"first": first, "second": second}
        cfg = {"file": "model.gguf", "gpu-ids": [0, 1]}
        registry = MagicMock()
        registry.resolve.return_value = "new"
        registry.get.return_value = cfg
        registry.validate.return_value = (True, "OK")
        registry.get_fixed_gpu.return_value = None
        registry.is_fixed.return_value = False
        restored = []

        async def restore(victims):
            restored.extend(name for name, _ in victims)
            return []

        with patch.object(mm, "registry", registry), \
             patch.object(manager, "_restore_incumbents", new=AsyncMock(side_effect=restore)):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                run(manager.start_model("new"))

        self.assertEqual(first.stop_calls, 1)
        self.assertEqual(second.stop_calls, 1)
        self.assertEqual(restored, ["first"])
        self.assertNotIn("new", manager.active)

    def test_multi_gpu_telemetry_is_suppressed(self):
        scalar = MagicMock(gpu=0, gpus=[0])
        sharded = MagicMock(gpu=0, gpus=[0, 1])
        self.assertEqual(_telemetry_gpu_index(scalar), 0)
        self.assertIsNone(_telemetry_gpu_index(sharded))

    def test_launcher_uses_ordered_multi_gpu_visibility(self):
        cfg = {
            "file": "model.gguf",
            "gpu-ids": [1, 0],
            "extra-args": ["--tensor-split", "3,1"],
        }
        args = argparse.Namespace(model="model", port=8123, ctx_size=None, gpu=None)
        completed = MagicMock(returncode=0)
        with patch.object(launcher, "parse_args", return_value=args), \
             patch.object(launcher, "detect_gpu_count", return_value=2), \
             patch.object(launcher.registry, "get", return_value=cfg), \
             patch.object(launcher.registry, "get_fixed_gpu", return_value=None), \
             patch.object(launcher.registry, "validate", return_value=(True, "OK")), \
             patch.object(launcher.os.path, "exists", return_value=True), \
             patch.object(launcher.subprocess, "run", return_value=completed) as run_mock:
            launcher.main()
        self.assertEqual(run_mock.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,0")


if __name__ == "__main__":
    unittest.main()
