"""Runtime-only clone/declone/pin/unpin and preset reconciliation tests."""

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import grimoire.model_manager as mm
from grimoire.model_manager import GpuPlacement, ModelManager
from grimoire.presets import PresetManager


def run(coro):
    return asyncio.run(coro)


class FakeActive:
    def __init__(self, name, gpus=(0,), port=8001):
        self.name = name
        self.gpus = list(gpus)
        self.gpu = self.gpus[0]
        self.port = port
        self.cfg = {"file": "model.gguf"}
        self.started = datetime.now(timezone.utc)
        self.status = "loaded"
        self.stop_calls = 0

    def is_running(self):
        return True

    def stop(self):
        self.stop_calls += 1


class RuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager(gpu_count=2)
        self.registry = MagicMock()
        self.registry.resolve.side_effect = lambda name: name if name == "model" else None
        self.registry.get.return_value = {"file": "model.gguf"}
        self.registry.validate.return_value = (True, "OK")
        self.registry.get_fixed_gpu.return_value = None
        self.registry.is_fixed.return_value = False
        self.registry.get_family_defaults.return_value = {}

    def test_pin_unloaded_and_explicit_unpin_over_registry_pin(self):
        with patch.object(mm, "registry", self.registry):
            active, metadata = run(self.manager.pin_model("model", 1))
            self.assertIsNone(active)
            self.assertEqual(metadata["gpu"], None)
            self.assertEqual(metadata["gpus"], [])
            self.assertEqual(metadata["requested_gpus"], [1])
            self.assertEqual(metadata["pin_source"], "runtime")

            self.registry.get_fixed_gpu.return_value = 0
            self.registry.is_fixed.return_value = True
            _, metadata = run(self.manager.unpin_model("model"))
            self.assertFalse(metadata["pinned"])
            self.assertEqual(metadata["pin_source"], "runtime")

    def test_clone_unloaded_loads_and_declone_unloaded_only_clears(self):
        async def launch(name):
            active = FakeActive(name, (0, 1))
            self.manager.active[name] = active
            return active

        with patch.object(mm, "registry", self.registry), \
                patch.object(self.manager, "_start_model_locked", new=AsyncMock(side_effect=launch)):
            active, metadata = run(self.manager.clone_model("model", [0, 1], [1, 1]))
            self.assertEqual(active.gpus, [0, 1])
            self.assertEqual(metadata["placement_source"], "runtime")
            self.manager.active.clear()
            active, metadata = run(self.manager.declone_model("model"))
            self.assertIsNone(active)
            self.assertIsNone(metadata["runtime_override"])

    def test_clone_validation_and_tensor_selector_conflict(self):
        with patch.object(mm, "registry", self.registry):
            with self.assertRaisesRegex(ValueError, "one value"):
                run(self.manager.clone_model("model", [0, 1], [1]))
            with self.assertRaisesRegex(ValueError, "positive total"):
                run(self.manager.clone_model("model", [0, 1], [0, 0]))
            for value in ([True, 1], ["1", 1], [object(), 1]):
                with self.assertRaisesRegex(ValueError, "numeric values"):
                    run(self.manager.clone_model("model", [0, 1], value))
            self.registry.get.return_value = {"file": "model.gguf", "extra-args": ["--tensor-split", "2,1"]}
            with self.assertRaisesRegex(ValueError, "conflicts"):
                run(self.manager.clone_model("model", [0, 1], [1, 1]))

    def test_durable_sharding_survives_runtime_unpin_metadata(self):
        self.registry.get.return_value = {"file": "model.gguf", "gpu-ids": [1, 0]}
        self.registry.get_fixed_gpu.return_value = 1
        self.registry.is_fixed.return_value = True
        with patch.object(mm, "registry", self.registry):
            _, metadata = run(self.manager.unpin_model("model"))
        self.assertEqual(metadata["requested_gpus"], [1, 0])
        self.assertEqual(metadata["placement_source"], "registry")
        self.assertFalse(metadata["pinned"])
        self.assertEqual(metadata["pin_source"], "runtime")

    def test_override_publication_waits_for_replacement_success(self):
        async def scenario():
            old = FakeActive("model", (0,))
            replacement = FakeActive("model", (0, 1))
            self.manager.active = {"model": old}
            self.manager._runtime_overrides["model"] = mm.RuntimeModelOverride(pin_state=0)
            startup_entered = asyncio.Event()
            allow_startup = asyncio.Event()

            async def launch(_name):
                startup_entered.set()
                await allow_startup.wait()
                self.manager.active["model"] = replacement
                return replacement

            with patch.object(mm, "registry", self.registry), \
                    patch.object(self.manager, "_start_model_locked", new=AsyncMock(side_effect=launch)):
                task = asyncio.create_task(self.manager.clone_model("model", [0, 1]))
                await startup_entered.wait()
                pending = self.manager.override_metadata("model")
                self.assertEqual(pending["runtime_override"]["gpu_ids"], None)
                self.assertEqual(pending["runtime_override"]["pin"], 0)
                self.assertEqual(pending["requested_gpus"], [0])
                allow_startup.set()
                await task
                committed = self.manager.override_metadata("model")
                self.assertEqual(committed["runtime_override"]["gpu_ids"], [0, 1])
                self.assertEqual(committed["requested_gpus"], [0, 1])

        run(scenario())

    def test_active_clone_rolls_back_backend_and_override_on_start_failure(self):
        old = FakeActive("model", (0,))
        self.manager.active = {"model": old}
        restored = []

        async def restore(active):
            restored.append(active)

        with patch.object(mm, "registry", self.registry), \
                patch.object(self.manager, "_start_model_locked", new=AsyncMock(side_effect=RuntimeError("start failed"))), \
                patch.object(self.manager, "_start_active_model", new=AsyncMock(side_effect=restore)):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                run(self.manager.clone_model("model", [0, 1]))

        self.assertEqual(old.stop_calls, 1)
        self.assertEqual(restored, [old])
        self.assertIs(self.manager.active["model"], old)
        self.assertEqual(self.manager.runtime_override_names(), [])

    def test_declone_active_reloads_registry_baseline(self):
        old = FakeActive("model", (0, 1))
        replacement = FakeActive("model", (0,))
        self.manager.active = {"model": old}
        self.manager._runtime_overrides["model"] = mm.RuntimeModelOverride((0, 1), (1.0, 1.0))

        async def launch(_name):
            self.manager.active["model"] = replacement
            return replacement

        with patch.object(mm, "registry", self.registry), \
                patch.object(self.manager, "_start_model_locked", new=AsyncMock(side_effect=launch)):
            active, metadata = run(self.manager.declone_model("model"))

        self.assertIs(active, replacement)
        self.assertEqual(old.stop_calls, 1)
        self.assertIsNone(metadata["runtime_override"])
        self.assertEqual(metadata["gpus"], [0])

    def test_pin_same_placement_does_not_reload_but_move_does(self):
        old = FakeActive("model", (0,))
        self.manager.active = {"model": old}
        replacement = FakeActive("model", (1,))

        async def launch(_name):
            self.manager.active["model"] = replacement
            return replacement

        with patch.object(mm, "registry", self.registry), \
                patch.object(self.manager, "_start_model_locked", new=AsyncMock(side_effect=launch)) as start:
            run(self.manager.pin_model("model", 0))
            self.assertEqual(old.stop_calls, 0)
            start.assert_not_awaited()
            run(self.manager.pin_model("model", 1))
            self.assertEqual(old.stop_calls, 1)
            start.assert_awaited_once_with("model")
            self.assertIs(self.manager.active["model"], replacement)

    def test_runtime_pin_protects_and_explicit_unpin_exposes_victim(self):
        incumbent = FakeActive("model", (0,))
        self.manager.active = {"model": incumbent}
        with patch.object(mm, "registry", self.registry):
            run(self.manager.pin_model("model", 0))
            with self.assertRaisesRegex(RuntimeError, "Cannot evict pinned"):
                run(self.manager._allocate_gpu("other", {"gpu-ids": [0, 1]}))
            run(self.manager.unpin_model("model"))
            _, victims = run(self.manager._allocate_gpu("other", {"gpu-ids": [0, 1]}))
            self.assertEqual(victims, [("model", incumbent)])

    def test_manual_preset_transition_retains_overrides_and_enforces_mask(self):
        self.manager._runtime_overrides["model"] = mm.RuntimeModelOverride(pin_state=1)
        active, cleared = run(self.manager.prepare_preset_activation("manual", manual_control=True, gpu_mask={0}))
        self.assertEqual(active, [])
        self.assertEqual(cleared, [])
        self.assertEqual(self.manager.runtime_override_names(), ["model"])
        with patch.object(mm, "registry", self.registry):
            with self.assertRaisesRegex(RuntimeError, "excluded by active GPU mask"):
                run(self.manager.pin_model("model", 1))

    def test_new_manager_has_no_runtime_overrides(self):
        self.manager._runtime_overrides["model"] = mm.RuntimeModelOverride(pin_state=0)
        self.assertEqual(ModelManager(gpu_count=2).runtime_override_names(), [])


class PresetRuntimeResetTests(unittest.TestCase):
    def test_first_activation_does_not_take_same_preset_unchanged_path(self):
        with tempfile.TemporaryDirectory() as td:
            presets = PresetManager(state_dir=td)
            presets.upsert("locked", "", ["target"], {"target": 0}, gpus=[0])
            manager = MagicMock()
            manager.preset_lock = None
            async def prepare(name, **_kwargs):
                manager.preset_lock = name
                return [], []
            manager.prepare_preset_activation = AsyncMock(side_effect=prepare)
            manager.stop_model = AsyncMock(return_value=True)
            manager.start_model = AsyncMock()
            registry = MagicMock()
            registry.get.return_value = {"file": "model.gguf"}
            registry.swap_fixed.return_value = {}

            result = run(presets.activate("locked", manager, registry))

            self.assertNotIn("unchanged", result)
            registry.swap_fixed.assert_called_once_with({"target": 0})
            manager.start_model.assert_awaited_once_with("target", _preset_bypass=True)
            self.assertEqual(presets.get_active_name(), "locked")

    def test_locked_activation_clears_overrides_restarts_targets_and_reports(self):
        with tempfile.TemporaryDirectory() as td:
            presets = PresetManager(state_dir=td)
            presets.upsert("locked", "", ["target"], {}, gpus=[0, 1])
            manager = MagicMock()
            manager.prepare_preset_activation = AsyncMock(return_value=(["target", "other"], ["target", "other", "unloaded"]))
            manager.preset_lock = "locked"
            manager.preset_allows_manual_control = False
            manager.gpu_mask = {0, 1}
            manager.list_active.return_value = ["target", "other"]
            manager.stop_model = AsyncMock(return_value=True)
            manager.start_model = AsyncMock()
            registry = MagicMock()
            registry.get.return_value = {"file": "model.gguf"}
            registry.list_fixed.return_value = {}
            registry.swap_fixed.return_value = {}

            result = run(presets.activate("locked", manager, registry))

            self.assertEqual(result["cleared_runtime_overrides"], ["target", "other", "unloaded"])
            self.assertEqual({call.args[0] for call in manager.stop_model.await_args_list}, {"target", "other"})
            self.assertEqual([call.args[0] for call in manager.start_model.await_args_list], ["target"])
            self.assertNotIn("unchanged", result)


if __name__ == "__main__":
    unittest.main()
