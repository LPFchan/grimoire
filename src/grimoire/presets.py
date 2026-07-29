"""Preset manager — named GPU model configurations with activation, locking, and boot restore."""

import copy
import json
import logging
import os
from typing import Optional

import asyncio

from grimoire.registry import ModelRegistry

logger = logging.getLogger(__name__)


DEFAULT_PRESETS_PATH = "/var/lib/grimoire/presets.json"
PRESETS_PATH = os.environ.get("GRIMOIRE_PRESETS_PATH", DEFAULT_PRESETS_PATH)


def stop_sort_key(model_name, registry):
    cfg = registry.get(model_name) or {}
    return (bool(cfg.get("vram-budget-mib")), model_name)


def start_sort_key(model_name, registry):
    cfg = registry.get(model_name) or {}
    return (not bool(cfg.get("vram-budget-mib")), model_name)


class PresetManager:
    def __init__(self, state_dir=None):
        path = os.path.join(state_dir, "presets.json") if state_dir else PRESETS_PATH
        self._path = path
        self._activation_lock = asyncio.Lock()
        data = self._load()
        self._pre_preset_fixed = (
            data.get("pre_preset_fixed")
            if isinstance(data.get("pre_preset_fixed"), dict)
            else None
        )

    def _load(self):
        try:
            with open(self._path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"presets": {}, "active": None, "pre_preset_fixed": None}

    def _save(self, data):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, self._path)
        logger.info(f"Presets saved to {self._path}")

    def _read_full(self):
        return self._load()

    def list(self):
        data = self._load()
        presets = data.get("presets", {})
        active = data.get("active")
        result = {}
        for name, preset in presets.items():
            result[name] = {
                "description": preset.get("description", ""),
                "model_count": len(preset.get("models", [])),
                "active": name == active,
            }
        return result

    def get(self, name):
        data = self._load()
        return data.get("presets", {}).get(name)

    def upsert(self, name, description, models, fixed, gpus=None):
        data = self._load()
        if "presets" not in data:
            data["presets"] = {}
        preset = {
            "description": description,
            "models": list(models),
            "fixed": dict(fixed) if fixed else {},
        }
        if gpus is not None:
            preset["gpus"] = list(gpus)
        data["presets"][name] = preset
        self._save(data)

    def delete(self, name):
        data = self._load()
        presets = data.get("presets", {})
        if name not in presets:
            return False
        if data.get("active") == name:
            raise RuntimeError(f"Cannot delete active preset '{name}'. Activate another preset first.")
        del presets[name]
        self._save(data)
        return True

    def get_active_name(self):
        data = self._load()
        return data.get("active")

    def get_pre_preset_fixed(self):
        return copy.deepcopy(self._pre_preset_fixed) if self._pre_preset_fixed else None

    def _set_active(self, name):
        data = self._load()
        data["active"] = name
        self._save(data)

    def _set_pre_preset_fixed(self, data):
        full = self._load()
        full["pre_preset_fixed"] = copy.deepcopy(data) if data else None
        self._pre_preset_fixed = copy.deepcopy(data) if data else None
        self._save(full)

    async def activate(self, name, manager, registry):
        async with self._activation_lock:
            preset = self.get(name)
            if preset is None:
                raise KeyError(f"Preset '{name}' not found")

            target = set(preset.get("models", []))
            manual_control = not target and not preset.get("fixed")

            missing = [m for m in target if not registry.get(m)]
            if missing:
                raise ValueError(f"Unknown models in preset: {missing}")

            gpu_list = preset.get("gpus")
            intended_mask = set(gpu_list) if gpu_list is not None else None
            same_preset = manager.preset_lock == name
            current, cleared_runtime_overrides = await manager.prepare_preset_activation(
                name,
                manual_control=manual_control,
                gpu_mask=intended_mask,
            )
            current = set(current)
            runtime_moved = current & target & set(cleared_runtime_overrides)

            if same_preset:
                missing_models = target - current
                preset_fixed = preset.get("fixed", {})
                current_fixed = dict(registry.list_fixed())
                current_subset = {k: current_fixed.get(k) for k in preset_fixed}
                fixed_drifted = current_subset != preset_fixed
                mask_changed = manager.gpu_mask != intended_mask
                manual_control_changed = getattr(manager, "preset_allows_manual_control", False) != manual_control
                if (not missing_models and not fixed_drifted and not mask_changed
                        and not manual_control_changed and not cleared_runtime_overrides):
                    return {"active": name, "unchanged": True}

            old_fixed = registry.swap_fixed(preset.get("fixed", {}))

            if self._pre_preset_fixed is None:
                self._pre_preset_fixed = old_fixed
                self._set_pre_preset_fixed(old_fixed)

            new_fixed = preset.get("fixed", {})
            moved = {
                m
                for m in (current & target)
                if (old_fixed.get(m) != new_fixed.get(m))
                or (m in old_fixed) != (m in new_fixed)
            }
            moved |= runtime_moved

            to_stop = (current - target) | moved
            to_start = target - (current - to_stop)

            stopped = []
            started = []
            failed = []
            warnings = []

            for m in sorted(to_stop, key=lambda m: stop_sort_key(m, registry)):
                try:
                    ok = await manager.stop_model(m, _preset_bypass=True)
                    if ok:
                        stopped.append(m)
                except Exception as exc:
                    msg = f"Failed to stop {m}: {exc}"
                    logger.warning(msg)
                    warnings.append(msg)

            for m in sorted(to_start, key=lambda m: start_sort_key(m, registry)):
                try:
                    await manager.start_model(m, _preset_bypass=True)
                    started.append(m)
                except Exception as exc:
                    msg = f"Failed to start {m}: {exc}"
                    logger.warning(msg)
                    failed.append(m)
                    warnings.append(msg)

            restored_fixed = None
            has_gpu_mask = preset.get("gpus") is not None
            if not target and not preset.get("fixed") and not has_gpu_mask:
                manager.preset_lock = None
                manager.preset_allows_manual_control = False
                manager.gpu_mask = None
                if self._pre_preset_fixed is not None:
                    registry.swap_fixed(self._pre_preset_fixed)
                    restored_fixed = dict(self._pre_preset_fixed)
                    self._pre_preset_fixed = None
                    self._set_pre_preset_fixed(None)

            active_name = None if manager.preset_lock is None else name
            self._set_active(active_name)

            return {
                "stopped": stopped,
                "started": started,
                "failed": failed,
                "warnings": warnings,
                "active": active_name,
                "cleared_runtime_overrides": cleared_runtime_overrides,
                "fixed": restored_fixed if restored_fixed is not None else new_fixed,
                "old_fixed": old_fixed,
            }


presets = PresetManager()
