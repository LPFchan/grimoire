"""Model registry - CRUD operations for models.json."""

import copy
import json
import logging
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

from grimoire import config

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("GRIMOIRE_MODELS_DIR", "/models")
DEFAULT_REGISTRY_PATH = "/var/lib/grimoire/models.json"
DEFAULT_REGISTRY_SEED_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)
REGISTRY_SEED_PATH = os.environ.get("GRIMOIRE_REGISTRY_SEED_PATH", DEFAULT_REGISTRY_SEED_PATH)

BACKEND_LLAMA = "llama"
BACKEND_DFLASH = "dflash"

_GGUF_MAGIC = 0x46554747
_GGUF_SUPPORTED_VERSIONS = {2, 3}
_GGUF_FIXED_VALUE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}
_NATIVE_DFLASH_ALLOWED_ARCHES = {"dflash-draft", "qwen35-dflash-draft"}
_NATIVE_DFLASH_EXPECTED_BLOCK_SIZE = 16
_NATIVE_DFLASH_EXPECTED_TARGET_LAYERS = 5
_NATIVE_DFLASH_MAX_LAYERS = 1024
_NATIVE_DFLASH_MAX_EMBD = 1 << 17
_NATIVE_DFLASH_MAX_FF = 1 << 19
_NATIVE_DFLASH_MAX_HEADS = 1024
_NATIVE_DFLASH_MAX_HEAD_DIM = 1024


class _GGUFReader:
    def __init__(self, fh):
        self._fh = fh

    def read_exact(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF")
        return data

    def skip_exact(self, n: int) -> None:
        remaining = n
        while remaining > 0:
            chunk = min(remaining, 1 << 20)
            self.read_exact(chunk)
            remaining -= chunk

    def u8(self) -> int:
        return self.read_exact(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read_exact(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read_exact(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read_exact(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read_exact(8))[0]

    def bool(self) -> bool:
        return self.read_exact(1)[0] != 0

    def string(self) -> str:
        length = self.u64()
        return self.read_exact(length).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        self.skip_exact(self.u64())


def _read_gguf_value(reader: _GGUFReader, value_type: int):
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        elem_type = reader.u32()
        count = reader.u64()
        return [_read_gguf_value(reader, elem_type) for _ in range(count)]
    if value_type == 0:
        return reader.u8()
    if value_type == 1:
        return reader.i8()
    if value_type == 2:
        return reader.u16()
    if value_type == 3:
        return reader.i16()
    if value_type == 4:
        return reader.u32()
    if value_type == 5:
        return reader.i32()
    if value_type == 6:
        return reader.f32()
    if value_type == 7:
        return reader.bool()
    if value_type == 10:
        return reader.u64()
    if value_type == 11:
        return reader.i64()
    if value_type == 12:
        return reader.f64()
    raise ValueError(f"unsupported GGUF value type {value_type}")


def _skip_gguf_value(reader: _GGUFReader, value_type: int) -> None:
    if value_type == 8:
        reader.skip_string()
        return
    if value_type == 9:
        elem_type = reader.u32()
        count = reader.u64()
        elem_size = _GGUF_FIXED_VALUE_SIZES.get(elem_type)
        if elem_size is not None:
            reader.skip_exact(elem_size * count)
            return
        for _ in range(count):
            _skip_gguf_value(reader, elem_type)
        return
    size = _GGUF_FIXED_VALUE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    reader.skip_exact(size)


def _native_dflash_metadata_keys() -> set[str]:
    suffixes = [
        "embedding_length",
        "block_count",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "dflash.block_size",
        "dflash.n_target_layers",
        "dflash.target_layer_ids",
        "dflash.n_target_features",
    ]
    keys = {"general.architecture"}
    for arch in _NATIVE_DFLASH_ALLOWED_ARCHES:
        keys.update(f"{arch}.{suffix}" for suffix in suffixes)
    return keys


def _read_gguf_metadata(reader: _GGUFReader, wanted_keys: Optional[set[str]] = None) -> tuple[int, dict[str, object]]:
    """Read selected GGUF metadata keys without slurping the whole file."""
    magic = reader.u32()
    if magic != _GGUF_MAGIC:
        raise ValueError("invalid GGUF magic")
    version = reader.u32()
    if version not in _GGUF_SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported GGUF version {version}")

    tensor_count = reader.u64()
    kv_count = reader.u64()

    metadata = {}
    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.u32()
        if wanted_keys is None or key in wanted_keys:
            metadata[key] = _read_gguf_value(reader, value_type)
        else:
            _skip_gguf_value(reader, value_type)

    return tensor_count, metadata


def _scan_gguf_tensor_names(reader: _GGUFReader, tensor_count: int, wanted_names: Optional[set[str]] = None) -> set[str]:
    names = set()
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        for _ in range(n_dims):
            reader.u64()
        reader.u32()
        reader.u64()
        if wanted_names is None or name in wanted_names:
            names.add(name)
    return names


def _validate_native_dflash_draft_gguf(path: str) -> Optional[str]:
    """Return an error string if a native DFlash draft GGUF lacks required contract data."""
    try:
        with open(path, "rb") as f:
            reader = _GGUFReader(f)
            tensor_count, metadata = _read_gguf_metadata(reader, _native_dflash_metadata_keys())

            arch = str(metadata.get("general.architecture") or "").lower()
            if arch not in _NATIVE_DFLASH_ALLOWED_ARCHES:
                return (
                    "Native DFlash draft GGUF has unexpected architecture "
                    f"'{metadata.get('general.architecture')}' (expected dflash-draft or qwen35-dflash-draft)"
                )

            required_keys = [
                f"{arch}.embedding_length",
                f"{arch}.block_count",
                f"{arch}.feed_forward_length",
                f"{arch}.attention.head_count",
                f"{arch}.attention.head_count_kv",
                f"{arch}.attention.key_length",
                f"{arch}.dflash.block_size",
                f"{arch}.dflash.target_layer_ids",
                f"{arch}.dflash.n_target_features",
            ]
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                return f"Native DFlash draft GGUF missing required metadata: {', '.join(missing_keys)}"

            n_embd = metadata.get(f"{arch}.embedding_length")
            n_layer = metadata.get(f"{arch}.block_count")
            n_ff = metadata.get(f"{arch}.feed_forward_length")
            n_head = metadata.get(f"{arch}.attention.head_count")
            n_head_kv = metadata.get(f"{arch}.attention.head_count_kv")
            head_dim = metadata.get(f"{arch}.attention.key_length")
            block_size = metadata.get(f"{arch}.dflash.block_size")
            target_layer_ids = metadata.get(f"{arch}.dflash.target_layer_ids")
            n_target_features = metadata.get(f"{arch}.dflash.n_target_features")
            n_target_layers = metadata.get(f"{arch}.dflash.n_target_layers")

            scalar_keys = {
                f"{arch}.embedding_length": n_embd,
                f"{arch}.block_count": n_layer,
                f"{arch}.feed_forward_length": n_ff,
                f"{arch}.attention.head_count": n_head,
                f"{arch}.attention.head_count_kv": n_head_kv,
                f"{arch}.attention.key_length": head_dim,
                f"{arch}.dflash.block_size": block_size,
                f"{arch}.dflash.n_target_features": n_target_features,
            }
            bad_scalar_keys = [key for key, value in scalar_keys.items() if not isinstance(value, int) or value <= 0]
            if bad_scalar_keys:
                return f"Native DFlash draft GGUF has invalid metadata: {', '.join(bad_scalar_keys)}"

            if n_layer > _NATIVE_DFLASH_MAX_LAYERS or n_embd > _NATIVE_DFLASH_MAX_EMBD or n_ff > _NATIVE_DFLASH_MAX_FF:
                return (
                    "Native DFlash draft GGUF has out-of-range hparams: "
                    f"n_embd={n_embd} n_layer={n_layer} n_ff={n_ff} "
                    f"n_head={n_head} n_head_kv={n_head_kv} head_dim={head_dim}"
                )
            if n_head > _NATIVE_DFLASH_MAX_HEADS or n_head_kv > _NATIVE_DFLASH_MAX_HEADS or head_dim > _NATIVE_DFLASH_MAX_HEAD_DIM:
                return (
                    "Native DFlash draft GGUF has out-of-range hparams: "
                    f"n_embd={n_embd} n_layer={n_layer} n_ff={n_ff} "
                    f"n_head={n_head} n_head_kv={n_head_kv} head_dim={head_dim}"
                )
            if n_head_kv > n_head or (n_head % n_head_kv) != 0:
                return (
                    "Native DFlash draft GGUF has inconsistent attention metadata: "
                    f"n_head={n_head} n_head_kv={n_head_kv}"
                )

            if not isinstance(target_layer_ids, list) or not target_layer_ids:
                return "Native DFlash draft GGUF has invalid or empty dflash.target_layer_ids metadata"
            if any(not isinstance(layer_id, int) or layer_id < 0 for layer_id in target_layer_ids):
                return "Native DFlash draft GGUF has invalid dflash.target_layer_ids metadata"

            if n_target_layers is None:
                n_target_layers = len(target_layer_ids)
            elif not isinstance(n_target_layers, int) or n_target_layers <= 0:
                return "Native DFlash draft GGUF has invalid dflash.n_target_layers metadata"
            elif n_target_layers != len(target_layer_ids):
                return (
                    "Native DFlash draft GGUF has inconsistent dflash metadata: "
                    f"n_target_layers={n_target_layers} but target_layer_ids has {len(target_layer_ids)} entries"
                )

            if block_size != _NATIVE_DFLASH_EXPECTED_BLOCK_SIZE or n_target_layers != _NATIVE_DFLASH_EXPECTED_TARGET_LAYERS:
                return (
                    "Native DFlash draft GGUF has incompatible compiled contract: "
                    f"dflash.block_size={block_size} (expected {_NATIVE_DFLASH_EXPECTED_BLOCK_SIZE}), "
                    f"dflash.n_target_layers={n_target_layers} (expected {_NATIVE_DFLASH_EXPECTED_TARGET_LAYERS})"
                )

            if n_target_features != n_target_layers * n_embd:
                return (
                    "Native DFlash draft GGUF has inconsistent dflash.n_target_features metadata: "
                    f"{n_target_features} != {n_target_layers} * {n_embd}"
                )

            wanted_tensors = {
                "dflash.fc.weight",
                "dflash_fc.weight",
                "dflash.hidden_norm.weight",
                "dflash_hidden_norm.weight",
                "output_norm.weight",
            }
            for layer_idx in range(n_layer):
                wanted_tensors.update(
                    {
                        f"blk.{layer_idx}.attn_norm.weight",
                        f"blk.{layer_idx}.ffn_norm.weight",
                        f"blk.{layer_idx}.post_attention_norm.weight",
                        f"blk.{layer_idx}.attn_q.weight",
                        f"blk.{layer_idx}.attn_k.weight",
                        f"blk.{layer_idx}.attn_v.weight",
                        f"blk.{layer_idx}.attn_output.weight",
                        f"blk.{layer_idx}.attn_q_norm.weight",
                        f"blk.{layer_idx}.attn_k_norm.weight",
                        f"blk.{layer_idx}.ffn_gate.weight",
                        f"blk.{layer_idx}.ffn_up.weight",
                        f"blk.{layer_idx}.ffn_down.weight",
                    }
                )

            tensor_names = _scan_gguf_tensor_names(reader, tensor_count, wanted_tensors)
    except Exception as e:
        return f"Native DFlash draft GGUF could not be read: {e}"

    if not (
        "dflash.fc.weight" in tensor_names or "dflash_fc.weight" in tensor_names
    ):
        return "Native DFlash draft GGUF missing required tensor dflash.fc.weight"
    if not (
        "dflash.hidden_norm.weight" in tensor_names or "dflash_hidden_norm.weight" in tensor_names
    ):
        return "Native DFlash draft GGUF missing required tensor dflash.hidden_norm.weight"
    if "output_norm.weight" not in tensor_names:
        return "Native DFlash draft GGUF missing required tensor output_norm.weight"

    for layer_idx in range(n_layer):
        missing = []
        if f"blk.{layer_idx}.attn_norm.weight" not in tensor_names:
            missing.append(f"blk.{layer_idx}.attn_norm.weight")
        if not (
            f"blk.{layer_idx}.ffn_norm.weight" in tensor_names
            or f"blk.{layer_idx}.post_attention_norm.weight" in tensor_names
        ):
            missing.append(f"blk.{layer_idx}.ffn_norm.weight")
        for suffix in (
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ):
            name = f"blk.{layer_idx}.{suffix}"
            if name not in tensor_names:
                missing.append(name)
        if missing:
            return (
                "Native DFlash draft GGUF missing required layer tensors: "
                + ", ".join(missing)
            )

    return None


def _get_backend(cfg: dict) -> str:
    """Get the backend type for a model config. Defaults to llama."""
    return cfg.get("backend", BACKEND_LLAMA)


def resolve_path(cfg: dict, key: str) -> Optional[str]:
    """Resolve a model config path (file, draft, drafter, mmproj, tokenizer).

    Absolute paths are returned as-is; relative paths are anchored at MODELS_DIR.
    """
    path = cfg.get(key)
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(MODELS_DIR, path)


def _looks_like_local_path(spec: str) -> bool:
    """True if *spec* should be treated as a filesystem path rather than an HF id.

    An ``hf:`` prefix explicitly marks the spec as a Hugging Face repo id.
    Without the prefix, any spec containing a path separator (``/``) or
    starting with ``./``, ``../``, or ``/`` is treated as a local path.
    Bare names without separators (e.g. ``gpt2``) are treated as HF ids.
    """
    if not isinstance(spec, str) or not spec:
        return False
    # Explicit ``hf:`` prefix overrides everything — definitely an HF repo id.
    if spec.startswith("hf:"):
        return False
    if os.path.isabs(spec):
        return True
    if spec.startswith("./") or spec.startswith("../"):
        return True
    # Any spec with a path separator is treated as a local path.
    # If a user wants to reference an HF repo id that happens to contain a
    # ``/`` (e.g. ``Qwen/Qwen3.6-27B``), they must use the ``hf:`` prefix.
    if os.sep in spec:
        return True
    # Bare name (no separators) — assume it's an HF repo id.
    return False

def _strip_hf_prefix(spec: str) -> str:
    """Strip the ``hf:`` prefix from *spec* if present."""
    if isinstance(spec, str) and spec.startswith("hf:"):
        return spec[3:]
    return spec


class ModelRegistry:
    """Model registry backed by JSON file.

    Schema:
    {
      "models": { "alias": { "file": "...", "ctx-size": 262144, ... } },
      "fixed": { "alias": 0 }
    }
    """

    def __init__(self, path=None, seed_path=None):
        self.path = path or REGISTRY_PATH
        self.seed_path = REGISTRY_SEED_PATH if seed_path is None else seed_path
        self._lock = RLock()
        self._stamp = None
        self._data = {"models": {}, "fixed": {}}
        self.reload()

    @staticmethod
    def _normalize(data):
        if not isinstance(data, dict):
            data = {}
        models = data.get("models", {})
        fixed = data.get("fixed", {})
        family_defaults = data.get("family_defaults", {})
        if not isinstance(models, dict):
            models = {}
        if not isinstance(fixed, dict):
            fixed = {}
        if not isinstance(family_defaults, dict):
            family_defaults = {}
        return {**data, "models": models, "fixed": fixed, "family_defaults": family_defaults}

    def _load(self):
        path = self.path
        if not os.path.exists(path) and self.seed_path and os.path.exists(self.seed_path):
            path = self.seed_path
        try:
            with open(path) as f:
                return self._normalize(json.load(f))
        except FileNotFoundError:
            return {"models": {}, "fixed": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse registry: {e}")
            return {"models": {}, "fixed": {}}

    def _stat_stamp(self):
        try:
            return os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            return None

    def _maybe_reload(self):
        stamp = self._stat_stamp()
        if stamp != self._stamp:
            self._data = self._load()
            self._stamp = stamp

    def reload(self):
        """Reload the registry from disk and return a snapshot."""
        with self._lock:
            self._data = self._load()
            self._stamp = self._stat_stamp()
            return copy.deepcopy(self._data)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.path)
        self._stamp = self._stat_stamp()
        logger.info(f"Registry saved to {self.path}")

    def snapshot(self):
        """Return a deep copy of the current registry data."""
        with self._lock:
            self._maybe_reload()
            return copy.deepcopy(self._data)

    def get(self, model_name):
        with self._lock:
            self._maybe_reload()
            cfg = self._data.get("models", {}).get(model_name)
            return copy.deepcopy(cfg) if cfg is not None else None

    def list_all(self):
        with self._lock:
            self._maybe_reload()
            return list(self._data.get("models", {}).keys())

    def list_fixed(self):
        with self._lock:
            self._maybe_reload()
            return dict(self._data.get("fixed", {}))

    def get_family_defaults(self, family):
        with self._lock:
            self._maybe_reload()
            family_defaults = self._data.get("family_defaults", {})
            if not isinstance(family_defaults, dict):
                return {}
            return copy.deepcopy(family_defaults.get(family, {}) or {})

    @staticmethod
    def normalize_model_id(model_id):
        """Normalize gateway aliases, backend IDs, paths, and GGUF names for matching."""
        if not model_id:
            return ""
        value = str(model_id).strip()
        value = Path(value).name
        value = re.sub(r"\.gguf$", "", value, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def resolve(self, model_id):
        """Resolve an external model ID to a registry alias using fuzzy core rules."""
        if not model_id:
            return None
        with self._lock:
            self._maybe_reload()
            models = self._data.get("models", {})

            if model_id in models:
                return model_id

            normalized = self.normalize_model_id(model_id)
            if not normalized:
                return None

            for name, cfg in models.items():
                candidates = [
                    name,
                    cfg.get("alias"),
                    cfg.get("file"),
                    Path(str(cfg.get("file", ""))).name,
                ]
                candidates.extend(cfg.get("aliases", []) or [])
                for candidate in candidates:
                    if candidate and self.normalize_model_id(candidate) == normalized:
                        return name

            matches = []
            for name, cfg in models.items():
                candidates = [name, cfg.get("file"), Path(str(cfg.get("file", ""))).name]
                candidates.extend(cfg.get("aliases", []) or [])
                candidate_norms = [self.normalize_model_id(c) for c in candidates if c]
                if any(normalized in c or c in normalized for c in candidate_norms):
                    matches.append(name)

            if len(matches) == 1:
                return matches[0]
            return None

    def model_metadata(self, model_name):
        """Return public metadata for one registry model."""
        cfg = self.get(model_name)
        if not cfg:
            return None
        return {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "grimoire",
            "context": cfg.get("ctx-size"),
            "output": cfg.get("predict"),
            "family": cfg.get("family"),
            "capabilities": cfg.get("capabilities", ["completion"]),
            "cost": cfg.get("cost", {"input": 0, "output": 0}),
            "backend": _get_backend(cfg),
            "pinned_gpu": self.get_fixed_gpu(model_name),
        }

    def list_metadata(self):
        """Return public metadata for all registry models."""
        return [self.model_metadata(name) for name in self.list_all()]

    def is_fixed(self, model_name):
        """Check if a model is pinned to a GPU."""
        with self._lock:
            self._maybe_reload()
            return model_name in self._data.get("fixed", {})

    def get_fixed_gpu(self, model_name):
        """Get the pinned GPU ID for a model, or None."""
        with self._lock:
            self._maybe_reload()
            return self._data.get("fixed", {}).get(model_name)

    def add(self, model_name, config):
        with self._lock:
            self._maybe_reload()
            if model_name in self._data.get("models", {}):
                raise ValueError(f"Model '{model_name}' already exists")
            self._data.setdefault("models", {})[model_name] = {
                **config,
                "added": datetime.now(timezone.utc).isoformat()
            }
            self._save()
            return copy.deepcopy(self._data["models"][model_name])

    def update(self, model_name, updates):
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            self._data["models"][model_name].update(updates)
            self._save()
            return copy.deepcopy(self._data["models"][model_name])

    def remove(self, model_name):
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            del self._data["models"][model_name]
            self._data.get("fixed", {}).pop(model_name, None)
            self._save()

    def pin_gpu(self, model_name, gpu_id):
        """Pin a model to a specific GPU."""
        if not isinstance(gpu_id, int) or gpu_id < 0:
            raise ValueError("GPU ID must be a non-negative integer")
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            self._data.setdefault("fixed", {})[model_name] = gpu_id
            self._save()

    def unpin_gpu(self, model_name):
        """Remove GPU pinning for a model."""
        with self._lock:
            self._maybe_reload()
            if model_name not in self._data.get("models", {}):
                raise KeyError(f"Model '{model_name}' not found")
            self._data.get("fixed", {}).pop(model_name, None)
            self._save()

    def validate(self, model_name, gpu_count=None):
        """Check if a model config is valid."""
        cfg = self.get(model_name)
        if not cfg:
            return False, f"Model '{model_name}' not found"

        backend = _get_backend(cfg)
        if backend == BACKEND_LLAMA:
            if not cfg.get("file"):
                return False, "Missing 'file' field"
            model_path = os.path.join(MODELS_DIR, cfg["file"])
            if not os.path.exists(model_path):
                return False, f"Model file not found at {model_path}"
            if cfg.get("pflash"):
                drafter = resolve_path(cfg, "drafter")
                if not drafter or not os.path.exists(drafter):
                    return False, f"PFlash drafter not found at {drafter}"
            if cfg.get("park-unpark") and not os.path.exists(config.PFLASH_SHIM_PATH):
                return False, f"park-unpark shim not found at {config.PFLASH_SHIM_PATH}"
            if cfg.get("speculative-type") == "dflash":
                draft = resolve_path(cfg, "spec-draft-model") or resolve_path(cfg, "draft")
                if not draft or not os.path.exists(draft):
                    return False, f"Native DFlash draft model not found at {draft}"
                if not str(draft).lower().endswith(".gguf"):
                    return False, f"Native DFlash draft model must be GGUF: {draft}"
                draft_error = _validate_native_dflash_draft_gguf(draft)
                if draft_error:
                    return False, draft_error
        elif backend == BACKEND_DFLASH:
            target = resolve_path(cfg, "target")
            if not target or not os.path.exists(target):
                return False, f"Target model not found at {target}"
            if cfg.get("snapshot-mode") != "compact-full":
                return False, "DFlash models must set 'snapshot-mode' to 'compact-full'"
            staging_slot = cfg.get("snapshot-staging-slot")
            if staging_slot is None:
                return False, "DFlash models must set 'snapshot-staging-slot'"
            try:
                staging_slot = int(staging_slot)
            except (TypeError, ValueError):
                return False, f"Invalid snapshot staging slot: {staging_slot}"
            if not 0 <= staging_slot < 8:
                return False, f"Snapshot staging slot must be in range 0-7: {staging_slot}"
            session_cap = cfg.get("session-kv-slots")
            if session_cap is None:
                return False, "DFlash models must set 'session-kv-slots'"
            try:
                session_cap = int(session_cap)
            except (TypeError, ValueError):
                return False, f"Invalid session-kv-slots value: {session_cap}"
            if session_cap < 0:
                return False, f"session-kv-slots must be non-negative: {session_cap}"
            use_dflash = cfg.get("dflash", True)
            if use_dflash:
                draft = resolve_path(cfg, "draft")
                if not draft or not os.path.exists(draft):
                    return False, f"Draft model not found at {draft}"
            use_pflash = cfg.get("pflash", True)
            if use_pflash:
                drafter = resolve_path(cfg, "drafter")
                if drafter and not os.path.exists(drafter):
                    return False, f"Drafter model not found at {drafter}"
            tokenizer = cfg.get("tokenizer")
            if not tokenizer:
                return False, (
                    "Missing 'tokenizer' field: dflash backends need an explicit "
                    "tokenizer (HF repo id or local path); runtime download is not safe"
                )
            if _looks_like_local_path(tokenizer):
                tok_path = resolve_path(cfg, "tokenizer")
                if tok_path and not os.path.exists(tok_path):
                    return False, f"Tokenizer path not found at {tok_path}"
        else:
            return False, f"Unknown backend '{backend}'"

        fixed_gpu = self.get_fixed_gpu(model_name)
        if fixed_gpu is not None:
            if not isinstance(fixed_gpu, int) or fixed_gpu < 0:
                return False, f"Invalid pinned GPU ID for '{model_name}': {fixed_gpu}"
            if gpu_count is not None and fixed_gpu >= gpu_count:
                return False, f"Pinned GPU {fixed_gpu} is outside available range 0-{gpu_count - 1}"

        return True, "OK"


registry = ModelRegistry()
