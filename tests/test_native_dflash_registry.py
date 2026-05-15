import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import grimoire.registry as registry_mod


class NativeDflashRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models_dir = Path(self.tmp.name)
        self._orig_models_dir = registry_mod.MODELS_DIR
        registry_mod.MODELS_DIR = str(self.models_dir)
        self.registry = registry_mod.ModelRegistry(
            path=str(self.models_dir / "models.json"),
            seed_path=None,
        )
        (self.models_dir / "target.gguf").write_bytes(b"target")

    def tearDown(self):
        registry_mod.MODELS_DIR = self._orig_models_dir

    def _write_gguf(self, path: Path, metadata: dict, tensors: list[str]) -> None:
        def u32(v):
            return struct.pack("<I", v)

        def u64(v):
            return struct.pack("<Q", v)

        def i32(v):
            return struct.pack("<i", v)

        def string(v: str):
            data = v.encode("utf-8")
            return u64(len(data)) + data

        def value(v):
            if isinstance(v, str):
                return u32(8) + string(v)
            if isinstance(v, bool):
                return u32(7) + (b"\x01" if v else b"\x00")
            if isinstance(v, int):
                return u32(5) + i32(v)
            if isinstance(v, list):
                payload = b"".join(i32(int(item)) for item in v)
                return u32(9) + u32(5) + u64(len(v)) + payload
            raise TypeError(f"unsupported metadata value {v!r}")

        body = [u32(0x46554747), u32(3), u64(len(tensors)), u64(len(metadata))]
        for key, raw in metadata.items():
            body.append(string(key))
            body.append(value(raw))

        for name in tensors:
            body.append(string(name))
            body.append(u32(1))
            body.append(u64(1))
            body.append(u32(0))
            body.append(u64(0))

        path.write_bytes(b"".join(body))

    def _add_native(self, draft_name: str):
        self.registry.add(
            "native-canary",
            {
                "file": "target.gguf",
                "draft": draft_name,
                "speculative-type": "dflash",
            },
        )

    def test_native_dflash_entry_accepts_valid_contract_gguf(self):
        draft = self.models_dir / "draft.gguf"
        self._write_gguf(
            draft,
            {
                "general.architecture": "dflash-draft",
                "dflash-draft.dflash.block_size": 16,
                "dflash-draft.dflash.target_layer_ids": [1, 16, 31, 46, 61],
                "dflash-draft.dflash.n_target_features": 25600,
            },
            ["dflash_fc.weight", "dflash_hidden_norm.weight", "output_norm.weight"],
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertTrue(valid, reason)

    def test_native_dflash_entry_rejects_missing_dflash_metadata(self):
        draft = self.models_dir / "draft-missing-meta.gguf"
        self._write_gguf(
            draft,
            {
                "general.architecture": "dflash-draft",
                "dflash-draft.dflash.block_size": 16,
            },
            ["dflash_fc.weight", "dflash_hidden_norm.weight", "output_norm.weight"],
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("missing required metadata", reason)

    def test_native_dflash_entry_rejects_wrong_architecture(self):
        draft = self.models_dir / "draft-wrong-arch.gguf"
        self._write_gguf(
            draft,
            {
                "general.architecture": "qwen36",
                "qwen36.dflash.block_size": 16,
                "qwen36.dflash.target_layer_ids": [1, 2],
                "qwen36.dflash.n_target_features": 1024,
            },
            ["dflash_fc.weight", "dflash_hidden_norm.weight", "output_norm.weight"],
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("unexpected architecture", reason)

    def test_native_dflash_entry_rejects_missing_required_tensors(self):
        draft = self.models_dir / "draft-missing-tensor.gguf"
        self._write_gguf(
            draft,
            {
                "general.architecture": "dflash-draft",
                "dflash-draft.dflash.block_size": 16,
                "dflash-draft.dflash.target_layer_ids": [1, 16, 31, 46, 61],
                "dflash-draft.dflash.n_target_features": 25600,
            },
            ["dflash_fc.weight", "output_norm.weight"],
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("missing required tensor", reason)

    def test_registry_source_checks_contract_metadata_not_just_extension(self):
        registry_src = (ROOT / "src" / "grimoire" / "registry.py").read_text()
        self.assertIn("_validate_native_dflash_draft_gguf", registry_src)
        self.assertIn("general.architecture", registry_src)
        self.assertIn("dflash.target_layer_ids", registry_src)
        self.assertIn("dflash.n_target_features", registry_src)
        self.assertIn("dflash.fc.weight", registry_src)
        self.assertIn("output_norm.weight", registry_src)


if __name__ == "__main__":
    unittest.main()
