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

    @staticmethod
    def _draft_metadata(**overrides):
        metadata = {
            "general.architecture": "dflash-draft",
            "dflash-draft.embedding_length": 5120,
            "dflash-draft.block_count": 5,
            "dflash-draft.feed_forward_length": 17408,
            "dflash-draft.attention.head_count": 32,
            "dflash-draft.attention.head_count_kv": 8,
            "dflash-draft.attention.key_length": 128,
            "dflash-draft.dflash.block_size": 16,
            "dflash-draft.dflash.n_target_layers": 5,
            "dflash-draft.dflash.target_layer_ids": [1, 16, 31, 46, 61],
            "dflash-draft.dflash.n_target_features": 25600,
        }
        metadata.update(overrides)
        return metadata

    @staticmethod
    def _draft_tensors(layer_count: int = 5):
        tensors = ["dflash_fc.weight", "dflash_hidden_norm.weight", "output_norm.weight"]
        for layer_idx in range(layer_count):
            tensors.extend(
                [
                    f"blk.{layer_idx}.attn_norm.weight",
                    f"blk.{layer_idx}.ffn_norm.weight",
                    f"blk.{layer_idx}.attn_q.weight",
                    f"blk.{layer_idx}.attn_k.weight",
                    f"blk.{layer_idx}.attn_v.weight",
                    f"blk.{layer_idx}.attn_output.weight",
                    f"blk.{layer_idx}.attn_q_norm.weight",
                    f"blk.{layer_idx}.attn_k_norm.weight",
                    f"blk.{layer_idx}.ffn_gate.weight",
                    f"blk.{layer_idx}.ffn_up.weight",
                    f"blk.{layer_idx}.ffn_down.weight",
                ]
            )
        return tensors

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
            self._draft_metadata(),
            self._draft_tensors(),
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
            self._draft_tensors(),
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
            self._draft_tensors(),
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("unexpected architecture", reason)

    def test_native_dflash_entry_rejects_missing_required_tensors(self):
        draft = self.models_dir / "draft-missing-tensor.gguf"
        tensors = self._draft_tensors()
        tensors.remove("blk.3.ffn_up.weight")
        self._write_gguf(
            draft,
            self._draft_metadata(),
            tensors,
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("missing required layer tensors", reason)
        self.assertIn("blk.3.ffn_up.weight", reason)

    def test_native_dflash_entry_rejects_wrong_compiled_contract(self):
        draft = self.models_dir / "draft-wrong-contract.gguf"
        self._write_gguf(
            draft,
            self._draft_metadata(**{"dflash-draft.dflash.block_size": 32}),
            self._draft_tensors(),
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("incompatible compiled contract", reason)

    def test_native_dflash_entry_rejects_inconsistent_target_feature_count(self):
        draft = self.models_dir / "draft-bad-features.gguf"
        self._write_gguf(
            draft,
            self._draft_metadata(**{"dflash-draft.dflash.n_target_features": 12345}),
            self._draft_tensors(),
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("n_target_features", reason)

    def test_native_dflash_entry_rejects_inconsistent_attention_metadata(self):
        draft = self.models_dir / "draft-bad-heads.gguf"
        self._write_gguf(
            draft,
            self._draft_metadata(**{"dflash-draft.attention.head_count": 30, "dflash-draft.attention.head_count_kv": 8}),
            self._draft_tensors(),
        )
        self._add_native(draft.name)
        valid, reason = self.registry.validate("native-canary")
        self.assertFalse(valid)
        self.assertIn("inconsistent attention metadata", reason)

    def test_registry_source_checks_contract_metadata_not_just_extension(self):
        registry_src = (ROOT / "src" / "grimoire" / "registry.py").read_text()
        self.assertIn("_validate_native_dflash_draft_gguf", registry_src)
        self.assertIn("general.architecture", registry_src)
        self.assertIn("embedding_length", registry_src)
        self.assertIn("feed_forward_length", registry_src)
        self.assertIn("attention.head_count", registry_src)
        self.assertIn("dflash.n_target_layers", registry_src)
        self.assertIn("dflash.target_layer_ids", registry_src)
        self.assertIn("dflash.n_target_features", registry_src)
        self.assertIn("_GGUFReader", registry_src)
        self.assertIn("_scan_gguf_tensor_names", registry_src)
        self.assertIn("blk.{layer_idx}.attn_norm.weight", registry_src)
        self.assertIn("output_norm.weight", registry_src)


if __name__ == "__main__":
    unittest.main()
