"""Model-name resolution in the registry.

Callers reach a model by its registered name, by the basename of its GGUF, or
by an alias. Resolution normalizes names before comparing, which is what makes
the last test here matter: normalization must not turn a distinct model into a
substring match for another one.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import grimoire.registry as registry_mod


class RegistryResolveTests(unittest.TestCase):
    def _registry(self, models):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / "models.json")
        with open(path, "w") as f:
            json.dump({"models": models}, f)
        return registry_mod.ModelRegistry(path=path, seed_path=None)

    def test_resolve_exact_normalized_name(self):
        reg = self._registry({"My-Model": {"file": "m.gguf"}})
        self.assertEqual(reg.resolve("my-model"), "My-Model")

    def test_resolve_by_file_basename(self):
        reg = self._registry({"m": {"file": "gguf/My-Model-Q4_K_M.gguf"}})
        self.assertEqual(reg.resolve("My-Model-Q4_K_M.gguf"), "m")

    def test_resolve_by_alias(self):
        reg = self._registry({"m": {"file": "m.gguf", "aliases": ["nickname"]}})
        self.assertEqual(reg.resolve("nickname"), "m")

    def test_resolve_returns_none_for_unknown(self):
        reg = self._registry({"m": {"file": "m.gguf"}})
        self.assertIsNone(reg.resolve("nobody"))

    def test_resolve_no_false_positive_substring(self):
        """A normalized name that merely contains another must not match it.

        `normalize("qwen-3.6-27B")` is `qwen3627b`, which is a substring of
        `normalize("turbo-qwen3.6-27B")`. Asking for the prefixed model when only
        the bare one is registered must fail rather than silently serve the
        wrong weights.
        """
        reg = self._registry({"qwen-3.6-27B": {"file": "gguf/Qwen3.6-27B.gguf"}})
        self.assertIsNone(reg.resolve("turbo-qwen3.6-27B"))
        self.assertIsNone(reg.resolve("turboqwen3627b"))


if __name__ == "__main__":
    unittest.main()
