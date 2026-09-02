"""Retired model configuration must be refused, not quietly ignored.

model_manager only emits `--spec-type` for the modes it implements, so an
unrecognised `speculative-type` does not fail — it loads an ordinary
non-speculative model that looks like it worked. When DFlash and PFlash were
removed their validation went with them, which turned a hard failure into a
silent one. These tests pin the refusal.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grimoire import config
import grimoire.registry as registry_mod


class RetiredConfigRejectedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "m.gguf").write_bytes(b"weights")
        self._orig = registry_mod.MODELS_DIR
        registry_mod.MODELS_DIR = str(self.dir)
        self.addCleanup(lambda: setattr(registry_mod, "MODELS_DIR", self._orig))

    def _validate(self, extra):
        path = str(self.dir / "models.json")
        reg = registry_mod.ModelRegistry(path=path, seed_path=None)
        reg._data = {"models": {"m": {"file": "m.gguf", **extra}}, "fixed": {}}
        return reg.validate("m", gpu_count=2)

    def test_supported_speculative_types_are_accepted(self):
        for spec in sorted(config.SUPPORTED_SPECULATIVE_TYPES):
            extra = {"speculative-type": spec}
            if spec == "mtp":
                (self.dir / "head.gguf").write_bytes(b"head")
                extra["mtp-head"] = "head.gguf"
            ok, msg = self._validate(extra)
            self.assertTrue(ok, f"{spec}: {msg}")

    def test_no_speculation_is_accepted(self):
        ok, msg = self._validate({})
        self.assertTrue(ok, msg)

    def test_retired_speculative_types_are_refused(self):
        for spec in ("dflash", "pflash"):
            ok, msg = self._validate({"speculative-type": spec})
            self.assertFalse(ok, f"{spec} should be refused")
            self.assertIn(spec, msg)

    def test_unknown_speculative_type_is_refused(self):
        """A typo must fail loudly rather than load a plain model."""
        ok, msg = self._validate({"speculative-type": "mtpp"})
        self.assertFalse(ok)
        self.assertIn("mtpp", msg)

    def test_retired_boolean_fields_are_refused(self):
        for field in config.RETIRED_MODEL_FIELDS:
            ok, msg = self._validate({field: True})
            self.assertFalse(ok, f"{field} should be refused")
            self.assertIn(field, msg)

    def test_retired_fields_are_ignored_when_falsy(self):
        """An explicit false is not a request to use the feature."""
        ok, msg = self._validate({field: False for field in config.RETIRED_MODEL_FIELDS})
        self.assertTrue(ok, msg)

    def test_manager_only_implements_the_supported_types(self):
        """The allowlist must not drift from what the launcher can emit."""
        source = (ROOT / "src" / "grimoire" / "model_manager.py").read_text()
        self.assertIn('if spec_type in ("nextn", "mtp"):', source)
        self.assertEqual(config.SUPPORTED_SPECULATIVE_TYPES, {"nextn", "mtp"})


if __name__ == "__main__":
    unittest.main()
