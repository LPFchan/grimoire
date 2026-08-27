"""OpenAI-compatible capability metadata derived from registry aliases."""

import json
import tempfile
import unittest
from pathlib import Path

from grimoire.registry import ModelRegistry


class ModelMetadataCapabilityTests(unittest.TestCase):
    def make_registry(self, models, family_defaults=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "models.json"
        path.write_text(
            json.dumps(
                {
                    "models": models,
                    "fixed": {},
                    "family_defaults": family_defaults or {},
                }
            )
        )
        return ModelRegistry(path=str(path), seed_path="")

    def test_input_modalities_preserve_capabilities_for_text_and_multimodal_models(self):
        registry = self.make_registry(
            {
                "text-model": {"capabilities": ["completion"]},
                "vision-model": {"capabilities": ["completion", "multimodal"]},
                "vision-alias": {"capabilities": ["vision"]},
            }
        )

        text = registry.model_metadata("text-model")
        vision = registry.model_metadata("vision-model")
        alias = registry.model_metadata("vision-alias")
        self.assertEqual(text["capabilities"], ["completion"])
        self.assertEqual(text["input_modalities"], ["text"])
        self.assertEqual(vision["capabilities"], ["completion", "multimodal"])
        self.assertEqual(vision["input_modalities"], ["text", "image"])
        self.assertEqual(alias["input_modalities"], ["text", "image"])

    def test_qwen_effort_alias_is_fixed_to_native_level(self):
        registry = self.make_registry(
            {
                "qwen-low": {
                    "family": "qwen",
                    "capabilities": ["completion"],
                    "extra-args": [
                        "--chat-template-kwargs",
                        '{"reasoning_effort":"low"}',
                    ],
                }
            },
            {"qwen": {"extra-args": ["--chat-template-kwargs", '{"preserve_thinking":true}']}},
        )

        self.assertEqual(
            registry.model_metadata("qwen-low")["reasoning"],
            {
                "supported_efforts": ["low"],
                "default_effort": "low",
                "default_enabled": True,
                "mandatory": True,
            },
        )

    def test_muse_strength_alias_is_advertised_without_mapping_label(self):
        registry = self.make_registry(
            {
                "muse-high": {
                    "family": "muse",
                    "extra-args": [
                        "--chat-template-kwargs",
                        '{"reasoning_strength":"high"}',
                    ],
                }
            }
        )
        self.assertEqual(
            registry.model_metadata("muse-high")["reasoning"],
            {
                "supported_efforts": ["high"],
                "default_effort": "high",
                "default_enabled": True,
                "mandatory": True,
            },
        )

    def test_base_alias_explicitly_advertises_reasoning_unsupported(self):
        registry = self.make_registry({"qwen-base": {"family": "qwen"}})
        self.assertEqual(
            registry.model_metadata("qwen-base")["reasoning"],
            {"supported": False, "supported_efforts": []},
        )

    def test_malformed_or_conflicting_reasoning_configuration_is_unknown(self):
        registry = self.make_registry(
            {
                "bad-json": {
                    "extra-args": ["--chat-template-kwargs", '{"reasoning_effort":'],
                },
                "duplicate-key": {
                    "extra-args": [
                        "--chat-template-kwargs",
                        '{"reasoning_effort":"low","reasoning_effort":"high"}',
                    ],
                },
                "both-keys": {
                    "chat-template-kwargs": {
                        "reasoning_effort": "low",
                        "reasoning_strength": "low",
                    }
                },
                "empty": {
                    "chat-template-kwargs": {"reasoning_effort": ""},
                },
            }
        )
        for name in ("bad-json", "duplicate-key", "both-keys", "empty"):
            self.assertEqual(registry.model_metadata(name)["reasoning"], {}, name)


if __name__ == "__main__":
    unittest.main()
