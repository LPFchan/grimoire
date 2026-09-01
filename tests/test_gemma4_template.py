import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "gemma-4-31B-it-chat_template.jinja"
OFFICIAL_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"


class Gemma4TemplateTests(unittest.TestCase):
    def test_registry_uses_pinned_canonical_template_for_gemma4_family(self):
        registry = json.loads((ROOT / "etc" / "models.json").read_text())

        self.assertEqual(
            registry["family_defaults"]["gemma4"]["chat-template-file"],
            "/templates/gemma-4-31B-it-chat_template.jinja",
        )
        self.assertTrue(
            all(
                "chat-template-file" not in config
                for config in registry["models"].values()
                if config.get("family") == "gemma4"
            )
        )

    def test_template_matches_google_july_2026_release(self):
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), OFFICIAL_SHA256)
        self.assertEqual(
            TEMPLATE.read_bytes(),
            (ROOT / "templates" / "chat_template.jinja").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
