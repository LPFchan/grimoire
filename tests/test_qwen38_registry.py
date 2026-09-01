import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Qwen38RegistryTests(unittest.TestCase):
    def test_reasoning_aliases_use_cpu_vision_237568_and_ubatch_128(self):
        models = json.loads((ROOT / "etc" / "models.json").read_text())["models"]

        self.assertNotIn("qwen3.8-27B", models)
        for effort in ("low", "medium", "xhigh"):
            config = models[f"qwen3.8-27B-{effort}"]
            self.assertEqual(config["ctx-size"], 237568)
            self.assertEqual(config["mmproj"], "gguf/mmproj-BF16.gguf")
            self.assertEqual(config["speculative-type"], "mtp")
            self.assertEqual(config["mtp-head"], "gguf/mtp-Qwen3.8-27B-Q4_0.gguf")
            self.assertEqual(
                config["extra-args"],
                [
                    "--no-mmproj-offload",
                    "--ubatch-size",
                    "128",
                    "--chat-template-kwargs",
                    json.dumps({"reasoning_effort": effort}, separators=(",", ":")),
                ],
            )

    def test_uncensored_aliases_inherit_reasoning_profiles_with_matching_companions(self):
        models = json.loads((ROOT / "etc" / "models.json").read_text())["models"]

        for effort in ("low", "medium", "xhigh"):
            base = models[f"qwen3.8-27B-{effort}"]
            config = models[f"qwen3.8-27B-uncensored-{effort}"]

            inherited_keys = set(base) - {"file", "mmproj", "mtp-head", "added"}
            for key in inherited_keys:
                self.assertEqual(config[key], base[key])

            self.assertEqual(config["file"], "gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
            self.assertEqual(
                config["mmproj"],
                "gguf/mmproj-Qwen3.8-27B-Uncensored-F16.gguf",
            )
            self.assertEqual(
                config["mtp-head"],
                "gguf/Qwen3.8-27B-Uncensored-draft-Q4_0.gguf",
            )


if __name__ == "__main__":
    unittest.main()
