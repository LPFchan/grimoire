import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Qwen38RegistryTests(unittest.TestCase):
    def test_reasoning_aliases_use_gpu_vision_190k_and_ubatch_128(self):
        models = json.loads((ROOT / "etc" / "models.json").read_text())["models"]

        self.assertNotIn("qwen3.8-27B", models)
        for effort in ("low", "medium", "xhigh"):
            config = models[f"qwen3.8-27B-{effort}"]
            self.assertEqual(config["ctx-size"], 190000)
            self.assertEqual(config["mmproj"], "gguf/mmproj-BF16.gguf")
            self.assertEqual(config["speculative-type"], "mtp")
            self.assertEqual(config["mtp-head"], "gguf/mtp-Qwen3.8-27B-Q4_0.gguf")
            self.assertNotIn("--no-mmproj-offload", config["extra-args"])
            self.assertEqual(
                config["extra-args"],
                [
                    "--ubatch-size",
                    "128",
                    "--chat-template-kwargs",
                    json.dumps({"reasoning_effort": effort}, separators=(",", ":")),
                ],
            )


if __name__ == "__main__":
    unittest.main()
