import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "intake-peft-checkpoint.py"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    models = tmp_path / "models"
    checkpoint = models / "checkpoint-1"
    output_dir = models / "gguf"
    checkpoint.mkdir(parents=True)
    output_dir.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/stale/base", "trainable_token_indices": [10, 11]}),
        encoding="utf-8",
    )
    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    base_gguf = output_dir / "base.gguf"
    base_gguf.write_bytes(b"base")
    shared_gguf = output_dir / "shared-tokenizer.gguf"
    shared_gguf.write_bytes(b"shared")
    return models, checkpoint, base_gguf, shared_gguf


def _command(models: Path, checkpoint: Path, base_gguf: Path, shared_gguf: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--base-gguf",
        str(base_gguf),
        "--output-dir",
        str(models / "gguf"),
        "--host-models-dir",
        str(models),
        "--shared-tokenizer-gguf",
        str(shared_gguf),
        "--model-alias",
        "test-alias",
        "--skip-convert",
        "--dry-run",
    ]


def test_shared_tokenizer_is_reused_without_broad_reuse_flag(tmp_path: Path) -> None:
    models, checkpoint, base_gguf, shared_gguf = _fixture(tmp_path)

    result = subprocess.run(
        _command(models, checkpoint, base_gguf, shared_gguf),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"reuse shared tokenizer GGUF: {shared_gguf}" in result.stdout
    assert '"file": "gguf/shared-tokenizer.gguf"' in result.stdout
    assert not (models / "gguf" / "base-test-alias-tokenizer.gguf").exists()


def test_shared_tokenizer_must_exist(tmp_path: Path) -> None:
    models, checkpoint, base_gguf, shared_gguf = _fixture(tmp_path)
    shared_gguf.unlink()

    result = subprocess.run(
        _command(models, checkpoint, base_gguf, shared_gguf),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "shared tokenizer GGUF not found" in result.stderr


def test_shared_and_output_tokenizer_paths_are_mutually_exclusive(tmp_path: Path) -> None:
    models, checkpoint, base_gguf, shared_gguf = _fixture(tmp_path)
    command = _command(models, checkpoint, base_gguf, shared_gguf)
    command.extend(["--tokenizer-gguf", str(models / "gguf" / "other.gguf")])

    result = subprocess.run(command, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "not allowed with argument --shared-tokenizer-gguf" in result.stderr
