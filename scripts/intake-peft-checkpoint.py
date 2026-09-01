#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_IN_CONTAINER = "/app/src/grimoire/pflash/deps/llama.cpp/convert_lora_to_gguf.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_sha256(path: Path, *, dry_run: bool) -> str:
    if dry_run:
        return "dry-run:not-computed"
    return _sha256(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _to_container_path(path: Path, host_models_dir: Path, container_models_dir: str) -> str:
    resolved = path.resolve()
    host_root = host_models_dir.resolve()
    if _is_relative_to(resolved, host_root):
        return str(Path(container_models_dir) / resolved.relative_to(host_root))
    if str(path).startswith(container_models_dir + "/"):
        return str(path)
    raise SystemExit(
        f"path is not under host models dir {host_models_dir} and cannot be mapped into the container: {path}"
    )


def _to_registry_model_file(path: Path, host_models_dir: Path) -> str:
    resolved = path.resolve()
    host_root = host_models_dir.resolve()
    if _is_relative_to(resolved, host_root):
        return str(resolved.relative_to(host_root))
    return str(path)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _token_id_summary(adapter_config: Path) -> dict[str, Any]:
    data = _read_json(adapter_config)
    ids = data.get("trainable_token_indices")
    if isinstance(ids, dict):
        flattened: list[int] = []
        for value in ids.values():
            if isinstance(value, list):
                flattened.extend(value)
        ids = flattened
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"adapter config has no usable trainable_token_indices: {adapter_config}")
    bad = [item for item in ids if isinstance(item, bool) or not isinstance(item, int)]
    if bad:
        raise SystemExit(f"adapter config contains non-integer trainable token ids: {bad[:5]!r}")
    unique = sorted(set(ids))
    return {
        "count": len(unique),
        "min": min(unique),
        "max": max(unique),
        "contiguous": len(unique) == (max(unique) - min(unique) + 1),
    }


def _upsert_registry(path: Path, model_alias: str, config: dict[str, Any]) -> None:
    data = _read_json(path) if path.exists() else {}
    if not isinstance(data, dict):
        raise SystemExit(f"registry is not a JSON object: {path}")
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        raise SystemExit(f"registry models field is not an object: {path}")
    existing = models.get(model_alias)
    if existing is not None and not isinstance(existing, dict):
        raise SystemExit(f"registry model entry is not an object: {model_alias}")
    merged = dict(existing or {})
    merged.update(config)
    if existing is None:
        merged["added"] = datetime.now(timezone.utc).isoformat()
    models[model_alias] = merged
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PEFT checkpoint into Grimoire-serving artifacts and a provenance manifest."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="PEFT checkpoint directory")
    parser.add_argument("--base-gguf", required=True, type=Path, help="base model GGUF to copy with tokenizer updates")
    parser.add_argument(
        "--base-hf",
        type=Path,
        help="HF base config/tokenizer directory for llama.cpp's LoRA converter; avoids stale training-box paths",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/yeowool/models/gguf"))
    parser.add_argument("--artifact-stem", help="output stem; defaults to model alias or checkpoint directory name")
    parser.add_argument("--adapter-gguf", type=Path, help="explicit output adapter GGUF path")
    tokenizer_output = parser.add_mutually_exclusive_group()
    tokenizer_output.add_argument("--tokenizer-gguf", type=Path, help="explicit tokenizer-aligned base GGUF output path")
    tokenizer_output.add_argument(
        "--shared-tokenizer-gguf",
        type=Path,
        help="existing verified tokenizer-aligned base GGUF shared by compatible checkpoints",
    )
    parser.add_argument("--manifest", type=Path, help="explicit provenance manifest path")
    parser.add_argument("--model-alias", help="optional Grimoire registry model alias")
    parser.add_argument("--ctx-size", type=int, help="ctx-size to write when --update-registry is used")
    parser.add_argument("--registry-path", type=Path, default=Path("state/models.json"))
    parser.add_argument("--update-registry", action="store_true", help="upsert --model-alias into the local registry")
    parser.add_argument("--docker-compose", nargs="+", default=["docker", "compose"])
    parser.add_argument("--service", default="grimoire")
    parser.add_argument("--host-models-dir", type=Path, default=Path(os.environ.get("MODELS_DIR", "/home/yeowool/models")))
    parser.add_argument("--container-models-dir", default="/models")
    parser.add_argument("--outtype", default="auto", choices=("f32", "f16", "bf16", "q8_0", "auto"))
    parser.add_argument("--reuse-existing", action="store_true", help="reuse existing output artifacts")
    parser.add_argument("--force", action="store_true", help="overwrite output artifacts")
    parser.add_argument("--skip-convert", action="store_true", help="skip LoRA adapter conversion")
    parser.add_argument("--skip-tokenizer", action="store_true", help="skip tokenizer GGUF rewrite")
    parser.add_argument("--dry-run", action="store_true", help="print commands and run non-mutating validations")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_gguf = args.base_gguf.resolve()
    output_dir = args.output_dir.resolve()
    artifact_stem = args.artifact_stem or args.model_alias or checkpoint.name
    adapter_config = checkpoint / "adapter_config.json"
    tokenizer_json = checkpoint / "tokenizer.json"
    adapter_model = checkpoint / "adapter_model.safetensors"
    adapter_gguf = (args.adapter_gguf or output_dir / f"{artifact_stem}-lora-tokenrep.gguf").resolve()
    shared_tokenizer_gguf = args.shared_tokenizer_gguf.resolve() if args.shared_tokenizer_gguf else None
    tokenizer_gguf = (
        shared_tokenizer_gguf
        or (args.tokenizer_gguf or output_dir / f"{base_gguf.stem}-{artifact_stem}-tokenizer.gguf").resolve()
    )
    manifest = (args.manifest or output_dir / f"{artifact_stem}.intake.json").resolve()

    _require_dir(checkpoint, "checkpoint directory")
    _require_file(adapter_config, "adapter config")
    _require_file(tokenizer_json, "checkpoint tokenizer")
    _require_file(adapter_model, "adapter weights")
    _require_file(base_gguf, "base GGUF")
    if shared_tokenizer_gguf:
        _require_file(shared_tokenizer_gguf, "shared tokenizer GGUF")
    if args.base_hf:
        _require_dir(args.base_hf, "base HF directory")
        _require_file(args.base_hf / "config.json", "base HF config")

    token_summary = _token_id_summary(adapter_config)
    base_model_name = _read_json(adapter_config).get("base_model_name_or_path")

    if not output_dir.exists() and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [adapter_gguf]
    if shared_tokenizer_gguf is None:
        outputs.append(tokenizer_gguf)
    for output in outputs:
        if output.exists() and not (args.force or args.reuse_existing or args.dry_run):
            raise SystemExit(f"output exists; pass --reuse-existing or --force: {output}")

    container_checkpoint = _to_container_path(checkpoint, args.host_models_dir, args.container_models_dir)
    container_adapter_gguf = _to_container_path(adapter_gguf, args.host_models_dir, args.container_models_dir)
    converter_cmd = [
        *args.docker_compose,
        "exec",
        "-T",
        args.service,
        "python",
        CONVERTER_IN_CONTAINER,
        "--outfile",
        container_adapter_gguf,
        "--outtype",
        args.outtype,
    ]
    if args.base_hf:
        converter_cmd.extend(["--base", _to_container_path(args.base_hf, args.host_models_dir, args.container_models_dir)])
    converter_cmd.append(container_checkpoint)

    tokenizer_cmd: list[str] | None = None
    if shared_tokenizer_gguf is None:
        tokenizer_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/write-gguf-tokenizer-from-hf.py"),
            "--input",
            str(base_gguf),
            "--output",
            str(tokenizer_gguf),
            "--tokenizer-json",
            str(tokenizer_json),
            "--adapter-config",
            str(adapter_config),
        ]
        if args.force:
            tokenizer_cmd.append("--force")
        if args.dry_run:
            tokenizer_cmd.append("--dry-run")

    print(f"checkpoint: {checkpoint}")
    print(f"base_gguf: {base_gguf}")
    print(f"artifact_stem: {artifact_stem}")
    print(f"trainable_tokens: count={token_summary['count']} min={token_summary['min']} max={token_summary['max']} contiguous={token_summary['contiguous']}")
    if base_model_name and not args.base_hf:
        print(f"adapter base_model_name_or_path: {base_model_name}")
        print("warning: pass --base-hf if that path is not valid inside the Grimoire container")

    if args.skip_convert or (adapter_gguf.exists() and args.reuse_existing):
        print(f"reuse adapter GGUF: {adapter_gguf}")
    else:
        _run(converter_cmd, dry_run=args.dry_run)

    if shared_tokenizer_gguf:
        print(f"reuse shared tokenizer GGUF: {tokenizer_gguf}")
    elif args.skip_tokenizer or (tokenizer_gguf.exists() and args.reuse_existing):
        print(f"reuse tokenizer GGUF: {tokenizer_gguf}")
    else:
        assert tokenizer_cmd is not None
        _run(tokenizer_cmd, dry_run=False)

    registry_config: dict[str, Any] | None = None
    if args.model_alias:
        registry_config = {
            "file": _to_registry_model_file(tokenizer_gguf, args.host_models_dir),
            "extra-args": ["--lora", _to_container_path(adapter_gguf, args.host_models_dir, args.container_models_dir)],
        }
        if args.ctx_size is not None:
            registry_config["ctx-size"] = args.ctx_size
        print("registry_config:")
        print(json.dumps({args.model_alias: registry_config}, indent=2))
        if args.update_registry:
            if args.dry_run:
                print(f"dry-run: would update registry {args.registry_path}")
            else:
                _upsert_registry(args.registry_path, args.model_alias, registry_config)

    manifest_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "base_gguf": str(base_gguf),
        "base_hf": str(args.base_hf.resolve()) if args.base_hf else None,
        "base_model_name_or_path": base_model_name,
        "artifact_stem": artifact_stem,
        "adapter_gguf": str(adapter_gguf),
        "tokenizer_gguf": str(tokenizer_gguf),
        "shared_tokenizer_gguf": shared_tokenizer_gguf is not None,
        "token_summary": token_summary,
        "hashes": {
            "adapter_config": _maybe_sha256(adapter_config, dry_run=args.dry_run),
            "tokenizer_json": _maybe_sha256(tokenizer_json, dry_run=args.dry_run),
            "adapter_model": _maybe_sha256(adapter_model, dry_run=args.dry_run),
            "base_gguf": _maybe_sha256(base_gguf, dry_run=args.dry_run),
        },
        "registry_model": args.model_alias,
        "registry_config": registry_config,
        "commands": {
            "convert_lora": converter_cmd,
            "write_tokenizer": tokenizer_cmd,
        },
    }
    if not args.dry_run:
        manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote manifest: {manifest}")
    else:
        print(f"dry-run: would write manifest {manifest}")


if __name__ == "__main__":
    main()
