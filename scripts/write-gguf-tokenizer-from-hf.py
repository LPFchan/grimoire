#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

from tqdm import tqdm

LOGGER = logging.getLogger("write-gguf-tokenizer-from-hf")


class MetadataDetails(NamedTuple):
    type: Any
    value: Any
    sub_type: Any | None = None


def _load_gguf(llama_cpp_dir: Path):
    gguf_py = llama_cpp_dir / "gguf-py"
    if not gguf_py.exists():
        raise SystemExit(f"llama.cpp gguf-py directory not found: {gguf_py}")
    sys.path.insert(0, str(gguf_py))
    import gguf  # type: ignore[import-not-found]

    return gguf


def _field_contents(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None


def _copy_with_metadata(reader: Any, writer: Any, gguf: Any, new_metadata: dict[str, MetadataDetails]) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        replacement = new_metadata.pop(field.name, None)
        if replacement is not None:
            LOGGER.info("replacing metadata field %s", field.name)
            writer.add_key_value(
                field.name,
                replacement.value,
                replacement.type,
                sub_type=replacement.sub_type if replacement.sub_type is not None else sub_type,
            )
        elif value is not None:
            writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    for key, replacement in new_metadata.items():
        LOGGER.info("adding metadata field %s", key)
        writer.add_key_value(key, replacement.value, replacement.type, sub_type=replacement.sub_type)

    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    progress = tqdm(desc="Writing tensors", total=total_bytes, unit="byte", unit_scale=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        progress.update(tensor.n_bytes)
    progress.close()
    writer.close()


def _token_type_for_added_token(gguf: Any, token: dict[str, Any]) -> int:
    return int(gguf.TokenType.CONTROL if token.get("special") else gguf.TokenType.USER_DEFINED)


def _coerce_token_ids(value: Any, source: str) -> set[int]:
    if isinstance(value, dict):
        ids: set[int] = set()
        for nested in value.values():
            ids.update(_coerce_token_ids(nested, source))
        return ids
    if not isinstance(value, list):
        raise SystemExit(f"{source} must contain a list of token ids")

    ids = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SystemExit(f"{source} contains a non-integer token id: {item!r}")
        if item < 0:
            raise SystemExit(f"{source} contains a negative token id: {item}")
        ids.add(item)
    if not ids:
        raise SystemExit(f"{source} did not contain any token ids")
    return ids


def _load_adapter_token_ids(path: Path) -> set[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"adapter config is not an object: {path}")
    if "trainable_token_indices" not in data:
        raise SystemExit(f"adapter config has no trainable_token_indices: {path}")
    return _coerce_token_ids(data["trainable_token_indices"], f"{path}:trainable_token_indices")


def _load_token_ids_json(path: Path) -> set[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("trainable_token_indices", "token_ids", "ids"):
            if key in data:
                return _coerce_token_ids(data[key], f"{path}:{key}")
        raise SystemExit(f"token id json has no trainable_token_indices, token_ids, or ids key: {path}")
    return _coerce_token_ids(data, str(path))


def _is_selected_token_id(token_id: int, exact_ids: set[int] | None, min_id: int, max_id: int | None) -> bool:
    if exact_ids is not None:
        return token_id in exact_ids
    return token_id >= min_id and (max_id is None or token_id <= max_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a GGUF while replacing tokenizer.ggml.tokens from a Hugging Face tokenizer.json."
    )
    parser.add_argument("--input", required=True, type=Path, help="source GGUF")
    parser.add_argument("--output", required=True, type=Path, help="destination GGUF")
    parser.add_argument("--tokenizer-json", required=True, type=Path, help="Hugging Face tokenizer.json")
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("src/grimoire/pflash/deps/llama.cpp"),
        help="llama.cpp checkout containing gguf-py",
    )
    parser.add_argument("--min-id", type=int, default=0, help="lowest token id to rewrite")
    parser.add_argument("--max-id", type=int, default=None, help="highest token id to rewrite, inclusive")
    parser.add_argument(
        "--adapter-config",
        type=Path,
        help="PEFT adapter_config.json; uses exact trainable_token_indices instead of a numeric range",
    )
    parser.add_argument(
        "--token-ids-json",
        type=Path,
        help="JSON list or object containing exact token ids to rewrite",
    )
    parser.add_argument(
        "--allow-missing-token-ids",
        action="store_true",
        help="do not fail when an exact token id is absent from tokenizer.json added_tokens",
    )
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing output")
    parser.add_argument("--force", action="store_true", help="overwrite output if it already exists")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    exact_token_ids: set[int] | None = None
    exact_sources: list[str] = []
    if args.adapter_config:
        exact_token_ids = _load_adapter_token_ids(args.adapter_config)
        exact_sources.append(str(args.adapter_config))
    if args.token_ids_json:
        token_ids = _load_token_ids_json(args.token_ids_json)
        exact_token_ids = token_ids if exact_token_ids is None else exact_token_ids | token_ids
        exact_sources.append(str(args.token_ids_json))
    if exact_token_ids is not None:
        LOGGER.info(
            "using exact token ids from %s: count=%d min=%d max=%d",
            ", ".join(exact_sources),
            len(exact_token_ids),
            min(exact_token_ids),
            max(exact_token_ids),
        )

    gguf = _load_gguf(args.llama_cpp_dir)
    LOGGER.info("loading tokenizer: %s", args.tokenizer_json)
    tokenizer_data = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    added_tokens = tokenizer_data.get("added_tokens") or []
    if not isinstance(added_tokens, list):
        raise SystemExit("tokenizer.json added_tokens is not a list")

    LOGGER.info("loading GGUF metadata: %s", args.input)
    reader = gguf.GGUFReader(args.input, "r")
    arch = _field_contents(reader, gguf.Keys.General.ARCHITECTURE)
    if arch is None:
        raise SystemExit("source GGUF has no general.architecture")

    tokens = list(_field_contents(reader, gguf.Keys.Tokenizer.LIST) or [])
    token_types = list(_field_contents(reader, gguf.Keys.Tokenizer.TOKEN_TYPE) or [])
    if not tokens:
        raise SystemExit("source GGUF has no tokenizer.ggml.tokens")
    if token_types and len(token_types) != len(tokens):
        raise SystemExit("tokenizer.ggml.token_type length does not match tokenizer.ggml.tokens")

    changed_tokens: list[tuple[int, str, str]] = []
    changed_types = 0
    skipped = 0
    seen_exact_token_ids: set[int] = set()
    for token in added_tokens:
        if not isinstance(token, dict) or "id" not in token or "content" not in token:
            skipped += 1
            continue
        token_id = int(token["id"])
        if not _is_selected_token_id(token_id, exact_token_ids, args.min_id, args.max_id):
            skipped += 1
            continue
        seen_exact_token_ids.add(token_id)
        if token_id < 0 or token_id >= len(tokens):
            raise SystemExit(f"token id {token_id} is outside GGUF vocab size {len(tokens)}")

        content = str(token["content"])
        old = tokens[token_id]
        if old != content:
            tokens[token_id] = content
            changed_tokens.append((token_id, old, content))

        if token_types:
            new_type = _token_type_for_added_token(gguf, token)
            if int(token_types[token_id]) != new_type:
                token_types[token_id] = new_type
                changed_types += 1

    if exact_token_ids is not None:
        missing = sorted(exact_token_ids - seen_exact_token_ids)
        if missing and not args.allow_missing_token_ids:
            preview = ", ".join(str(token_id) for token_id in missing[:20])
            if len(missing) > 20:
                preview += f", ... {len(missing) - 20} more"
            raise SystemExit(
                "exact token ids were not present in tokenizer.json added_tokens: "
                f"count={len(missing)} ids={preview}"
            )
        if missing:
            LOGGER.warning("missing exact token ids in tokenizer.json added_tokens: count=%d", len(missing))

    LOGGER.info(
        "selected=%d changed_tokens=%d changed_token_types=%d skipped=%d",
        len(added_tokens) - skipped,
        len(changed_tokens),
        changed_types,
        skipped,
    )
    for token_id, old, new in changed_tokens[:10]:
        LOGGER.info("token %d: %r -> %r", token_id, old, new)
    if len(changed_tokens) > 10:
        LOGGER.info("... %d additional token string changes", len(changed_tokens) - 10)

    if args.dry_run:
        return
    if args.output.exists() and not args.force:
        raise SystemExit(f"output exists, pass --force to overwrite: {args.output}")

    output_parent = args.output.parent
    if output_parent and not output_parent.exists():
        output_parent.mkdir(parents=True, exist_ok=True)

    new_metadata = {
        gguf.Keys.Tokenizer.LIST: MetadataDetails(
            gguf.GGUFValueType.ARRAY,
            tokens,
            sub_type=gguf.GGUFValueType.STRING,
        )
    }
    if token_types:
        new_metadata[gguf.Keys.Tokenizer.TOKEN_TYPE] = MetadataDetails(
            gguf.GGUFValueType.ARRAY,
            token_types,
            sub_type=gguf.GGUFValueType.INT32,
        )

    LOGGER.info("writing GGUF copy: %s", args.output)
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    alignment = _field_contents(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    _copy_with_metadata(reader, writer, gguf, new_metadata)


if __name__ == "__main__":
    main()
