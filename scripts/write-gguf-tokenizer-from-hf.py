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
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing output")
    parser.add_argument("--force", action="store_true", help="overwrite output if it already exists")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

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
    for token in added_tokens:
        if not isinstance(token, dict) or "id" not in token or "content" not in token:
            skipped += 1
            continue
        token_id = int(token["id"])
        if token_id < args.min_id or (args.max_id is not None and token_id > args.max_id):
            skipped += 1
            continue
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
