#!/usr/bin/env python3
"""Grimoire CLI - model management commands."""

import argparse
import json
import logging
import os
import sys
import urllib.request

from grimoire.registry import registry, MODELS_DIR

logger = logging.getLogger(__name__)


def cmd_list(args):
    """List all registered models."""
    models = registry.list_all()
    if not models:
        print("No models registered")
        return

    fixed = registry._data.get("fixed", {})
    for name in models:
        cfg = registry.get(name)
        pinned = name in fixed
        gpu_label = f"GPU {fixed[name]} (pinned)" if pinned else "dynamic"
        file = cfg.get("file", "?")
        print(f"{'📌' if pinned else '○'} {name:30s} {gpu_label:16s} {file}")


def cmd_ingest(args):
    """Download and register a new model."""
    model_alias = args.alias
    model_url = args.url
    ctx_size = args.ctx_size or 131072

    if not model_alias or not model_url:
        print("Error: Missing --alias or --url", file=sys.stderr)
        sys.exit(1)

    model_filename = model_url.split("/")[-1]
    model_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Error: Model file already exists at {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Downloading model from {model_url} to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete")
    except Exception as e:
        print(f"Failed to download model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        registry.add(model_alias, {
            "file": f"gguf/{model_filename}",
            "mmproj": None,
            "ctx-size": ctx_size,
        })
        print(f"Added model {model_alias} to registry")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_remove(args):
    """Remove a model from the registry."""
    try:
        registry.remove(args.model)
        print(f"Removed model {args.model}")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_pin(args):
    """Pin a model to a specific GPU."""
    try:
        registry.pin_gpu(args.model, args.gpu)
        print(f"Pinned {args.model} to GPU {args.gpu}")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_unpin(args):
    """Remove GPU pinning for a model."""
    try:
        registry.unpin_gpu(args.model)
        print(f"Unpinned {args.model}")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Show system status."""
    print("Grimoire System Status")
    print("=" * 50)
    print(f"Registry path: {registry.path}")
    print(f"Models directory: {MODELS_DIR}")
    print()

    fixed = registry._data.get("fixed", {})
    print("Pinned models:")
    if fixed:
        for name, gpu in fixed.items():
            print(f"  {name:30s} → GPU {gpu}")
    else:
        print("  (none)")
    print()

    print("All models:")
    for name in registry.list_all():
        cfg = registry.get(name)
        pinned = name in fixed
        print(f"  {name:30s} {'(pinned)' if pinned else ''} {cfg.get('file', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Grimoire model management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    subparsers.add_parser("list", help="List all registered models")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Download and register a new model")
    ingest_parser.add_argument("--alias", required=True, help="Model alias name")
    ingest_parser.add_argument("--url", required=True, help="Model file URL")
    ingest_parser.add_argument("--ctx-size", type=int, default=131072, help="Context size")

    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a model from registry")
    remove_parser.add_argument("model", help="Model name to remove")

    # pin
    pin_parser = subparsers.add_parser("pin", help="Pin a model to a GPU")
    pin_parser.add_argument("model", help="Model name")
    pin_parser.add_argument("gpu", type=int, help="GPU ID")

    # unpin
    unpin_parser = subparsers.add_parser("unpin", help="Remove GPU pinning")
    unpin_parser.add_argument("model", help="Model name")

    # status
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "list": cmd_list,
        "ingest": cmd_ingest,
        "remove": cmd_remove,
        "pin": cmd_pin,
        "unpin": cmd_unpin,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
