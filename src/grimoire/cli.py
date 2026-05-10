#!/usr/bin/env python3
"""Grimoire CLI - model management commands."""

import argparse
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone

from grimoire.registry import registry, MODELS_DIR

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "/etc/grimoire/models.json"
REGISTRY_PATH = os.environ.get("GRIMOIRE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)


def cmd_list(args):
    """List all registered models."""
    models = registry.list_all()
    if not models:
        print("No models registered")
        return

    for name in models:
        cfg = registry.get(name)
        active = "✓ active" if name in active_models else "○"
        gpu = cfg.get("gpu", "?")
        file = cfg.get("file", "?")
        print(f"{active} {name:30s} GPU {gpu} {file}")


def cmd_ingest(args):
    """Download and register a new model."""
    model_alias = args.alias
    model_url = args.url
    gpu = args.gpu or 0
    ctx_size = args.ctx_size or 131072

    if not model_alias or not model_url:
        print("Error: Missing --alias or --url", file=sys.stderr)
        sys.exit(1)

    # Download model file
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

    # Add to registry
    try:
        registry.add(model_alias, {
            "file": f"gguf/{model_filename}",
            "mmproj": None,
            "ctx-size": ctx_size,
            "gpu": gpu,
            "has-multimodal": False,
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


def cmd_status(args):
    """Show system status."""
    print("Grimoire System Status")
    print("=" * 50)
    print(f"Registry path: {REGISTRY_PATH}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"GPU count: {len(active_models)} active")
    print()
    print("Active models:")
    for name, data in active_models.items():
        print(f"  {name:30s} GPU {data['gpu']} Port {data['port']}")
    print()
    print("Registered models:")
    for name in registry.list_all():
        cfg = registry.get(name)
        print(f"  {name:30s} GPU {cfg.get('gpu', '?')} {cfg.get('file', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Grimoire model management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all registered models")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Download and register a new model")
    ingest_parser.add_argument("--alias", required=True, help="Model alias name")
    ingest_parser.add_argument("--url", required=True, help="Model file URL")
    ingest_parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    ingest_parser.add_argument("--ctx-size", type=int, default=131072, help="Context size (default: 131072)")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a model from registry")
    remove_parser.add_argument("model", help="Model name to remove")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
