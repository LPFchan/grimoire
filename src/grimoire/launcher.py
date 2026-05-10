#!/usr/bin/env python3
"""Grimoire launcher - reads model name from CLI, looks up metadata, launches llama-server."""

import argparse
import logging
import os
import sys
import subprocess

from grimoire.registry import registry, MODELS_DIR

logger = logging.getLogger(__name__)

LLAMA_SERVER_BIN = "/opt/model-a-llama-cpp/bin/llama-server"


def parse_args():
    parser = argparse.ArgumentParser(description="Grimoire model launcher")
    parser.add_argument("model", help="Model name (must exist in registry)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--ctx-size", type=int, help="Override context size from registry")
    parser.add_argument("--gpu", type=int, help="Override GPU ID from registry")
    return parser.parse_args()


def build_cmd(cfg, port):
    """Build llama-server command from model config."""
    model_path = os.path.join(MODELS_DIR, cfg["file"])
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    ctx_size = cfg.get("ctx-size", 131072)
    if args.ctx_size:
        ctx_size = args.ctx_size

    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--n-gpu-layers", "999",
        "--jinja",
        "--flash-attn", "on",
        "--metrics",
        "--predict", str(cfg.get("predict", 16384)),
    ]

    if cfg.get("cache-type-k"):
        cmd.extend(["--cache-type-k", cfg["cache-type-k"]])
    if cfg.get("cache-type-v"):
        cmd.extend(["--cache-type-v", cfg["cache-type-v"]])

    if cfg.get("mmproj"):
        mmproj_path = os.path.join(MODELS_DIR, cfg["mmproj"])
        if os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])

    return cmd


def main():
    global args
    args = parse_args()

    cfg = registry.get(args.model)
    if not cfg:
        logger.error(f"Model '{args.model}' not found in registry")
        sys.exit(1)

    gpu = cfg.get("gpu", 0)
    if args.gpu is not None:
        gpu = args.gpu

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = build_cmd(cfg, args.port)

    logger.info(f"Starting {args.model} on GPU {gpu}, port {args.port}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
