#!/usr/bin/env python3
"""Grimoire launcher - reads model name from CLI, looks up metadata, launches llama-server."""

import argparse
import logging
import os
import sys
import subprocess

from grimoire.registry import registry, MODELS_DIR
from grimoire.model_manager import (
    GpuPlacement,
    configure_gpu_environment,
    detect_gpu_count,
    effective_extra_args,
)

logger = logging.getLogger(__name__)

LLAMA_SERVER_BIN = "/opt/grimoire-llama-cpp/bin/llama-server"


def parse_args():
    parser = argparse.ArgumentParser(description="Grimoire model launcher")
    parser.add_argument("model", help="Model name (must exist in registry)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--ctx-size", type=int, help="Override context size from registry")
    parser.add_argument("--gpu", type=int, help="Override GPU ID from registry")
    return parser.parse_args()


def build_cmd(cfg, port, ctx_size_override=None):
    """Build llama-server command from model config."""
    model_path = os.path.join(MODELS_DIR, cfg["file"])
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    ctx_size = cfg.get("ctx-size", 131072)
    if ctx_size_override:
        ctx_size = ctx_size_override

    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--host", "127.0.0.1",
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
        if not os.path.exists(mmproj_path):
            logger.error(f"MMProj file not found: {mmproj_path}")
            sys.exit(1)
        cmd.extend(["--mmproj", mmproj_path])

    cmd.extend(effective_extra_args(cfg))

    return cmd


def main():
    args = parse_args()

    cfg = registry.get(args.model)
    if not cfg:
        logger.error(f"Model '{args.model}' not found in registry")
        sys.exit(1)

    gpu_count = detect_gpu_count()
    valid, reason = registry.validate(args.model, gpu_count=gpu_count)
    if not valid:
        logger.error(reason)
        sys.exit(1)

    pinned_gpu = registry.get_fixed_gpu(args.model)
    configured_gpu_ids = cfg.get("gpu-ids")
    if args.gpu is not None:
        if args.gpu < 0:
            logger.error("GPU ID must be a non-negative integer")
            sys.exit(1)
        if configured_gpu_ids is not None:
            logger.error("--gpu cannot override a model configured with 'gpu-ids'")
            sys.exit(1)
        placement = GpuPlacement((args.gpu,))
    elif configured_gpu_ids is not None:
        placement = GpuPlacement(tuple(configured_gpu_ids))
    else:
        placement = GpuPlacement((pinned_gpu if pinned_gpu is not None else 0,))

    env = os.environ.copy()
    try:
        configure_gpu_environment(env, cfg, placement)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    cmd = build_cmd(cfg, args.port, args.ctx_size)

    logger.info(f"Starting {args.model} on GPU placement {list(placement.device_ids)}, port {args.port}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
