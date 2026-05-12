"""Grimoire configuration constants and environment helpers."""

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid integer for {name}: {value}")
        return default


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


# Backend binary
LLAMA_SERVER_BIN = "/opt/grimoire-llama-cpp/bin/llama-server"

# Model defaults
DEFAULT_CTX_SIZE = 131072
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_PREDICT = 16384

# Auth
API_KEY = os.environ.get("GRIMOIRE_API_KEY", "")
ADMIN_TOKEN = os.environ.get("GRIMOIRE_ADMIN_TOKEN") or API_KEY
COOKIE_NAME = "gw_session"

# Tuning
DEFAULT_STARTUP_TIMEOUT = _env_int("GRIMOIRE_STARTUP_TIMEOUT", 600)
MAX_HISTORY_CAPTURE_BYTES = _env_int("GRIMOIRE_HISTORY_CAPTURE_BYTES", 2 * 1024 * 1024)
MAX_USAGE_CAPTURE_BYTES = _env_int("GRIMOIRE_USAGE_CAPTURE_BYTES", 1024 * 1024)
QWEN_PROMPT_BLOCK_CACHE_SIZE = max(0, _env_int("GRIMOIRE_QWEN_PROMPT_BLOCK_CACHE_SIZE", 2048))
LEGACY_STATS_PATH = os.environ.get("GRIMOIRE_LEGACY_STATS_PATH", "/var/lib/grimoire/token-stats.json")
ALLOW_ANONYMOUS = _env_bool("GRIMOIRE_ALLOW_ANONYMOUS", False)
WEBUI_DIR = os.environ.get("GRIMOIRE_WEBUI_DIR", "/opt/grimoire-webui")

# Proxy header filtering
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SENSITIVE_PROXY_HEADERS = {
    "authorization",
    "cookie",
    "x-grimoire-token",
    "x-api-key",
}

# Process management
PR_SET_PDEATHSIG = 1

# Model status lifecycle
MODEL_STATUS_UNLOADED = "unloaded"
MODEL_STATUS_LOADING = "loading"
MODEL_STATUS_LOADED = "loaded"
MODEL_STATUS_FAILED = "failed"

# Dashboard
DASHBOARD_WINDOWS_S = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
}
DASHBOARD_BINS = 60

# DFlash
DFLASH_PROTECTED_TOOLS = {"obsidian_read-note"}
DFLASH_SUPPORTED_SAMPLING = {"temperature", "top_p", "top_k", "seed"}
DFLASH_IGNORED_SAMPLING = {
    "min_p", "repetition_penalty", "frequency_penalty",
    "presence_penalty", "logit_bias", "typical_p", "tfs_z", "mirostat",
}

# Default generation parameters (for /props synthetic response)
DEFAULT_GENERATION_PARAMS = {
    "n_predict": DEFAULT_PREDICT,
    "seed": -1,
    "temperature": 0.8,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "typ_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "dry_sequence_breakers": [],
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "stop": [],
    "max_tokens": DEFAULT_PREDICT,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": False,
    "stream": True,
    "logit_bias": [],
    "n_probs": 0,
    "min_keep": 0,
    "grammar": "",
    "grammar_lazy": False,
    "grammar_triggers": [],
    "preserved_tokens": [],
    "chat_format": "",
    "reasoning_format": "auto",
    "reasoning_in_content": False,
    "generation_prompt": "",
    "samplers": [],
    "backend_sampling": False,
    "speculative.n_max": 16,
    "speculative.n_min": 0,
    "speculative.p_min": 0.75,
    "timings_per_token": False,
    "post_sampling_probs": False,
    "lora": [],
}
