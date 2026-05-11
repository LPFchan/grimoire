"""DFlash backend integration for Grimoire.

Wraps the DFlash daemon (test_dflash) and provides:
  - Daemon lifecycle (spawn, health, stop)
  - Prefix cache (LRU KV snapshots with disk persistence)
  - Session KV (per-conversation snapshot tracking)
  - PFlash speculative prefill (long-context compression)
"""

from .daemon import DflashDaemon
from .prefix_cache import PrefixCache
from .prefill import PrefillConfig
from .session_kv import SessionKV

__all__ = ["DflashDaemon", "PrefixCache", "PrefillConfig", "SessionKV"]
