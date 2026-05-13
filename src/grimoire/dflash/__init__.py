"""DFlash backend integration for Grimoire.

Wraps the DFlash daemon (test_dflash) and provides:
  - Daemon lifecycle (spawn, health, stop)
  - Prefix cache (LRU KV snapshots with disk persistence)
  - PFlash speculative prefill (long-context compression)
"""

from .daemon import DflashDaemon
from .prefix_cache import PrefixCache
from .prefill import PrefillConfig

__all__ = ["DflashDaemon", "PrefixCache", "PrefillConfig"]
