"""DFlash backend integration for Grimoire.

Wraps the DFlash daemon (test_dflash) and provides:
  - Daemon lifecycle (spawn, health, stop)
  - Prefix cache (persisted compact full prefix snapshots)
  - Session KV (per-conversation compact full snapshot tracking)
  - Snapshot store (RAM-backed with async disk mirroring)
  - PFlash speculative prefill (long-context compression)
"""

from .daemon import DflashDaemon, PflashDaemon
from .prefix_cache import PrefixCache
from .prefill import PrefillConfig
from .session_kv import SessionKV
from .snapshot_swap import SnapshotSwap

__all__ = ["DflashDaemon", "PflashDaemon", "PrefixCache", "PrefillConfig", "SessionKV", "SnapshotSwap"]
