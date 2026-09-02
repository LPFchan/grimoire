"""KV cache for llama-server slots.

`KVCacheStore` holds slots keyed by a content hash of the prompt prefix, tiered
from RAM (tmpfs) to disk.
"""

from grimoire.cache.kv_cache_store import KVCacheStore

__all__ = ["KVCacheStore"]
