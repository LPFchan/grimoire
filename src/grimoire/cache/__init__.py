"""Prompt and KV caching.

`KVCacheStore` holds llama.cpp KV slots keyed by a content hash of the prompt
prefix, tiered RAM to disk. `PromptBlock` describes the block layout those
prefixes are addressed by.
"""

from grimoire.cache.blocks import PromptBlock
from grimoire.cache.kv_cache_store import KVCacheStore

__all__ = ["KVCacheStore", "PromptBlock"]
