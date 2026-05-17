"""PFlash compression backend for Grimoire.

Provides PFlash daemon lifecycle and prompt prefill compression
for long-context prompts.
"""

from .daemon import PflashDaemon
from .prefill import PrefillConfig

__all__ = ["PflashDaemon", "PrefillConfig"]
