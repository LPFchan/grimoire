"""Compatibility wrapper for the Grimoire gateway.

The canonical gateway implementation lives in grimoire.entrypoint. This module
keeps the historical `grimoire-gateway` console script and ASGI import path from
drifting into a second, inconsistent implementation.
"""

from grimoire.entrypoint import app, main, manager

__all__ = ["app", "main", "manager"]
