"""Pytest setup: ensure grimoire env vars are set before any test module imports the package."""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("GRIMOIRE_HISTORY_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-history.sqlite3"))
os.environ.setdefault("GRIMOIRE_USAGE_PATH", str(Path(tempfile.gettempdir()) / "grimoire-test-usage.sqlite3"))
os.environ.setdefault("GRIMOIRE_REGISTRY_SEED_PATH", str(ROOT / "etc" / "models.json"))
os.environ.setdefault(
    "GRIMOIRE_REGISTRY_PATH",
    str(Path(tempfile.gettempdir()) / "grimoire-test-registry.json"),
)
