"""Server-side per-api-key settings store.

Mirrors the browser localStorage config (sampling params, UI prefs, MCP servers)
so settings persist across devices.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from threading import RLock

from grimoire.history import identity_hash, utcnow

DEFAULT_SETTINGS_PATH = os.environ.get("GRIMOIRE_SETTINGS_PATH", "/var/lib/grimoire/settings.sqlite3")
FALLBACK_SETTINGS_PATH = os.path.expanduser("~/.local/share/grimoire/settings.sqlite3")


class SettingsStore:
    """SQLite-backed settings store keyed by API-key hash."""

    def __init__(self, path=DEFAULT_SETTINGS_PATH):
        self.path = path
        self._lock = RLock()
        self._init_db()

    def _connect(self):
        try:
            return sqlite3.connect(self.path, check_same_thread=False)
        except (PermissionError, sqlite3.OperationalError):
            alt = FALLBACK_SETTINGS_PATH
            os.makedirs(os.path.dirname(alt), exist_ok=True)
            return sqlite3.connect(alt, check_same_thread=False)

    def _init_db(self):
        with self._lock:
            db = self._connect()
            try:
                db.execute(
                    """CREATE TABLE IF NOT EXISTS settings (
                        user_hash TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_hash, key)
                    )"""
                )
                db.commit()
            finally:
                db.close()

    def get_all(self, user_hash: str) -> dict[str, str]:
        with self._lock:
            db = self._connect()
            try:
                rows = db.execute(
                    "SELECT key, value FROM settings WHERE user_hash = ?", (user_hash,)
                ).fetchall()
                return {row[0]: row[1] for row in rows}
            finally:
                db.close()

    def get(self, user_hash: str, key: str) -> str | None:
        with self._lock:
            db = self._connect()
            try:
                row = db.execute(
                    "SELECT value FROM settings WHERE user_hash = ? AND key = ?", (user_hash, key)
                ).fetchone()
                return row[0] if row else None
            finally:
                db.close()

    def set(self, user_hash: str, key: str, value: str):
        with self._lock:
            db = self._connect()
            try:
                db.execute(
                    """INSERT OR REPLACE INTO settings (user_hash, key, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (user_hash, key, value, utcnow()),
                )
                db.commit()
            finally:
                db.close()

    def set_many(self, user_hash: str, kv: dict[str, str]):
        with self._lock:
            db = self._connect()
            try:
                now = utcnow()
                db.executemany(
                    """INSERT OR REPLACE INTO settings (user_hash, key, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    [(user_hash, k, v, now) for k, v in kv.items()],
                )
                db.commit()
            finally:
                db.close()

    def delete(self, user_hash: str, key: str):
        with self._lock:
            db = self._connect()
            try:
                db.execute(
                    "DELETE FROM settings WHERE user_hash = ? AND key = ?", (user_hash, key)
                )
                db.commit()
            finally:
                db.close()

    def import_bulk(self, user_hash: str, kv: dict[str, str]):
        self.set_many(user_hash, kv)


settings_store = SettingsStore()
