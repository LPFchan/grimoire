"""Server-side per-api-key conversation history."""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from threading import RLock


DEFAULT_HISTORY_PATH = os.environ.get("GRIMOIRE_HISTORY_PATH", "/var/lib/grimoire/history.sqlite3")
FALLBACK_HISTORY_PATH = os.path.expanduser("~/.local/share/grimoire/history.sqlite3")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def identity_hash(token):
    """Hash an API key into a stable non-secret identity key."""
    if not token:
        token = "anonymous"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class HistoryStore:
    """Small SQLite-backed conversation store keyed by API-key hash."""

    def __init__(self, path=DEFAULT_HISTORY_PATH):
        self.path = path
        self._lock = RLock()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except PermissionError:
            self.path = FALLBACK_HISTORY_PATH
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_hash, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);
                """
            )

    def _ensure_owner(self, conn, user_hash, conversation_id):
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_hash = ?",
            (conversation_id, user_hash),
        ).fetchone()
        if not row:
            raise KeyError(f"Conversation '{conversation_id}' not found")

    def list_conversations(self, user_hash, limit=100):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                WHERE user_hash = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_hash, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def create_conversation(self, user_hash, title="New chat", model=None, messages=None):
        conversation_id = str(uuid.uuid4())
        now = utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_hash, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_hash, title or "New chat", model, now, now),
            )
            for message in messages or []:
                self._insert_message(conn, conversation_id, message.get("role", "user"), message.get("content"))
        return self.get_conversation(user_hash, conversation_id)

    def get_conversation(self, user_hash, conversation_id):
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            conversation = conn.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_hash = ?
                """,
                (conversation_id, user_hash),
            ).fetchone()
            messages = conn.execute(
                """
                SELECT id, role, content_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

            result = dict(conversation)
            result["messages"] = [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": json.loads(row["content_json"]),
                    "created_at": row["created_at"],
                }
                for row in messages
            ]
            return result

    def replace_conversation(self, user_hash, conversation_id, title=None, model=None, messages=None):
        now = utcnow()
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            updates = ["updated_at = ?"]
            values = [now]
            if title is not None:
                updates.append("title = ?")
                values.append(title or "New chat")
            if model is not None:
                updates.append("model = ?")
                values.append(model)
            values.append(conversation_id)
            conn.execute(f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", values)

            if messages is not None:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                for message in messages:
                    self._insert_message(conn, conversation_id, message.get("role", "user"), message.get("content"))
        return self.get_conversation(user_hash, conversation_id)

    def delete_conversation(self, user_hash, conversation_id):
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def append_message(self, user_hash, conversation_id, role, content, model=None):
        now = utcnow()
        with self._lock, self._connect() as conn:
            self._ensure_owner(conn, user_hash, conversation_id)
            self._insert_message(conn, conversation_id, role, content)
            if model is not None:
                conn.execute(
                    "UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                    (model, now, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

    def _insert_message(self, conn, conversation_id, role, content):
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), conversation_id, role, json.dumps(content), utcnow()),
        )


history_store = HistoryStore()
