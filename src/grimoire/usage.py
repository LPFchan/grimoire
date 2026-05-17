"""Token and cost accounting for Grimoire."""

import os
import json
import sqlite3
from datetime import datetime, timezone
from threading import RLock

DEFAULT_USAGE_PATH = os.environ.get("GRIMOIRE_USAGE_PATH", "/var/lib/grimoire/usage.sqlite3")
FALLBACK_USAGE_PATH = os.path.expanduser("~/.local/share/grimoire/usage.sqlite3")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _dollar_cost(tokens, rate_per_million):
    if not tokens or tokens <= 0 or not rate_per_million:
        return 0.0
    return tokens / 1_000_000 * float(rate_per_million)


class UsageStore:
    """SQLite-backed token and equivalent-cost tally."""

    def __init__(self, path=DEFAULT_USAGE_PATH):
        self.path = path
        self._lock = RLock()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except PermissionError:
            self.path = FALLBACK_USAGE_PATH
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
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    input_cost REAL NOT NULL,
                    output_cost REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_user_model
                    ON usage_events(user_hash, model);
                CREATE INDEX IF NOT EXISTS idx_usage_created
                    ON usage_events(created_at);

                CREATE TABLE IF NOT EXISTS usage_imports (
                    source TEXT PRIMARY KEY,
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS card_arrangements (
                    user_hash TEXT PRIMARY KEY,
                    card_order TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._migrate_add_cache_columns(conn)

    def _migrate_add_cache_columns(self, conn):
        for col, typ in [("cache_read_input_tokens", "INTEGER"), ("cache_read_input_cost", "REAL")]:
            try:
                conn.execute(f"ALTER TABLE usage_events ADD COLUMN {col} {typ} NOT NULL DEFAULT 0")
            except sqlite3.OperationalError:
                pass

    def record(self, user_hash, model, input_tokens, output_tokens, cost_rates=None, cache_read_input_tokens=None):
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        if input_tokens <= 0 and output_tokens <= 0:
            return

        cache_read_input_tokens = int(cache_read_input_tokens or 0)
        if cache_read_input_tokens > input_tokens:
            cache_read_input_tokens = input_tokens

        cost_rates = cost_rates or {}
        input_cost = _dollar_cost(input_tokens, cost_rates.get("input"))
        output_cost = _dollar_cost(output_tokens, cost_rates.get("output"))
        cache_read_input_cost = _dollar_cost(cache_read_input_tokens, cost_rates.get("cache_read"))

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_events
                    (user_hash, model, input_tokens, output_tokens, input_cost, output_cost,
                     cache_read_input_tokens, cache_read_input_cost, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_hash, model, input_tokens, output_tokens, input_cost, output_cost,
                 cache_read_input_tokens, cache_read_input_cost, utcnow()),
            )

    def summary(self, user_hash=None):
        where = "WHERE user_hash = ?" if user_hash else ""
        params = (user_hash,) if user_hash else ()
        with self._lock, self._connect() as conn:
            total = conn.execute(
                f"""
                SELECT
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(input_cost), 0) AS input_cost,
                    COALESCE(SUM(output_cost), 0) AS output_cost,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cache_read_input_cost), 0) AS cache_read_input_cost
                FROM usage_events
                {where}
                """,
                params,
            ).fetchone()
            models = conn.execute(
                f"""
                SELECT
                    model,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(input_cost), 0) AS input_cost,
                    COALESCE(SUM(output_cost), 0) AS output_cost,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cache_read_input_cost), 0) AS cache_read_input_cost
                FROM usage_events
                {where}
                GROUP BY model
                ORDER BY (SUM(input_tokens) + SUM(output_tokens)) DESC
                """,
                params,
            ).fetchall()

        def row_dict(row):
            input_cost = float(row["input_cost"] or 0)
            output_cost = float(row["output_cost"] or 0)
            cache_read_input_cost = float(row["cache_read_input_cost"] or 0)
            cache_read_input_tokens = int(row["cache_read_input_tokens"] or 0)
            input_tokens = int(row["input_tokens"] or 0)
            output_tokens = int(row["output_tokens"] or 0)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
                "cache_read_input_tokens": cache_read_input_tokens,
                "cache_read_input_cost": cache_read_input_cost,
            }

        return {
            "total": row_dict(total),
            "models": {row["model"]: row_dict(row) for row in models},
        }

    def binned_window(self, user_hash, ts_from, ts_to, bins):
        empty = {
            "input_tokens_series": [0] * max(bins, 0),
            "output_tokens_series": [0] * max(bins, 0),
            "input_cost_series": [0.0] * max(bins, 0),
            "output_cost_series": [0.0] * max(bins, 0),
            "cache_read_input_tokens_series": [0] * max(bins, 0),
            "cache_read_input_cost_series": [0.0] * max(bins, 0),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_input_cost": 0.0,
            "total_output_cost": 0.0,
            "total_cache_read_input_tokens": 0,
            "total_cache_read_input_cost": 0.0,
        }
        if ts_to <= ts_from or bins <= 0:
            return empty
        width = (ts_to - ts_from) / bins
        where = (
            "WHERE CAST(strftime('%s', created_at) AS INTEGER) >= ? "
            "AND CAST(strftime('%s', created_at) AS INTEGER) < ?"
        )
        params = [int(ts_from), int(ts_to)]
        if user_hash is not None:
            where += " AND user_hash = ?"
            params.append(user_hash)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    CAST((CAST(strftime('%s', created_at) AS REAL) - ?) / ? AS INTEGER) AS bin,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(input_cost), 0) AS input_cost,
                    COALESCE(SUM(output_cost), 0) AS output_cost,
                    COALESCE(SUM(cache_read_input_tokens), 0) AS cache_read_input_tokens,
                    COALESCE(SUM(cache_read_input_cost), 0) AS cache_read_input_cost
                FROM usage_events
                {where}
                GROUP BY bin
                """,
                (float(ts_from), float(width), *params),
            ).fetchall()
        in_tok = [0] * bins
        out_tok = [0] * bins
        in_cost = [0.0] * bins
        out_cost = [0.0] * bins
        cr_tok = [0] * bins
        cr_cost = [0.0] * bins
        total_in_tok = 0
        total_out_tok = 0
        total_in_cost = 0.0
        total_out_cost = 0.0
        total_cr_tok = 0
        total_cr_cost = 0.0
        for r in rows:
            try:
                b = int(r["bin"])
            except (TypeError, ValueError):
                continue
            if 0 <= b < bins:
                in_tok[b] = int(r["input_tokens"] or 0)
                out_tok[b] = int(r["output_tokens"] or 0)
                in_cost[b] = float(r["input_cost"] or 0)
                out_cost[b] = float(r["output_cost"] or 0)
                cr_tok[b] = int(r["cache_read_input_tokens"] or 0)
                cr_cost[b] = float(r["cache_read_input_cost"] or 0)
                total_in_tok += in_tok[b]
                total_out_tok += out_tok[b]
                total_in_cost += in_cost[b]
                total_out_cost += out_cost[b]
                total_cr_tok += cr_tok[b]
                total_cr_cost += cr_cost[b]
        return {
            "input_tokens_series": in_tok,
            "output_tokens_series": out_tok,
            "input_cost_series": in_cost,
            "output_cost_series": out_cost,
            "cache_read_input_tokens_series": cr_tok,
            "cache_read_input_cost_series": cr_cost,
            "total_input_tokens": total_in_tok,
            "total_output_tokens": total_out_tok,
            "total_input_cost": total_in_cost,
            "total_output_cost": total_out_cost,
            "total_cache_read_input_tokens": total_cr_tok,
            "total_cache_read_input_cost": total_cr_cost,
        }

    def earliest_event_ts(self, user_hash=None):
        where = "WHERE user_hash = ?" if user_hash else ""
        params = (user_hash,) if user_hash else ()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT MIN(CAST(strftime('%s', created_at) AS INTEGER)) AS t "
                f"FROM usage_events {where}",
                params,
            ).fetchone()
        return int(row["t"]) if row and row["t"] is not None else None

    def import_legacy_token_stats(self, path, user_hash, cost_by_model=None):
        if not path or not os.path.exists(path):
            return False

        source = os.path.abspath(path)
        with self._lock, self._connect() as conn:
            existing = conn.execute("SELECT source FROM usage_imports WHERE source = ?", (source,)).fetchone()
            if existing:
                return False

            with open(path) as f:
                data = json.load(f)

            cost_by_model = cost_by_model or {}
            imported_input = 0
            imported_output = 0
            for model, counts in (data.get("models") or {}).items():
                input_tokens = int(counts.get("input", 0) or 0)
                output_tokens = int(counts.get("output", 0) or 0)
                imported_input += input_tokens
                imported_output += output_tokens
                self._record_with_conn(
                    conn,
                    user_hash,
                    model,
                    input_tokens,
                    output_tokens,
                    cost_by_model.get(model),
                )

            total = data.get("total") or {}
            total_input = int(total.get("input", 0) or 0)
            total_output = int(total.get("output", 0) or 0)
            extra_input = max(0, total_input - imported_input)
            extra_output = max(0, total_output - imported_output)
            if extra_input or extra_output:
                self._record_with_conn(conn, user_hash, "legacy-unattributed", extra_input, extra_output, {})

            conn.execute(
                "INSERT INTO usage_imports (source, imported_at) VALUES (?, ?)",
                (source, utcnow()),
            )
        return True

    def _record_with_conn(self, conn, user_hash, model, input_tokens, output_tokens, cost_rates=None):
        cost_rates = cost_rates or {}
        input_cost = _dollar_cost(input_tokens, cost_rates.get("input"))
        output_cost = _dollar_cost(output_tokens, cost_rates.get("output"))
        conn.execute(
            """
            INSERT INTO usage_events
                (user_hash, model, input_tokens, output_tokens, input_cost, output_cost,
                 cache_read_input_tokens, cache_read_input_cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_hash, model, input_tokens, output_tokens, input_cost, output_cost,
             0, 0.0, utcnow()),
        )


    def save_card_arrangement(self, user_hash, card_order):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO card_arrangements (user_hash, card_order, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_hash) DO UPDATE SET
                    card_order = excluded.card_order,
                    updated_at = excluded.updated_at
                """,
                (user_hash, json.dumps(card_order), utcnow()),
            )

    def load_card_arrangement(self, user_hash):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT card_order FROM card_arrangements WHERE user_hash = ?",
                (user_hash,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["card_order"])
        except (json.JSONDecodeError, TypeError):
            return None


usage_store = UsageStore()
