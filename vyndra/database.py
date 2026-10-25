"""SQLite persistence for VYNDRA users, meals, messages, and privacy-safe events."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("VYNDRA_DB_PATH", str(ROOT / "data" / "vyndra.sqlite3"))).expanduser()
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, display_name TEXT NOT NULL DEFAULT '',
    dietary_preferences TEXT NOT NULL DEFAULT '[]', allergens TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS meals (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 source TEXT NOT NULL DEFAULT 'manual', notes TEXT NOT NULL DEFAULT '', user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
 total_calories REAL NOT NULL DEFAULT 0, total_protein_g REAL NOT NULL DEFAULT 0,
 total_carbs_g REAL NOT NULL DEFAULT 0, total_fat_g REAL NOT NULL DEFAULT 0,
 total_fiber_g REAL NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS meal_items (
 id INTEGER PRIMARY KEY AUTOINCREMENT, meal_id INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
 food_name TEXT NOT NULL, quantity_g REAL NOT NULL CHECK(quantity_g > 0 AND quantity_g <= 5000),
 calories REAL NOT NULL DEFAULT 0, protein_g REAL NOT NULL DEFAULT 0, carbs_g REAL NOT NULL DEFAULT 0,
 fat_g REAL NOT NULL DEFAULT 0, fiber_g REAL NOT NULL DEFAULT 0, match_type TEXT NOT NULL DEFAULT 'nutrition_lookup'
);
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 intent_label TEXT NOT NULL DEFAULT 'not_food', probability REAL NOT NULL DEFAULT 0,
 meal_id INTEGER REFERENCES meals(id) ON DELETE SET NULL, nudge INTEGER NOT NULL DEFAULT 0,
 nudge_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS reactions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
 meal_id INTEGER REFERENCES meals(id) ON DELETE CASCADE, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 reaction TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 CHECK(message_id IS NOT NULL OR meal_id IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS chat_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 intent TEXT NOT NULL DEFAULT '', sentiment TEXT NOT NULL DEFAULT '', food_mentions_json TEXT NOT NULL DEFAULT '[]',
 text_length INTEGER NOT NULL DEFAULT 0, text_sha256 TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_meals_created_at ON meals(created_at);
CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id ON meal_items(meal_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at);
"""


@contextmanager
def connect():
    """Yield a transaction-managed SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create new tables and safely add columns to databases from earlier versions."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(meals)")}
        if "user_id" not in columns:
            conn.execute("ALTER TABLE meals ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE SET NULL")
        # Create this index only after migrating an older meals table that lacked user_id.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meals_user_id ON meals(user_id)")


def create_user(user_id: str, display_name: str = "", dietary_preferences=None, allergens=None) -> dict:
    init_db()
    preferences = json.dumps(dietary_preferences or [], ensure_ascii=False)
    allergy_json = json.dumps(allergens or [], ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            "INSERT INTO users(id,display_name,dietary_preferences,allergens) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name, "
            "dietary_preferences=excluded.dietary_preferences, allergens=excluded.allergens",
            (user_id, display_name, preferences, allergy_json),
        )
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _user_dict(row)


def _user_dict(row) -> dict:
    out = dict(row)
    out["dietary_preferences"] = json.loads(out["dietary_preferences"])
    out["allergens"] = json.loads(out["allergens"])
    return out


def get_user(user_id: str):
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _user_dict(row) if row else None


def list_users(limit: int = 100) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_user_dict(row) for row in rows]


def save_meal(items: list[dict], source: str = "manual", notes: str = "", metadata=None, user_id=None) -> dict:
    init_db()
    totals = {key: sum(float(item.get(key, 0) or 0) for item in items) for key in
              ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]}
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO meals(source,notes,user_id,total_calories,total_protein_g,total_carbs_g,total_fat_g,total_fiber_g,metadata_json) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (source, notes, user_id, totals["calories"], totals["protein_g"], totals["carbs_g"],
             totals["fat_g"], totals["fiber_g"], json.dumps(metadata or {}, ensure_ascii=False)),
        )
        meal_id = cur.lastrowid
        for item in items:
            conn.execute(
                "INSERT INTO meal_items(meal_id,food_name,quantity_g,calories,protein_g,carbs_g,fat_g,fiber_g,match_type) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (meal_id, item["food_name"], item["quantity_g"], item.get("calories", 0),
                 item.get("protein_g", 0), item.get("carbs_g", 0), item.get("fat_g", 0),
                 item.get("fiber_g", 0), item.get("match_type", "nutrition_lookup")),
            )
    return get_meal(meal_id)


def get_meal(meal_id: int):
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM meals WHERE id=?", (meal_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["metadata"] = json.loads(out.pop("metadata_json"))
        out["items"] = [dict(r) for r in conn.execute(
            "SELECT id,food_name,quantity_g,calories,protein_g,carbs_g,fat_g,fiber_g,match_type FROM meal_items WHERE meal_id=? ORDER BY id",
            (meal_id,),
        ).fetchall()]
        return out


def list_meals(limit: int = 20, user_id=None) -> list[dict]:
    init_db()
    with connect() as conn:
        if user_id is not None:
            rows = conn.execute("SELECT id,created_at,source,notes,user_id,total_calories,total_protein_g,total_carbs_g,total_fat_g,total_fiber_g FROM meals WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        else:
            rows = conn.execute("SELECT id,created_at,source,notes,user_id,total_calories,total_protein_g,total_carbs_g,total_fat_g,total_fiber_g FROM meals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def save_message(user_id: str, conversation_id: str, text: str, intent_label: str, probability: float,
                 nudge: bool, nudge_text: str = "", meal_id=None) -> dict:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages(user_id,conversation_id,text,intent_label,probability,meal_id,nudge,nudge_text) VALUES(?,?,?,?,?,?,?,?)",
            (user_id, conversation_id, text, intent_label, probability, meal_id, int(nudge), nudge_text),
        )
        row = conn.execute("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)).fetchone()
    return _message_dict(row)


def _message_dict(row) -> dict:
    out = dict(row)
    out["nudge"] = bool(out["nudge"])
    return out


def get_conversation(conversation_id: str) -> list[dict]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    return [_message_dict(row) for row in rows]


def save_chat_event(intent, sentiment, mentions, text):
    """Persist only derived metadata and a one-way digest for the legacy analysis endpoint."""
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_events(intent,sentiment,food_mentions_json,text_length,text_sha256) VALUES(?,?,?,?,?)",
            (str(intent)[:80], str(sentiment)[:80], json.dumps(mentions), len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()),
        )


def user_meal_rows(user_id: str, since: str):
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT m.id AS meal_id, m.created_at, m.total_calories, m.total_protein_g, m.total_carbs_g, m.total_fat_g, m.total_fiber_g, i.food_name "
            "FROM meals m LEFT JOIN meal_items i ON i.meal_id=m.id WHERE m.user_id=? AND m.created_at>=? ORDER BY m.created_at DESC",
            (user_id, since),
        ).fetchall()
    return [dict(row) for row in rows]
