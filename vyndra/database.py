"""SQLite persistence for VYNDRA. Uses parameterized SQL and a local file database."""
from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("VYNDRA_DB_PATH", str(ROOT / "data" / "vyndra.sqlite3"))).expanduser()
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meals (
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 source TEXT NOT NULL DEFAULT 'manual', notes TEXT NOT NULL DEFAULT '', total_calories REAL NOT NULL DEFAULT 0,
 total_protein_g REAL NOT NULL DEFAULT 0, total_carbs_g REAL NOT NULL DEFAULT 0, total_fat_g REAL NOT NULL DEFAULT 0,
 total_fiber_g REAL NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS meal_items (
 id INTEGER PRIMARY KEY AUTOINCREMENT, meal_id INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
 food_name TEXT NOT NULL, quantity_g REAL NOT NULL CHECK(quantity_g > 0 AND quantity_g <= 5000),
 calories REAL NOT NULL DEFAULT 0, protein_g REAL NOT NULL DEFAULT 0, carbs_g REAL NOT NULL DEFAULT 0,
 fat_g REAL NOT NULL DEFAULT 0, fiber_g REAL NOT NULL DEFAULT 0, match_type TEXT NOT NULL DEFAULT 'nutrition_lookup'
);
CREATE TABLE IF NOT EXISTS chat_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 intent TEXT NOT NULL DEFAULT '', sentiment TEXT NOT NULL DEFAULT '', food_mentions_json TEXT NOT NULL DEFAULT '[]',
 text_length INTEGER NOT NULL DEFAULT 0, text_sha256 TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_meals_created_at ON meals(created_at);
CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id ON meal_items(meal_id);
"""
@contextmanager
def connect():
 DB_PATH.parent.mkdir(parents=True, exist_ok=True)
 conn=sqlite3.connect(str(DB_PATH), timeout=10)
 conn.row_factory=sqlite3.Row
 try:
  conn.execute('PRAGMA foreign_keys=ON')
  yield conn
  conn.commit()
 except Exception:
  conn.rollback(); raise
 finally: conn.close()
def init_db():
 with connect() as c: c.executescript(SCHEMA)
def save_meal(items, source='manual', notes='', metadata=None):
 init_db()
 totals={k:sum(float(i.get(k,0) or 0) for i in items) for k in ['calories','protein_g','carbs_g','fat_g','fiber_g']}
 with connect() as c:
  cur=c.execute('INSERT INTO meals(source,notes,total_calories,total_protein_g,total_carbs_g,total_fat_g,total_fiber_g,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
   (source, notes, totals['calories'], totals['protein_g'], totals['carbs_g'], totals['fat_g'], totals['fiber_g'], json.dumps(metadata or {})))
  meal_id=cur.lastrowid
  for i in items:
   c.execute('INSERT INTO meal_items(meal_id,food_name,quantity_g,calories,protein_g,carbs_g,fat_g,fiber_g,match_type) VALUES(?,?,?,?,?,?,?,?,?)',
    (meal_id,i['food_name'],i['quantity_g'],i.get('calories',0),i.get('protein_g',0),i.get('carbs_g',0),i.get('fat_g',0),i.get('fiber_g',0),i.get('match_type','nutrition_lookup')))
 return get_meal(meal_id)
def get_meal(meal_id):
 init_db()
 with connect() as c:
  row=c.execute('SELECT * FROM meals WHERE id=?',(meal_id,)).fetchone()
  if row is None: return None
  out=dict(row); out['metadata']=json.loads(out.pop('metadata_json'))
  out['items']=[dict(r) for r in c.execute('SELECT id,food_name,quantity_g,calories,protein_g,carbs_g,fat_g,fiber_g,match_type FROM meal_items WHERE meal_id=? ORDER BY id',(meal_id,)).fetchall()]
  return out
def list_meals(limit=20):
 init_db()
 with connect() as c: return [dict(r) for r in c.execute('SELECT id,created_at,source,notes,total_calories,total_protein_g,total_carbs_g,total_fat_g,total_fiber_g FROM meals ORDER BY id DESC LIMIT ?', (limit,)).fetchall()]
def save_chat_event(intent, sentiment, mentions, text):
 import hashlib
 init_db()
 with connect() as c:
  c.execute('INSERT INTO chat_events(intent,sentiment,food_mentions_json,text_length,text_sha256) VALUES(?,?,?,?,?)',
   (str(intent)[:80],str(sentiment)[:80],json.dumps(mentions),len(text),hashlib.sha256(text.encode('utf-8')).hexdigest()))
