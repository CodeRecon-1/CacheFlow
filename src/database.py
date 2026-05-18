"""
Database layer: SQLite for metadata/analytics + ChromaDB for semantic search
"""
import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

DB_PATH = Path("data/optimizer.db")
CHROMA_PATH = Path("data/chroma")


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_chroma() -> chromadb.Client:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )


def init_db():
    """Initialize all SQLite tables."""
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS cache_entries (
            id          TEXT PRIMARY KEY,
            prompt_hash TEXT NOT NULL,
            prompt      TEXT NOT NULL,
            response    TEXT NOT NULL,
            model       TEXT NOT NULL,
            provider    TEXT NOT NULL,
            tokens_in   INTEGER DEFAULT 0,
            tokens_out  INTEGER DEFAULT 0,
            created_at  REAL NOT NULL,
            last_hit    REAL,
            hit_count   INTEGER DEFAULT 0,
            ttl         INTEGER DEFAULT 86400
        );

        CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache_entries(prompt_hash);
        CREATE INDEX IF NOT EXISTS idx_cache_model ON cache_entries(model);

        CREATE TABLE IF NOT EXISTS request_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            REAL NOT NULL,
            provider      TEXT NOT NULL,
            model         TEXT NOT NULL,
            cache_hit     INTEGER NOT NULL,
            hit_type      TEXT,          -- 'exact' | 'semantic' | 'mock' | null
            similarity    REAL,
            tokens_in     INTEGER DEFAULT 0,
            tokens_out    INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0,
            latency_ms    INTEGER DEFAULT 0,
            saved_usd     REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS budgets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            limit_usd   REAL NOT NULL,
            spent_usd   REAL DEFAULT 0,
            period      TEXT DEFAULT 'monthly',
            reset_at    REAL,
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS mock_templates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            pattern     TEXT NOT NULL,
            response    TEXT NOT NULL,
            model       TEXT,
            enabled     INTEGER DEFAULT 1,
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS variants (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id    TEXT NOT NULL,
            prompt      TEXT NOT NULL,
            variant_idx INTEGER NOT NULL,
            response    TEXT NOT NULL,
            model       TEXT NOT NULL,
            created_at  REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_variants_group ON variants(group_id);
        CREATE INDEX IF NOT EXISTS idx_log_ts ON request_log(ts);
    """)

    conn.commit()
    conn.close()


def hash_prompt(prompt: str, model: str, system: str = "") -> str:
    key = f"{model}::{system}::{prompt}"
    return hashlib.sha256(key.encode()).hexdigest()


# ─── Cache helpers ───────────────────────────────────────────────────────────

def cache_get_exact(prompt_hash: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM cache_entries WHERE prompt_hash=? AND (created_at + ttl) > ?",
        (prompt_hash, time.time())
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE cache_entries SET hit_count=hit_count+1, last_hit=? WHERE id=?",
            (time.time(), row["id"])
        )
        conn.commit()
    conn.close()
    return dict(row) if row else None


def cache_set(entry_id: str, prompt_hash: str, prompt: str, response: str,
              model: str, provider: str, tokens_in: int, tokens_out: int, ttl: int = 86400):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO cache_entries
        (id, prompt_hash, prompt, response, model, provider, tokens_in, tokens_out, created_at, ttl)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (entry_id, prompt_hash, prompt, response, model, provider, tokens_in, tokens_out, time.time(), ttl))
    conn.commit()
    conn.close()


def cache_list(limit: int = 50) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM cache_entries ORDER BY last_hit DESC, created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cache_delete(entry_id: str):
    conn = get_db()
    conn.execute("DELETE FROM cache_entries WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()


def cache_clear():
    conn = get_db()
    conn.execute("DELETE FROM cache_entries")
    conn.commit()
    conn.close()


# ─── Analytics helpers ────────────────────────────────────────────────────────

def log_request(provider: str, model: str, cache_hit: bool, hit_type: Optional[str],
                similarity: Optional[float], tokens_in: int, tokens_out: int,
                cost_usd: float, latency_ms: int, saved_usd: float):
    conn = get_db()
    conn.execute("""
        INSERT INTO request_log
        (ts, provider, model, cache_hit, hit_type, similarity, tokens_in, tokens_out, cost_usd, latency_ms, saved_usd)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (time.time(), provider, model, int(cache_hit), hit_type, similarity,
          tokens_in, tokens_out, cost_usd, latency_ms, saved_usd))
    conn.commit()
    conn.close()


def get_analytics(days: int = 7) -> dict:
    since = time.time() - days * 86400
    conn = get_db()
    c = conn.cursor()

    totals = c.execute("""
        SELECT
            COUNT(*) as total_requests,
            SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) as cache_hits,
            SUM(cost_usd) as total_cost,
            SUM(saved_usd) as total_saved,
            SUM(tokens_in + tokens_out) as total_tokens,
            AVG(latency_ms) as avg_latency
        FROM request_log WHERE ts >= ?
    """, (since,)).fetchone()

    by_model = c.execute("""
        SELECT model, COUNT(*) as reqs, SUM(cost_usd) as cost,
               SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) as hits
        FROM request_log WHERE ts >= ?
        GROUP BY model ORDER BY reqs DESC
    """, (since,)).fetchall()

    by_day = c.execute("""
        SELECT DATE(ts, 'unixepoch') as day,
               COUNT(*) as reqs, SUM(cost_usd) as cost, SUM(saved_usd) as saved,
               SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) as hits
        FROM request_log WHERE ts >= ?
        GROUP BY day ORDER BY day
    """, (since,)).fetchall()

    hit_types = c.execute("""
        SELECT hit_type, COUNT(*) as cnt
        FROM request_log WHERE ts >= ? AND cache_hit=1
        GROUP BY hit_type
    """, (since,)).fetchall()

    conn.close()

    t = dict(totals) if totals else {}
    hit_rate = (t.get("cache_hits", 0) / t["total_requests"] * 100) if t.get("total_requests") else 0

    return {
        "totals": {**t, "hit_rate": round(hit_rate, 1)},
        "by_model": [dict(r) for r in by_model],
        "by_day": [dict(r) for r in by_day],
        "hit_types": [dict(r) for r in hit_types],
    }


# ─── Mock templates ───────────────────────────────────────────────────────────

def mock_list() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM mock_templates ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mock_add(name: str, pattern: str, response: str, model: Optional[str] = None) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO mock_templates (name, pattern, response, model, created_at) VALUES (?,?,?,?,?)",
        (name, pattern, response, model, time.time())
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def mock_delete(mock_id: int):
    conn = get_db()
    conn.execute("DELETE FROM mock_templates WHERE id=?", (mock_id,))
    conn.commit()
    conn.close()


# ─── Budgets ─────────────────────────────────────────────────────────────────

def budget_list() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM budgets ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def budget_add(name: str, limit_usd: float, period: str = "monthly") -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO budgets (name, limit_usd, period, created_at) VALUES (?,?,?,?)",
        (name, limit_usd, period, time.time())
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def budget_spend(name: str, amount: float):
    conn = get_db()
    conn.execute("UPDATE budgets SET spent_usd=spent_usd+? WHERE name=?", (amount, name))
    conn.commit()
    conn.close()


def budget_delete(budget_id: int):
    conn = get_db()
    conn.execute("DELETE FROM budgets WHERE id=?", (budget_id,))
    conn.commit()
    conn.close()
