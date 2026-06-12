"""
db.py — SQLite persistence layer for ReStoreAI
Replaces the in-memory report_storage and analysis_history dicts.
Reports and analytics survive server restarts on Render.
"""

import sqlite3
import json
import os

DB_PATH = os.environ.get("DB_PATH", "restoreai.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables on first run. Safe to call every startup."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                data        TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT,
                device         TEXT,
                confidence     REAL,
                repairability  TEXT,
                repairability_class TEXT,
                damage         REAL,
                age            INTEGER,
                location       TEXT,
                cracks         REAL,
                rust           REAL,
                broken         REAL,
                cost_min       INTEGER,
                cost_max       INTEGER
            )
        """)
        conn.commit()


def save_report(report_id: str, created_at: str, data: dict):
    """Persist a full report as JSON blob."""
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reports (id, created_at, data) VALUES (?, ?, ?)",
            (report_id, created_at, json.dumps(data))
        )
        conn.commit()


def get_report(report_id: str):
    """Return report dict or None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT data FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
    if row:
        return json.loads(row["data"])
    return None


def save_analysis(entry: dict):
    """Append one analysis result to history."""
    with _conn() as conn:
        conn.execute("""
            INSERT INTO analysis_history
                (timestamp, device, confidence, repairability, repairability_class,
                 damage, age, location, cracks, rust, broken, cost_min, cost_max)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get("timestamp"),
            entry.get("device"),
            entry.get("confidence"),
            entry.get("repairability"),
            entry.get("repairability_class"),
            entry.get("damage"),
            entry.get("age"),
            entry.get("location"),
            entry.get("cracks"),
            entry.get("rust"),
            entry.get("broken"),
            entry.get("cost_min"),
            entry.get("cost_max"),
        ))
        conn.commit()


def get_history(limit: int = 100) -> list:
    """Return up to `limit` most recent analysis entries as list of dicts."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
