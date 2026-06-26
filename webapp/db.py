"""SQLite storage for local Probability Cup UI history."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB_PATH = ".local/probability_cup_history.sqlite3"


def get_db_path() -> str:
    return os.environ.get("PROB_CUP_DB_PATH", DEFAULT_DB_PATH)


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            match_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            match_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS question_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            csv_text TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            csv_text TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS scoring_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            csv_text TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS context_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            context_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS assistant_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            assistant_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS run_logs (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL
        );
        """
    )
    conn.commit()


def query_all(conn: sqlite3.Connection, sql: str,
              params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, tuple(params)))


def query_one(conn: sqlite3.Connection, sql: str,
              params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, tuple(params)).fetchone()
