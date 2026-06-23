"""History operations for the local Probability Cup UI."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import sqlite3

from webapp.db import query_all, query_one


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_session(conn: sqlite3.Connection, *, match_id: str, home_team: str,
                   away_team: str, match_date: str = "",
                   notes: str = "") -> int:
    if not match_id.strip() or not home_team.strip() or not away_team.strip():
        raise ValueError("match_id, home_team, and away_team are required")
    ts = now_iso()
    cur = conn.execute(
        """
        INSERT INTO sessions
            (match_id, home_team, away_team, match_date, created_at, updated_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (match_id.strip(), home_team.strip(), away_team.strip(),
         match_date.strip() or None, ts, ts, notes.strip() or None),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_session_timestamp(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso(), session_id))
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row:
    row = query_one(conn, "SELECT * FROM sessions WHERE id = ?", (session_id,))
    if row is None:
        raise ValueError(f"session not found: {session_id}")
    return row


def list_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = query_all(
        conn,
        """
        SELECT s.*,
               (SELECT MAX(created_at) FROM prediction_snapshots p WHERE p.session_id = s.id)
                   AS latest_prediction_at,
               (SELECT MAX(created_at) FROM scoring_snapshots sc WHERE sc.session_id = s.id)
                   AS latest_scoring_at
        FROM sessions s
        ORDER BY s.updated_at DESC, s.id DESC
        """,
    )
    return [dict(row) for row in rows]


def save_question_snapshot(conn: sqlite3.Connection, session_id: int,
                           csv_text: str, source: str = "manual") -> int:
    return _save_snapshot(conn, "question_snapshots", session_id, csv_text, source=source)


def save_prediction_snapshot(conn: sqlite3.Connection, session_id: int,
                             csv_text: str, summary: dict[str, Any]) -> int:
    return _save_snapshot(conn, "prediction_snapshots", session_id, csv_text, summary=summary)


def save_scoring_snapshot(conn: sqlite3.Connection, session_id: int,
                          csv_text: str, summary: dict[str, Any]) -> int:
    return _save_snapshot(conn, "scoring_snapshots", session_id, csv_text, summary=summary)


def _save_snapshot(conn: sqlite3.Connection, table: str, session_id: int,
                   csv_text: str, *, source: str | None = None,
                   summary: dict[str, Any] | None = None) -> int:
    get_session(conn, session_id)
    if not csv_text.strip():
        raise ValueError("csv_text is required")
    ts = now_iso()
    if table == "question_snapshots":
        cur = conn.execute(
            "INSERT INTO question_snapshots (session_id, created_at, source, csv_text) VALUES (?, ?, ?, ?)",
            (session_id, ts, source or "manual", csv_text),
        )
    elif table in {"prediction_snapshots", "scoring_snapshots"}:
        cur = conn.execute(
            f"INSERT INTO {table} (session_id, created_at, csv_text, summary_json) VALUES (?, ?, ?, ?)",
            (session_id, ts, csv_text, json.dumps(summary or {}, sort_keys=True)),
        )
    else:
        raise ValueError(f"unsupported snapshot table: {table}")
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (ts, session_id))
    conn.commit()
    return int(cur.lastrowid)


def latest_question_snapshot(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row | None:
    return _latest(conn, "question_snapshots", session_id)


def latest_prediction_snapshot(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row | None:
    return _latest(conn, "prediction_snapshots", session_id)


def latest_scoring_snapshot(conn: sqlite3.Connection, session_id: int) -> sqlite3.Row | None:
    return _latest(conn, "scoring_snapshots", session_id)


def _latest(conn: sqlite3.Connection, table: str, session_id: int) -> sqlite3.Row | None:
    return query_one(
        conn,
        f"SELECT * FROM {table} WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (session_id,),
    )


def snapshot_history(conn: sqlite3.Connection, session_id: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "questions": [dict(row) for row in query_all(
            conn,
            "SELECT * FROM question_snapshots WHERE session_id = ? ORDER BY created_at DESC, id DESC",
            (session_id,),
        )],
        "predictions": [dict(row) for row in query_all(
            conn,
            "SELECT * FROM prediction_snapshots WHERE session_id = ? ORDER BY created_at DESC, id DESC",
            (session_id,),
        )],
        "scoring": [dict(row) for row in query_all(
            conn,
            "SELECT * FROM scoring_snapshots WHERE session_id = ? ORDER BY created_at DESC, id DESC",
            (session_id,),
        )],
    }


def log_run(conn: sqlite3.Connection, level: str, message: str,
            session_id: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO run_logs (session_id, created_at, level, message) VALUES (?, ?, ?, ?)",
        (session_id, now_iso(), level, message),
    )
    conn.commit()
    return int(cur.lastrowid)

