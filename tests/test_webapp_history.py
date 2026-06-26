import tempfile

from webapp.db import connect, init_db
from webapp.history import (
    create_session,
    latest_assistant_snapshot,
    latest_context_snapshot,
    latest_prediction_snapshot,
    latest_question_snapshot,
    latest_scoring_snapshot,
    save_assistant_snapshot,
    save_context_snapshot,
    save_prediction_snapshot,
    save_question_snapshot,
    save_scoring_snapshot,
    snapshot_history,
)


def _conn():
    fh = tempfile.NamedTemporaryFile(suffix=".sqlite3")
    conn = connect(fh.name)
    init_db(conn)
    return fh, conn


def test_question_snapshot_save_load_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        save_question_snapshot(conn, session_id, "question_id,event_type\nq1,team_win\n")
        row = latest_question_snapshot(conn, session_id)
        assert "team_win" in row["csv_text"]
    finally:
        fh.close()


def test_prediction_snapshot_save_load_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        save_prediction_snapshot(conn, session_id, "question_id,p_final\nq1,0.6\n", {"rows": 1})
        row = latest_prediction_snapshot(conn, session_id)
        assert "p_final" in row["csv_text"]
        assert '"rows": 1' in row["summary_json"]
    finally:
        fh.close()


def test_scoring_snapshot_save_load_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        save_scoring_snapshot(conn, session_id, "question_id,user_brier\nq1,0.1\n", {"rows": 1})
        row = latest_scoring_snapshot(conn, session_id)
        assert "user_brier" in row["csv_text"]
    finally:
        fh.close()


def test_context_snapshot_save_load_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        save_context_snapshot(conn, session_id, '{"favorite_team": "H"}', source="test")
        row = latest_context_snapshot(conn, session_id)
        assert '"favorite_team": "H"' in row["context_json"]
        assert row["source"] == "test"
    finally:
        fh.close()


def test_assistant_snapshot_save_load_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        save_assistant_snapshot(conn, session_id, '[{"question_id": "q1"}]', source="test")
        row = latest_assistant_snapshot(conn, session_id)
        assert '"question_id": "q1"' in row["assistant_json"]
        assert row["source"] == "test"
    finally:
        fh.close()


def test_history_ordering_works():
    fh, conn = _conn()
    try:
        session_id = create_session(conn, match_id="M", home_team="H", away_team="A")
        first = save_question_snapshot(conn, session_id, "question_id\nq1\n")
        second = save_question_snapshot(conn, session_id, "question_id\nq2\n")
        context = save_context_snapshot(conn, session_id, '{"favorite_team": "H"}', source="test")
        assistant = save_assistant_snapshot(conn, session_id, '[{"question_id": "q1"}]', source="test")
        hist = snapshot_history(conn, session_id)
        assert hist["questions"][0]["id"] == second
        assert hist["questions"][1]["id"] == first
        assert hist["context"][0]["id"] == context
        assert hist["assistant"][0]["id"] == assistant
    finally:
        fh.close()
