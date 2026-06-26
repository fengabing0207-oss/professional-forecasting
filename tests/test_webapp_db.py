import tempfile

from webapp.db import connect, init_db
from webapp.history import create_session, get_session


def test_database_initializes_all_required_tables():
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as fh:
        conn = connect(fh.name)
        init_db(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert {
            "sessions",
            "question_snapshots",
            "prediction_snapshots",
            "scoring_snapshots",
            "context_snapshots",
            "assistant_snapshots",
            "market_snapshots",
            "run_logs",
        }.issubset(names)


def test_create_session_works():
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as fh:
        conn = connect(fh.name)
        init_db(conn)
        session_id = create_session(
            conn,
            match_id="NOR_SEN",
            home_team="Norway",
            away_team="Senegal",
        )
        row = get_session(conn, session_id)
        assert row["match_id"] == "NOR_SEN"
        assert row["home_team"] == "Norway"
