from webapp.app import create_app
from webapp.db import connect, init_db
from webapp.history import (
    create_session,
    save_prediction_snapshot,
    save_question_snapshot,
    save_scoring_snapshot,
)


def _client_with_db(monkeypatch, tmp_path):
    db_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("PROB_CUP_DB_PATH", str(db_path))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), db_path


def _session_with_snapshots(db_path):
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(
            conn,
            match_id="NOR/SEN 2026",
            home_team="Norway",
            away_team="Senegal",
        )
        save_question_snapshot(conn, session_id, "question_id,event_type\nq1,team_win\n")
        save_prediction_snapshot(conn, session_id, "question_id,p_final\nq1,0.63\n", {"rows": 1})
        save_scoring_snapshot(conn, session_id, "question_id,user_brier\nq1,0.1369\n", {"rows": 1})
    return session_id


def test_latest_question_csv_download_returns_attachment(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    session_id = _session_with_snapshots(db_path)

    response = client.get(f"/sessions/{session_id}/download/questions/latest")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment" in response.headers["Content-Disposition"]
    assert "NOR_SEN_2026_questions_latest.csv" in response.headers["Content-Disposition"]
    assert b"team_win" in response.data


def test_latest_prediction_csv_download_returns_200(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    session_id = _session_with_snapshots(db_path)

    response = client.get(f"/sessions/{session_id}/download/predictions/latest")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"p_final" in response.data


def test_latest_scoring_csv_download_returns_200(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    session_id = _session_with_snapshots(db_path)

    response = client.get(f"/sessions/{session_id}/download/scoring/latest")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"user_brier" in response.data


def test_missing_latest_snapshot_returns_404(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M", home_team="Home", away_team="Away")

    response = client.get(f"/sessions/{session_id}/download/questions/latest")

    assert response.status_code == 404


def test_homepage_renders_latest_links_and_missing_labels(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    _session_with_snapshots(db_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        create_session(conn, match_id="EMPTY", home_team="Home", away_team="Away")

    response = client.get("/")

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert "/download/questions/latest" in body
    assert "/download/predictions/latest" in body
    assert "/download/scoring/latest" in body
    assert "No questions yet" in body
    assert "No predictions yet" in body
    assert "No scoring yet" in body
