from webapp.app import create_app
from webapp.db import connect, init_db
from webapp.history import (
    create_session,
    latest_prediction_snapshot,
    save_question_snapshot,
)


def _client_with_db(monkeypatch, tmp_path):
    db_path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("PROB_CUP_DB_PATH", str(db_path))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), db_path


def _question_csv():
    return (
        "question_id,match_id,match_date,home_team,away_team,raw_question,event_type,selection,threshold,player,p_manual,manual_weight,parser_confidence,status,notes\n"
        "q1,m1,2026-06-24,England,Ghana,At halftime will the match be tied?,halftime_draw,draw_halftime,,,,,0.91,parsed,\n"
        "q2,m1,2026-06-24,England,Ghana,Will Ghana commit more fouls than England?,fouls_more_than_opponent,Ghana,,,,,0.82,parsed,\n"
    )


def test_live_route_get_without_question_snapshot_links_to_import(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")

    response = client.get(f"/sessions/{session_id}/live")

    assert response.status_code == 200
    assert b"No Questions Yet" in response.data
    assert b"Import Questions" in response.data


def test_live_route_get_with_question_snapshot_renders_cards(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    response = client.get(f"/sessions/{session_id}/live")

    assert response.status_code == 200
    assert b"Live Prediction Mode" in response.data
    assert b"Question 1 / 2" in response.data
    assert b"halftime_draw" in response.data
    assert b"market/manual-only" in response.data


def test_live_route_post_saves_prediction_snapshot_and_submission_sheet(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    response = client.post(
        f"/sessions/{session_id}/live",
        data={
            "favorite_team": "England",
            "expected_match_script": "Ghana underdog defending more",
            "tournament_context": "must-win",
            "player_context": "",
            "user_notes": "",
            "final_probability_percent_0": "51",
            "final_probability_percent_1": "60",
        },
    )

    assert response.status_code == 200
    assert b"Manual Submission Sheet" in response.data
    assert b"51" in response.data
    with connect(str(db_path)) as conn:
        snapshot = latest_prediction_snapshot(conn, session_id)
    assert snapshot is not None
    assert "0.51" in snapshot["csv_text"]
    assert "missing_probability" not in snapshot["csv_text"]


def test_live_route_post_rejects_ambiguous_decimal_percent(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    response = client.post(
        f"/sessions/{session_id}/live",
        data={"final_probability_percent_0": "0.51", "final_probability_percent_1": ""},
    )

    assert response.status_code == 200
    assert b"enter 51" in response.data
    with connect(str(db_path)) as conn:
        snapshot = latest_prediction_snapshot(conn, session_id)
    assert snapshot is None
