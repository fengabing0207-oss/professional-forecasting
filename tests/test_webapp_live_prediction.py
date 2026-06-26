import re

from webapp.app import create_app
from webapp.db import connect, init_db
from webapp.history import (
    create_session,
    latest_assistant_snapshot,
    latest_context_snapshot,
    latest_prediction_snapshot,
    save_assistant_snapshot,
    save_context_snapshot,
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


def _offsides_question_csv():
    return (
        "question_id,match_id,match_date,home_team,away_team,raw_question,event_type,selection,threshold,player,p_manual,manual_weight,parser_confidence,status,notes\n"
        "q1,m1,2026-06-24,England,Ghana,Will Ghana be caught offside 2 or more times?,offsides_threshold,Ghana,2,,,,0.91,parsed,\n"
    )


def _input_tag(body: str, input_id: str) -> str:
    match = re.search(rf"<input\b(?=[^>]*\sid=\"{re.escape(input_id)}\")[^>]*>", body, re.S)
    assert match is not None
    return match.group(0)


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
    assert b'id="final_probability_percent_0"' in response.data
    assert b'id="final_probability_slider_0"' in response.data
    assert b"syncFinalProbabilityControls" in response.data
    assert b"Apply assistant suggestions to all blank finals" in response.data
    assert b"Clear all finals" in response.data
    assert b"Use suggested" in response.data
    assert b"Clear final" in response.data
    body = response.data.decode("utf-8")
    first_input = _input_tag(body, "final_probability_percent_0")
    second_input = _input_tag(body, "final_probability_percent_1")
    assert 'value=""' in first_input
    assert 'value=""' in second_input
    assert 'value="51"' not in first_input
    assert 'value="51"' not in second_input
    assert '<form method="post" autocomplete="off">' in body
    assert 'autocomplete="off"' in body
    assert 'autocomplete="off"' in first_input
    assert 'data-server-final=""' in first_input
    assert 'data-final-state="blank"' in first_input
    assert 'id="apply_suggestions_to_blank"' in body
    assert 'id="clear_all_finals"' in body
    assert 'class="use-suggested-final"' in body
    assert 'class="clear-final"' in body
    assert "applySuggested(input, slider" in body
    assert "clearFinal(input, slider)" in body


def test_suggested_probability_does_not_become_final_on_get(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _offsides_question_csv(), source="test")

    response = client.get(f"/sessions/{session_id}/live")

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    final_input = _input_tag(body, "final_probability_percent_0")
    slider_input = _input_tag(body, "final_probability_slider_0")
    assert "50%" in body
    assert 'value=""' in final_input
    assert 'data-server-final=""' in final_input
    assert 'data-suggested="50"' in slider_input
    assert 'value="50"' in slider_input
    assert 'class="use-suggested-final"' in body
    assert 'data-suggested="50"' in body


def test_live_route_get_prefills_latest_context_snapshot(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")
        save_context_snapshot(
            conn,
            session_id,
            '{"favorite_team": "England", "expected_match_script": "open late", "tournament_context": "must-win", "player_context": "rotation risk", "user_notes": "watch cards"}',
            source="test",
        )

    response = client.get(f"/sessions/{session_id}/live")

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert 'value="England"' in body
    assert 'value="open late"' in body
    assert "Prefilled from latest saved context snapshot" in body


def test_live_route_post_saves_context_assistant_prediction_and_submission_sheet(monkeypatch, tmp_path):
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
    assert b"Audit Summary" in response.data
    assert b"Q1 51% - At halftime will the match be tied?" in response.data
    assert b"Q2 60% - Will Ghana commit more fouls than England?" in response.data
    with connect(str(db_path)) as conn:
        prediction = latest_prediction_snapshot(conn, session_id)
        context = latest_context_snapshot(conn, session_id)
        assistant = latest_assistant_snapshot(conn, session_id)
    assert prediction is not None
    assert context is not None
    assert assistant is not None
    assert '"favorite_team": "England"' in context["context_json"]
    assert '"final_probability_percent": "60"' in assistant["assistant_json"]
    assert "0.51" in prediction["csv_text"]
    assert "missing_probability" not in prediction["csv_text"]


def test_scoring_and_calibration_surface_saved_live_snapshots(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    client.post(
        f"/sessions/{session_id}/live",
        data={
            "favorite_team": "England",
            "expected_match_script": "",
            "tournament_context": "",
            "player_context": "",
            "user_notes": "",
            "final_probability_percent_0": "51",
            "final_probability_percent_1": "",
        },
    )

    scoring_response = client.get(f"/sessions/{session_id}/scoring")
    assert scoring_response.status_code == 200
    assert b"Latest Saved Final Probabilities" in scoring_response.data
    assert b"assistant suggestions/finals" in scoring_response.data

    calibration_response = client.get("/calibration")
    assert calibration_response.status_code == 200
    assert b"Saved live sessions are available" in calibration_response.data
    assert b"M1" in calibration_response.data


def test_blank_final_probabilities_are_excluded_from_manual_odds(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    response = client.post(
        f"/sessions/{session_id}/live",
        data={
            "favorite_team": "England",
            "expected_match_script": "",
            "tournament_context": "",
            "player_context": "",
            "user_notes": "",
            "final_probability_percent_0": "51",
            "final_probability_percent_1": "",
        },
    )

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    submission_match = re.search(r"<textarea rows=\"8\" readonly>(.*?)</textarea>", body, re.S)
    assert submission_match is not None
    submission_text = submission_match.group(1)
    assert "Q1 51% - At halftime will the match be tied?" in submission_text
    assert "Q2" not in submission_text
    with connect(str(db_path)) as conn:
        prediction = latest_prediction_snapshot(conn, session_id)
        assistant = latest_assistant_snapshot(conn, session_id)
    assert prediction is not None
    assert "q1" in prediction["csv_text"]
    assert "q2" in prediction["csv_text"]
    assert "0.51" in prediction["csv_text"]
    assert "missing_probability" in prediction["csv_text"]
    assert '"final_probability_percent": ""' in assistant["assistant_json"]


def test_latest_context_snapshot_viewer_route(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_context_snapshot(conn, session_id, '{"favorite_team": "England"}', source="test")

    response = client.get(f"/sessions/{session_id}/snapshots/context/latest")

    assert response.status_code == 200
    assert b"Latest context snapshot" in response.data
    assert b"favorite_team" in response.data


def test_latest_assistant_snapshot_viewer_route(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_assistant_snapshot(conn, session_id, '[{"question_id": "q1"}]', source="test")

    response = client.get(f"/sessions/{session_id}/snapshots/assistant/latest")

    assert response.status_code == 200
    assert b"Latest assistant snapshot" in response.data
    assert b"question_id" in response.data


def test_context_snapshot_viewer_route_by_id(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        snapshot_id = save_context_snapshot(conn, session_id, '{"favorite_team": "England"}', source="test")

    response = client.get(f"/sessions/{session_id}/snapshots/context/{snapshot_id}")

    assert response.status_code == 200
    assert b"Context snapshot" in response.data
    assert b"favorite_team" in response.data


def test_invalid_snapshot_kind_returns_404(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")

    response = client.get(f"/sessions/{session_id}/snapshots/badkind/latest")

    assert response.status_code == 404


def test_missing_snapshot_id_returns_404(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")

    response = client.get(f"/sessions/{session_id}/snapshots/context/999")

    assert response.status_code == 404


def test_live_route_post_rejects_ambiguous_decimal_percent(monkeypatch, tmp_path):
    client, db_path = _client_with_db(monkeypatch, tmp_path)
    with connect(str(db_path)) as conn:
        init_db(conn)
        session_id = create_session(conn, match_id="M1", home_team="England", away_team="Ghana")
        save_question_snapshot(conn, session_id, _question_csv(), source="test")

    response = client.post(
        f"/sessions/{session_id}/live",
        data={"final_probability_percent_0": "0.51", "final_probability_percent_1": "60"},
    )

    assert response.status_code == 200
    assert b"enter 51" in response.data
    assert b'value="60"' in response.data
    with connect(str(db_path)) as conn:
        prediction = latest_prediction_snapshot(conn, session_id)
        context = latest_context_snapshot(conn, session_id)
        assistant = latest_assistant_snapshot(conn, session_id)
    assert prediction is None
    assert context is None
    assert assistant is None
