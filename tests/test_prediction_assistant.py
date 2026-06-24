import subprocess

from webapp.engine_bridge import run_prediction_csv
from webapp.prediction_assistant import (
    assistant_rows_to_manual_odds_csv,
    normalize_event_type,
    normalize_final_probability_percent,
    suggest_probability_for_question,
)


def test_penalty_red_card_uses_context_adjusted_prior():
    out = suggest_probability_for_question(
        {"question_id": "q1", "event_type": "penalty_or_red_card", "raw_question": "Penalty or red card?", "selection": "yes"},
        match_context={"tournament_context": "knockout high-intensity"},
    )
    assert out["suggested_probability"] > 0.28
    assert out["suggested_range_low"] >= 0.29
    assert "increased" in out["reasoning"]


def test_fouls_underdog_adjusts_and_high_final_flags():
    out = suggest_probability_for_question(
        {
            "question_id": "q1",
            "event_type": "fouls_more_than_opponent",
            "raw_question": "Will Ghana commit more fouls?",
            "selection": "Ghana",
            "final_probability_percent": "61",
        },
        match_context={"favorite_team": "England"},
    )
    assert out["suggested_probability"] > 0.55
    assert "fouls_above_60" in out["risk_flags"]


def test_player_second_half_sot_context_and_high_final_flags():
    out = suggest_probability_for_question(
        {
            "question_id": "q1",
            "event_type": "player_second_half_shot_on_target",
            "raw_question": "Will the striker have a second half shot on target?",
            "player": "Striker",
            "final_probability_percent": "56",
        },
        match_context={"player_context": "primary attacker likely full match"},
    )
    assert out["suggested_probability"] > 0.48
    assert "player_2h_sot_above_55" in out["risk_flags"]


def test_player_second_half_sot_plural_alias_uses_same_handling():
    out = suggest_probability_for_question(
        {
            "question_id": "q1",
            "event_type": "player_second_half_shots_on_target",
            "raw_question": "Will the striker have a second half shot on target?",
            "player": "Striker",
            "final_probability_percent": "56",
        },
        match_context={"player_context": "primary attacker likely full match"},
    )
    assert out["event_type"] == "player_second_half_shots_on_target"
    assert out["normalized_event_type"] == "player_second_half_shot_on_target"
    assert out["suggested_probability"] > 0.48
    assert "high_variance_prop" in out["risk_flags"]
    assert "player_2h_sot_above_55" in out["risk_flags"]


def test_event_type_aliases_and_compound_raw_text_normalize():
    assert normalize_event_type("player_second_half_sot") == "player_second_half_shot_on_target"
    assert normalize_event_type("player_shots_on_target") == "player_shot_on_target"
    assert normalize_event_type("team_shots_on_target_threshold") == "team_shots_on_target_threshold"
    assert normalize_event_type(
        "unsupported_market_only",
        "Will both teams score and will there be 3+ total goals?",
    ) == "both_teams_score_and_total_goals_over"


def test_compound_btts_total_flags_compound_condition():
    out = suggest_probability_for_question({
        "question_id": "q1",
        "event_type": "both_teams_score_and_total_goals_over",
        "raw_question": "Will both teams score and there be 3+ total goals?",
        "selection": "yes",
    })
    assert out["suggested_probability"] == 0.44
    assert "compound_condition" in out["risk_flags"]


def test_underdog_five_plus_corners_stays_below_45():
    out = suggest_probability_for_question(
        {
            "question_id": "q1",
            "event_type": "team_corners_threshold",
            "raw_question": "Will Ghana have 5+ corners?",
            "selection": "Ghana",
            "threshold": "5",
        },
        match_context={"favorite_team": "England"},
    )
    assert out["suggested_probability"] < 0.45


def test_team_win_without_context_requires_user_judgment():
    out = suggest_probability_for_question({
        "question_id": "q1",
        "event_type": "team_win",
        "raw_question": "Will England win?",
        "selection": "England",
        "parser_confidence": "0.99",
    })
    assert out["suggested_probability"] is None
    assert out["suggested_probability_percent"] == ""
    assert "needs_manual" in out["risk_flags"]


def test_final_probability_normalization_and_rejection():
    assert normalize_final_probability_percent("51") == 0.51
    assert normalize_final_probability_percent("") is None
    try:
        normalize_final_probability_percent("0.51")
        assert False
    except ValueError as exc:
        assert "enter 51" in str(exc)


def test_assistant_does_not_invent_50_or_use_parser_confidence():
    out = suggest_probability_for_question({
        "question_id": "q1",
        "event_type": "some_unknown_event",
        "raw_question": "Will something odd happen?",
        "parser_confidence": "0.50",
    })
    assert out["suggested_probability"] is None
    assert out["suggested_probability_percent"] == ""
    assert out["normalized_event_type"] == "some_unknown_event"


def test_generated_manual_odds_csv_feeds_prediction_engine():
    questions = (
        "question_id,match_id,match_date,home_team,away_team,raw_question,event_type,selection,threshold,player,p_manual,manual_weight,parser_confidence,status,notes\n"
        "q1,m1,2026-06-24,England,Ghana,At halftime will the match be tied?,halftime_draw,draw_halftime,,,,,0.91,parsed,\n"
    )
    odds = assistant_rows_to_manual_odds_csv(
        [{"question_id": "q1", "final_probability_percent": "51"}],
        match_id="m1",
    )
    predictions = run_prediction_csv(questions, odds)
    assert "0.51" in predictions
    assert "missing_probability" not in predictions


def test_match_script_exposure_warning_detected():
    rows = [
        {
            "question_id": "q1",
            "event_type": "team_win",
            "raw_question": "Will England win?",
            "selection": "England",
            "final_probability_percent": "65",
        },
        {
            "question_id": "q2",
            "event_type": "team_shots_on_target_threshold",
            "raw_question": "Will Ghana have 4+ shots on target?",
            "selection": "Ghana",
            "threshold": "4",
            "final_probability_percent": "55",
        },
    ]
    suggested = [suggest_probability_for_question(row, {"favorite_team": "England"}) for row in rows]
    assert any(row["exposure_warnings"] == [] for row in suggested)
    from webapp.prediction_assistant import detect_match_script_exposure

    assert detect_match_script_exposure(suggested)


def test_local_output_and_live_data_paths_are_not_tracked():
    for path in ("outputs", ".local", "data/cup/live"):
        out = subprocess.check_output(["git", "ls-files", path], text=True).strip()
        assert out == ""
