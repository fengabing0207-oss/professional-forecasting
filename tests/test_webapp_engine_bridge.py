from webapp.engine_bridge import (
    flag_prediction_risks,
    manual_probability_rows_to_odds_csv,
    normalize_manual_probability_percent,
    parse_raw_questions_to_csv,
    question_csv_to_manual_probability_rows,
    run_prediction_csv,
    summarize_predictions_csv,
    summarize_scoring_csv,
)


def test_summarize_prediction_status_counts_works():
    csv_text = (
        "question_id,event_type,p_final,status\n"
        "q1,team_win,0.60,model_and_market\n"
        "q2,corners_threshold,,missing_probability\n"
    )
    summary = summarize_predictions_csv(csv_text)
    assert summary["status_counts"]["model_and_market"] == 1
    assert summary["status_counts"]["missing_probability"] == 1
    assert summary["missing_probability"] == 1


def test_risk_flags_catch_missing_probability():
    csv_text = "question_id,event_type,p_final,status\nq1,corners_threshold,,missing_probability\n"
    risks = flag_prediction_risks(csv_text)
    kinds = {risk["kind"] for risk in risks}
    assert "missing_probability" in kinds
    assert "blank_p_final" in kinds


def test_risk_flags_catch_extreme_high_low_probability():
    csv_text = (
        "question_id,event_type,p_final,status\n"
        "q1,corners_threshold,0.85,unsupported_market_only\n"
        "q2,corners_threshold,0.15,unsupported_market_only\n"
    )
    risks = flag_prediction_risks(csv_text)
    kinds = {risk["kind"] for risk in risks}
    assert "extreme_high" in kinds
    assert "extreme_low" in kinds


def test_scoring_summary_handles_missing_crowd_brier():
    csv_text = (
        "question_id,event_type,status,raw_question,user_brier,crowd_brier,rbp,actual_result\n"
        "q1,team_win,model_only,Will home win?,0.09,, ,1\n"
    )
    summary = summarize_scoring_csv(csv_text)
    assert summary["rows"] == 1
    assert summary["missing_crowd_brier"] == 1
    assert summary["average_user_brier"] == 0.09


def test_engine_bridge_parses_raw_questions_without_inventing_probabilities():
    out = parse_raw_questions_to_csv(
        "Q1: Will Norway win the match?\nQ2: Will the match get weird?\n",
        "Norway",
        "Senegal",
        "NOR_SEN",
    )
    assert "team_win" in out
    assert "needs_review" in out
    assert "p_market" not in out
    assert "p_final" not in out


def _manual_workbench_question_csv():
    return (
        "question_id,match_id,match_date,home_team,away_team,raw_question,event_type,selection,threshold,player,p_manual,manual_weight,parser_confidence,status,notes\n"
        "q1,m1,2026-06-22,Home,Away,At halftime will both teams have a shot on target?,halftime_draw,yes,,,,,0.91,needs_review,market/manual-only\n"
    )


def test_manual_probability_blank_remains_blank():
    rows = question_csv_to_manual_probability_rows(_manual_workbench_question_csv())
    assert rows[0]["manual_probability_percent"] == ""
    assert rows[0]["p_manual"] == ""
    odds_csv = manual_probability_rows_to_odds_csv(rows, match_id="m1")
    assert "question_id,market_id,outcome_key,odds_format,odds_value,direct_probability,bookmaker,retrieved_at,notes" in odds_csv
    assert "q1," not in odds_csv


def test_manual_probability_percent_normalizes_values():
    assert normalize_manual_probability_percent("51") == 0.51
    assert normalize_manual_probability_percent("100") == 1.0
    assert normalize_manual_probability_percent("0") == 0.0


def test_manual_probability_percent_rejects_invalid_values():
    try:
        normalize_manual_probability_percent("101")
        assert False
    except ValueError as exc:
        assert "between 0 and 100" in str(exc)
    try:
        normalize_manual_probability_percent("-1")
        assert False
    except ValueError as exc:
        assert "between 0 and 100" in str(exc)
    try:
        normalize_manual_probability_percent("0.51")
        assert False
    except ValueError as exc:
        assert "percent mode" in str(exc)


def test_parser_confidence_is_not_treated_as_manual_probability():
    rows = question_csv_to_manual_probability_rows(_manual_workbench_question_csv())
    assert rows[0]["p_manual"] == ""
    assert rows[0]["manual_probability_percent"] == ""


def test_generated_manual_odds_csv_has_correct_headers_and_values():
    rows = question_csv_to_manual_probability_rows(_manual_workbench_question_csv())
    rows[0]["manual_probability_percent"] = "51"
    odds_csv = manual_probability_rows_to_odds_csv(rows, match_id="m1")
    assert odds_csv.splitlines()[0] == (
        "question_id,market_id,outcome_key,odds_format,odds_value,direct_probability,bookmaker,retrieved_at,notes"
    )
    assert "q1,m1_q1,yes,direct_probability,,0.51,manual,,manual probability workbench" in odds_csv


def test_generated_manual_odds_can_feed_prediction_engine_for_market_only_question():
    rows = question_csv_to_manual_probability_rows(_manual_workbench_question_csv())
    rows[0]["manual_probability_percent"] = "58"
    odds_csv = manual_probability_rows_to_odds_csv(rows, match_id="m1")
    prediction_csv = run_prediction_csv(_manual_workbench_question_csv(), odds_csv)
    assert "missing_probability" not in prediction_csv
    assert "0.58" in prediction_csv
    assert "unsupported_market_only" in prediction_csv
