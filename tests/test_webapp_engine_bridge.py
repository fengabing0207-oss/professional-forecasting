from webapp.engine_bridge import (
    flag_prediction_risks,
    parse_raw_questions_to_csv,
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
