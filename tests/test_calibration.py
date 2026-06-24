import pytest

from webapp.calibration import (
    compute_brier_columns,
    find_largest_crowd_deviations,
    find_largest_rbp_losses,
    find_largest_rbp_wins,
    generate_guardrail_suggestions,
    load_settled_history_csv,
    summarize_by_event_type,
    summarize_by_probability_bucket,
    summarize_settled_performance,
)


def _settled_csv():
    return """session_id,match_id,match_date,home_team,away_team,question_id,raw_question,event_type,selection,user_prob,crowd_prob,actual_result,platform_rbp,notes
s1,NOR_SEN,2026-06-22,Norway,Senegal,q1,Will Norway win?,team_win,Norway,0.45,0.70,1,-8.0,dummy
s1,NOR_SEN,2026-06-22,Norway,Senegal,q2,Will Senegal commit more fouls?,fouls_more_than_opponent,Senegal,0.70,0.52,0,-12.0,dummy
s1,NOR_SEN,2026-06-22,Norway,Senegal,q3,Will Sadio Mane have a second half shot on target?,player_second_half_shots_on_target,Sadio Mane,0.60,0.48,0,-6.0,dummy
s1,NOR_SEN,2026-06-22,Norway,Senegal,q4,Will there be 9+ total corners?,corners_threshold,total,0.30,0.55,1,-4.0,dummy
s1,NOR_SEN,2026-06-22,Norway,Senegal,q5,Will both teams score and total goals over 2.5?,both_teams_score_and_total_goals_over,yes,0.55,0.50,0,-3.0,dummy
s1,NOR_SEN,2026-06-22,Norway,Senegal,q6,Will Norway record 6+ shots on target?,team_shots_on_target_threshold,Norway,0.35,,0,5.0,missing crowd
s1,NOR_SEN,2026-06-22,Norway,Senegal,q7,Will there be a penalty or red card?,penalty_or_red_card,match,0.15,0.20,0,4.0,dummy
"""


def _df():
    return load_settled_history_csv(_settled_csv())


def test_brier_columns_compute_correctly():
    scored = compute_brier_columns(_df())
    q1 = scored[scored["question_id"] == "q1"].iloc[0]
    assert q1["user_brier"] == pytest.approx((0.45 - 1) ** 2)
    assert q1["crowd_brier"] == pytest.approx((0.70 - 1) ** 2)
    assert q1["brier_edge"] == pytest.approx(q1["crowd_brier"] - q1["user_brier"])


def test_total_platform_rbp_summary_and_missing_crowd_work():
    summary = summarize_settled_performance(_df())
    assert summary["total_questions"] == 7
    assert summary["total_platform_rbp"] == pytest.approx(-24.0)
    assert summary["average_platform_rbp"] == pytest.approx(-24.0 / 7)
    assert summary["mean_crowd_brier"] is not None


def test_beat_and_below_crowd_counts_work():
    summary = summarize_settled_performance(_df())
    assert summary["beat_crowd_count"] == 2
    assert summary["below_crowd_count"] == 5


def test_event_type_summary_works():
    by_event = summarize_by_event_type(_df())
    fouls = by_event[by_event["event_type"] == "fouls_more_than_opponent"].iloc[0]
    assert fouls["questions"] == 1
    assert fouls["total_platform_rbp"] == pytest.approx(-12.0)
    assert fouls["mean_user_brier"] == pytest.approx(0.49)


def test_probability_bucket_summary_works():
    by_bucket = summarize_by_probability_bucket(_df())
    counts = dict(zip(by_bucket["probability_bucket"], by_bucket["questions"]))
    assert counts["0-20%"] == 1
    assert counts["20-40%"] == 2
    assert counts["40-60%"] == 2
    assert counts["60-80%"] == 2
    assert counts["80-100%"] == 0


def test_largest_losses_and_wins_work():
    losses = find_largest_rbp_losses(_df(), n=2)
    wins = find_largest_rbp_wins(_df(), n=2)
    assert list(losses["question_id"]) == ["q2", "q1"]
    assert list(wins["question_id"]) == ["q6", "q7"]


def test_crowd_deviation_works():
    deviations = find_largest_crowd_deviations(_df(), n=1)
    assert deviations.iloc[0]["question_id"] in {"q1", "q4"}
    assert deviations.iloc[0]["abs_user_crowd_deviation"] == pytest.approx(0.25)


def test_guardrail_suggestions_catch_fouls_overconfidence():
    suggestions = generate_guardrail_suggestions(_df())
    assert any("fouls" in suggestion.lower() and "60%" in suggestion for suggestion in suggestions)


def test_guardrail_suggestions_catch_player_second_half_sot_overconfidence():
    suggestions = generate_guardrail_suggestions(_df())
    assert any("second-half shot-on-target" in suggestion.lower() for suggestion in suggestions)


def test_missing_crowd_prob_does_not_crash_summary():
    df = _df()
    assert df["crowd_prob"].isna().sum() == 1
    summary = summarize_settled_performance(df)
    assert summary["average_abs_user_crowd_deviation"] is not None
