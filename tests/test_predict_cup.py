import numpy as np
import pandas as pd

from cup.predict_cup import build_predictions


def base_questions():
    return pd.DataFrame({
        "question_id": ["q1", "q2"],
        "match_id": ["m1", "m1"],
        "match_date": ["2026-06-22", "2026-06-22"],
        "home_team": ["Home", "Home"],
        "away_team": ["Away", "Away"],
        "raw_question": ["Will home win?", "Will there be 10 corners?"],
        "event_type": ["home_win", "corners_threshold"],
        "selection": ["home", ""],
        "threshold": ["", 9.5],
        "player": ["", ""],
        "p_manual": ["", ""],
        "manual_weight": ["", ""],
        "notes": ["", ""],
    })


def test_unsupported_event_falls_back_to_market_and_drops_model():
    odds = pd.DataFrame({
        "question_id": ["q2"],
        "market_id": [""],
        "outcome_key": ["yes"],
        "odds_format": ["direct"],
        "odds_value": [""],
        "direct_probability": [0.40],
        "bookmaker": ["manual"],
        "retrieved_at": [""],
        "notes": [""],
    })
    model = pd.DataFrame({
        "question_id": ["q1", "q2"],
        "p_model": [0.50, 0.95],
        "model_family": ["dixon_coles", "dixon_coles"],
        "notes": ["", ""],
    })
    out = build_predictions(base_questions(), odds, model)
    q2 = out[out["question_id"] == "q2"].iloc[0]
    assert q2["status"] == "unsupported_market_only"
    assert np.isclose(q2["p_final"], 0.40)
    assert pd.isna(q2["p_model"])


def test_supported_event_blends_market_and_model():
    odds = pd.DataFrame({
        "question_id": ["q1"],
        "market_id": [""],
        "outcome_key": ["yes"],
        "odds_format": ["direct"],
        "odds_value": [""],
        "direct_probability": [0.60],
        "bookmaker": ["manual"],
        "retrieved_at": [""],
        "notes": [""],
    })
    model = pd.DataFrame({
        "question_id": ["q1"],
        "p_model": [0.40],
        "model_family": ["dixon_coles"],
        "notes": [""],
    })
    out = build_predictions(base_questions().iloc[[0]], odds, model)
    q1 = out.iloc[0]
    assert q1["status"] == "model_and_market"
    assert np.isclose(q1["p_final"], 0.54)


def test_halftime_draw_with_manual_probability_is_manual_only():
    questions = base_questions().iloc[[0]].copy()
    questions.loc[:, "question_id"] = ["ht_manual"]
    questions.loc[:, "event_type"] = ["halftime_draw"]
    questions.loc[:, "selection"] = ["draw_halftime"]
    questions.loc[:, "p_manual"] = [0.44]
    odds = pd.DataFrame(columns=[
        "question_id", "market_id", "outcome_key", "odds_format",
        "odds_value", "direct_probability", "bookmaker", "retrieved_at", "notes",
    ])
    model = pd.DataFrame({
        "question_id": ["ht_manual"],
        "p_model": [0.95],
        "model_family": ["dixon_coles"],
        "notes": ["should not be used for halftime"],
    })
    out = build_predictions(questions, odds, model)
    row = out.iloc[0]
    assert row["status"] == "manual_only"
    assert np.isclose(row["p_final"], 0.44)
    assert pd.isna(row["p_model"])


def test_halftime_draw_with_market_probability_is_market_only_equivalent():
    questions = base_questions().iloc[[0]].copy()
    questions.loc[:, "question_id"] = ["ht_market"]
    questions.loc[:, "event_type"] = ["halftime_draw"]
    questions.loc[:, "selection"] = ["draw_halftime"]
    odds = pd.DataFrame({
        "question_id": ["ht_market"],
        "market_id": [""],
        "outcome_key": ["yes"],
        "odds_format": ["direct"],
        "odds_value": [""],
        "direct_probability": [0.38],
        "bookmaker": ["dummy"],
        "retrieved_at": [""],
        "notes": ["dummy"],
    })
    model = pd.DataFrame({
        "question_id": ["ht_market"],
        "p_model": [0.95],
        "model_family": ["dixon_coles"],
        "notes": ["should not be used for halftime"],
    })
    out = build_predictions(questions, odds, model)
    row = out.iloc[0]
    assert row["status"] == "unsupported_market_only"
    assert np.isclose(row["p_final"], 0.38)
    assert pd.isna(row["p_model"])


def test_halftime_event_without_probability_is_missing_probability():
    questions = base_questions().iloc[[0]].copy()
    questions.loc[:, "question_id"] = ["ht_missing"]
    questions.loc[:, "event_type"] = ["halftime_home_win"]
    questions.loc[:, "selection"] = ["home_halftime"]
    odds = pd.DataFrame(columns=[
        "question_id", "market_id", "outcome_key", "odds_format",
        "odds_value", "direct_probability", "bookmaker", "retrieved_at", "notes",
    ])
    model = pd.DataFrame({
        "question_id": ["ht_missing"],
        "p_model": [0.95],
        "model_family": ["dixon_coles"],
        "notes": ["should not be used for halftime"],
    })
    out = build_predictions(questions, odds, model)
    row = out.iloc[0]
    assert row["status"] == "missing_probability"
    assert pd.isna(row["p_final"])
    assert pd.isna(row["p_model"])
