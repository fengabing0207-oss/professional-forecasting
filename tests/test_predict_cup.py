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
