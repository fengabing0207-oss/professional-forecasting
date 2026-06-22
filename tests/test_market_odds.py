import numpy as np
import pandas as pd

from market.odds import (
    american_to_implied_probability,
    decimal_to_implied_probability,
    direct_probability,
    market_probabilities_from_odds,
    no_vig_normalize,
)


def test_decimal_odds_no_vig_conversion():
    raw = [decimal_to_implied_probability(x) for x in [2.00, 3.50, 4.00]]
    probs = no_vig_normalize(raw)
    expected = np.array(raw) / sum(raw)
    assert np.allclose(probs, expected)
    assert np.isclose(sum(probs), 1.0)


def test_american_odds_conversion():
    assert np.isclose(american_to_implied_probability(130), 100 / 230)
    assert np.isclose(american_to_implied_probability(-150), 150 / 250)


def test_direct_probability_passthrough():
    assert direct_probability(0.42) == 0.42


def test_market_probability_table_normalizes_group():
    odds = pd.DataFrame({
        "question_id": ["q_home", "q_draw", "q_away"],
        "market_id": ["m1", "m1", "m1"],
        "outcome_key": ["home", "draw", "away"],
        "odds_format": ["decimal", "decimal", "decimal"],
        "odds_value": [2.00, 3.50, 4.00],
        "direct_probability": [None, None, None],
        "bookmaker": ["manual", "manual", "manual"],
        "retrieved_at": ["", "", ""],
        "notes": ["", "", ""],
    })
    out = market_probabilities_from_odds(odds)
    assert np.isclose(out["p_market"].sum(), 1.0)
    assert set(out["question_id"]) == {"q_home", "q_draw", "q_away"}
