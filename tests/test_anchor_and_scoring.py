import numpy as np

from cup.scoring import brier_score, relative_brier_points
from market.anchor import blend_probability


def test_brier_score_calculation():
    assert np.isclose(brier_score(0.70, 1), 0.09)


def test_rbp_calculation():
    assert np.isclose(relative_brier_points(0.09, 0.16), 7.0)


def test_blending_market_and_model_defaults():
    result = blend_probability(p_market=0.60, p_model=0.40)
    assert result.status == "model_and_market"
    assert np.isclose(result.p_final, 0.54)


def test_blending_fallbacks_and_missing():
    assert blend_probability(p_market=0.60).status == "market_only"
    assert blend_probability(p_model=0.40).status == "model_only"
    assert blend_probability(p_manual=0.55).status == "manual_only"
    assert blend_probability().status == "missing_probability"


def test_unsupported_event_ignores_model_probability():
    result = blend_probability(
        p_market=0.35,
        p_model=0.90,
        model_supported=False,
    )
    assert result.status == "unsupported_market_only"
    assert result.p_final == 0.35
