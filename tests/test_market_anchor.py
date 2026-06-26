import pytest

from webapp.market_anchor import (
    implied_probability_from_odds,
    market_anchor_probability,
    normalize_probability_percent,
)


def test_probability_percent_normalization():
    assert normalize_probability_percent("51") == 0.51
    assert normalize_probability_percent("") is None
    with pytest.raises(ValueError, match="enter 51"):
        normalize_probability_percent("0.51")
    with pytest.raises(ValueError, match="between 0 and 100"):
        normalize_probability_percent("101")


def test_american_odds_implied_probability():
    assert implied_probability_from_odds("-160") == pytest.approx(160 / 260)
    assert implied_probability_from_odds("+220") == pytest.approx(100 / 320)


def test_decimal_odds_implied_probability():
    assert implied_probability_from_odds("1.62") == pytest.approx(1 / 1.62)
    assert implied_probability_from_odds("2.85") == pytest.approx(1 / 2.85)


def test_malformed_odds_raise_clear_error():
    with pytest.raises(ValueError, match="American or decimal odds"):
        implied_probability_from_odds("abc")
    with pytest.raises(ValueError, match="greater than 1"):
        implied_probability_from_odds("1.0")


def test_percent_takes_priority_over_odds():
    assert market_anchor_probability("47", "-160") == 0.47
