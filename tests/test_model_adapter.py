import numpy as np

from cup.model_adapter import goal_event_probabilities_from_score_matrix


def toy_matrix():
    return np.array([
        [0.10, 0.05, 0.02],
        [0.15, 0.20, 0.08],
        [0.25, 0.10, 0.05],
    ])


def test_goal_adapter_computes_home_draw_away_from_toy_matrix():
    matrix = toy_matrix()
    home = goal_event_probabilities_from_score_matrix(matrix, "Home", "Away", "home_win", "", None)
    draw = goal_event_probabilities_from_score_matrix(matrix, "Home", "Away", "draw", "", None)
    away = goal_event_probabilities_from_score_matrix(matrix, "Home", "Away", "away_win", "", None)
    assert home.status == "model_supported"
    assert np.isclose(home.p_model, 0.50)
    assert np.isclose(draw.p_model, 0.35)
    assert np.isclose(away.p_model, 0.15)


def test_goal_adapter_computes_btts_and_over_under_from_toy_matrix():
    matrix = toy_matrix()
    btts = goal_event_probabilities_from_score_matrix(
        matrix, "Home", "Away", "both_teams_score_yes", "", None
    )
    over = goal_event_probabilities_from_score_matrix(
        matrix, "Home", "Away", "total_goals_over", "total", 2.5
    )
    under = goal_event_probabilities_from_score_matrix(
        matrix, "Home", "Away", "total_goals_under", "total", 2.5
    )
    assert np.isclose(btts.p_model, 0.43)
    assert np.isclose(over.p_model, 0.23)
    assert np.isclose(under.p_model, 0.77)


def test_unsupported_prop_event_is_rejected_by_goal_adapter():
    result = goal_event_probabilities_from_score_matrix(
        toy_matrix(), "Home", "Away", "corners_threshold", "total", 9
    )
    assert result.status == "unsupported"
    assert result.p_model is None
