import numpy as np

from cup.question_mapper import (
    both_teams_score_yes_probability,
    clean_sheet_probability,
    draw_probability,
    exact_score_probability,
    home_win_probability,
    probability_from_score_matrix,
    total_goals_over_probability,
)


def toy_matrix():
    return np.array([
        [0.10, 0.05, 0.02],
        [0.15, 0.20, 0.08],
        [0.25, 0.10, 0.05],
    ])


def test_goal_event_probability_extraction_from_toy_score_matrix():
    matrix = toy_matrix()
    assert np.isclose(home_win_probability(matrix), 0.50)
    assert np.isclose(draw_probability(matrix), 0.35)
    assert np.isclose(total_goals_over_probability(matrix, 2.5), 0.23)
    assert np.isclose(both_teams_score_yes_probability(matrix), 0.43)
    assert np.isclose(clean_sheet_probability(matrix, "home"), 0.50)
    assert np.isclose(exact_score_probability(matrix, 2, 1), 0.10)


def test_probability_from_score_matrix_exact_score_and_team_not_lose():
    exact = probability_from_score_matrix(toy_matrix(), "exact_score", selection="2-1")
    assert np.isclose(exact.p_model, 0.10)

    not_lose = probability_from_score_matrix(
        toy_matrix(),
        "team_not_lose",
        selection="home",
    )
    assert np.isclose(not_lose.p_model, 0.85)
