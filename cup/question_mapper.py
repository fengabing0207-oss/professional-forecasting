"""Map Probability Cup questions to model-supported event probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.schema import GOAL_EVENT_TYPES, validate_event_type


@dataclass(frozen=True)
class GoalProbabilityResult:
    p_model: float | None
    model_family: str
    notes: str = ""


def _matrix(score_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score_matrix must be a square 2D array")
    total = matrix.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("score_matrix must have positive finite mass")
    return matrix / total


def _selected_side(selection: str, home_team: str = "", away_team: str = "") -> str:
    value = str(selection).strip().lower()
    home = str(home_team).strip().lower()
    away = str(away_team).strip().lower()
    if value in {"home", "home_team", home}:
        return "home"
    if value in {"away", "away_team", away}:
        return "away"
    raise ValueError(f"selection must identify home or away team, got {selection!r}")


def _threshold(value: float | str | None) -> float:
    if value is None or value == "":
        raise ValueError("threshold is required for this event type")
    return float(value)


def _exact_score(selection: str) -> tuple[int, int]:
    value = str(selection).strip().replace(":", "-")
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError("exact_score selection must look like '2-1'")
    home_goals, away_goals = int(parts[0]), int(parts[1])
    if home_goals < 0 or away_goals < 0:
        raise ValueError("exact_score goals must be non-negative")
    return home_goals, away_goals


def home_win_probability(score_matrix: np.ndarray) -> float:
    matrix = _matrix(score_matrix)
    return float(np.tril(matrix, -1).sum())


def draw_probability(score_matrix: np.ndarray) -> float:
    matrix = _matrix(score_matrix)
    return float(np.trace(matrix))


def away_win_probability(score_matrix: np.ndarray) -> float:
    matrix = _matrix(score_matrix)
    return float(np.triu(matrix, 1).sum())


def total_goals_over_probability(score_matrix: np.ndarray, threshold: float) -> float:
    matrix = _matrix(score_matrix)
    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    return float(matrix[goals > threshold].sum())


def total_goals_under_probability(score_matrix: np.ndarray, threshold: float) -> float:
    matrix = _matrix(score_matrix)
    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    return float(matrix[goals < threshold].sum())


def team_goals_over_probability(score_matrix: np.ndarray, side: str,
                                threshold: float) -> float:
    matrix = _matrix(score_matrix)
    goals = np.arange(matrix.shape[0])
    if side == "home":
        return float(matrix[goals > threshold, :].sum())
    if side == "away":
        return float(matrix[:, goals > threshold].sum())
    raise ValueError(f"unknown side: {side!r}")


def team_goals_under_probability(score_matrix: np.ndarray, side: str,
                                 threshold: float) -> float:
    matrix = _matrix(score_matrix)
    goals = np.arange(matrix.shape[0])
    if side == "home":
        return float(matrix[goals < threshold, :].sum())
    if side == "away":
        return float(matrix[:, goals < threshold].sum())
    raise ValueError(f"unknown side: {side!r}")


def both_teams_score_yes_probability(score_matrix: np.ndarray) -> float:
    matrix = _matrix(score_matrix)
    return float(matrix[1:, 1:].sum())


def clean_sheet_probability(score_matrix: np.ndarray, side: str) -> float:
    matrix = _matrix(score_matrix)
    if side == "home":
        return float(matrix[:, 0].sum())
    if side == "away":
        return float(matrix[0, :].sum())
    raise ValueError(f"unknown side: {side!r}")


def exact_score_probability(score_matrix: np.ndarray, home_goals: int,
                            away_goals: int) -> float:
    matrix = _matrix(score_matrix)
    if home_goals >= matrix.shape[0] or away_goals >= matrix.shape[1]:
        return 0.0
    return float(matrix[home_goals, away_goals])


def win_by_margin_probability(score_matrix: np.ndarray, side: str,
                              margin: float) -> float:
    matrix = _matrix(score_matrix)
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]
    if side == "home":
        return float(matrix[(home_goals - away_goals) >= margin].sum())
    if side == "away":
        return float(matrix[(away_goals - home_goals) >= margin].sum())
    raise ValueError(f"unknown side: {side!r}")


def probability_from_score_matrix(
    score_matrix: np.ndarray,
    event_type: str,
    *,
    selection: str = "",
    threshold: float | str | None = None,
    home_team: str = "",
    away_team: str = "",
    model_family: str = "score_matrix",
) -> GoalProbabilityResult:
    """Extract a goal-event probability from an existing score matrix.

    The existing Dixon-Coles adapter normalizes its finite score matrix after
    max-goal truncation. These helpers keep that convention: probabilities are
    calculated on the normalized matrix they receive, so any tail mass beyond
    the model's max-goals cutoff is handled exactly as in the current codebase.
    """
    kind = validate_event_type(event_type)
    if kind not in GOAL_EVENT_TYPES:
        return GoalProbabilityResult(None, "", "event type is market/manual-only")

    if kind == "home_win":
        return GoalProbabilityResult(home_win_probability(score_matrix), model_family)
    if kind == "draw":
        return GoalProbabilityResult(draw_probability(score_matrix), model_family)
    if kind == "away_win":
        return GoalProbabilityResult(away_win_probability(score_matrix), model_family)
    if kind == "team_win":
        side = _selected_side(selection, home_team, away_team)
        p = home_win_probability(score_matrix) if side == "home" else away_win_probability(score_matrix)
        return GoalProbabilityResult(p, model_family)
    if kind == "team_not_lose":
        side = _selected_side(selection, home_team, away_team)
        p = 1.0 - (away_win_probability(score_matrix) if side == "home"
                   else home_win_probability(score_matrix))
        return GoalProbabilityResult(float(p), model_family)
    if kind == "total_goals_over":
        return GoalProbabilityResult(
            total_goals_over_probability(score_matrix, _threshold(threshold)),
            model_family,
        )
    if kind == "total_goals_under":
        return GoalProbabilityResult(
            total_goals_under_probability(score_matrix, _threshold(threshold)),
            model_family,
        )
    if kind == "team_goals_over":
        side = _selected_side(selection, home_team, away_team)
        return GoalProbabilityResult(
            team_goals_over_probability(score_matrix, side, _threshold(threshold)),
            model_family,
        )
    if kind == "team_goals_under":
        side = _selected_side(selection, home_team, away_team)
        return GoalProbabilityResult(
            team_goals_under_probability(score_matrix, side, _threshold(threshold)),
            model_family,
        )
    if kind == "both_teams_score_yes":
        return GoalProbabilityResult(both_teams_score_yes_probability(score_matrix), model_family)
    if kind == "both_teams_score_no":
        return GoalProbabilityResult(1.0 - both_teams_score_yes_probability(score_matrix), model_family)
    if kind == "clean_sheet":
        side = _selected_side(selection, home_team, away_team)
        return GoalProbabilityResult(clean_sheet_probability(score_matrix, side), model_family)
    if kind == "exact_score":
        home_goals, away_goals = _exact_score(selection)
        return GoalProbabilityResult(
            exact_score_probability(score_matrix, home_goals, away_goals),
            model_family,
        )
    if kind == "win_by_margin":
        side = _selected_side(selection, home_team, away_team)
        return GoalProbabilityResult(
            win_by_margin_probability(score_matrix, side, _threshold(threshold)),
            model_family,
        )
    raise ValueError(f"unhandled event_type: {event_type!r}")
