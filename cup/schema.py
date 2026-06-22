"""Schemas and constants for Probability Cup questions."""
from __future__ import annotations

from dataclasses import dataclass


GOAL_EVENT_TYPES = {
    "home_win",
    "away_win",
    "draw",
    "team_win",
    "team_not_lose",
    "total_goals_over",
    "total_goals_under",
    "team_goals_over",
    "team_goals_under",
    "both_teams_score_yes",
    "both_teams_score_no",
    "clean_sheet",
    "exact_score",
    "win_by_margin",
}

MARKET_ONLY_EVENT_TYPES = {
    "corners_threshold",
    "team_corners_threshold",
    "shots_on_target_threshold",
    "team_shots_on_target_threshold",
    "offsides_threshold",
    "fouls_more_than_opponent",
    "cards_threshold",
    "red_card",
    "penalty_or_red_card",
    "player_anytime_scorer",
    "first_goal",
    "second_half_result",
    "unsupported_market_only",
}

ALL_EVENT_TYPES = GOAL_EVENT_TYPES | MARKET_ONLY_EVENT_TYPES

SUPPORTED_STATUSES = {
    "model_and_market",
    "market_only",
    "model_only",
    "manual_only",
    "unsupported_market_only",
    "missing_probability",
    "error",
}

QUESTION_COLUMNS = [
    "question_id",
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "raw_question",
    "event_type",
    "selection",
    "threshold",
    "player",
    "p_manual",
    "manual_weight",
    "notes",
]

ODDS_COLUMNS = [
    "question_id",
    "market_id",
    "outcome_key",
    "odds_format",
    "odds_value",
    "direct_probability",
    "bookmaker",
    "retrieved_at",
    "notes",
]

RESULT_COLUMNS = ["question_id", "actual_result", "crowd_brier", "notes"]


@dataclass(frozen=True)
class CupQuestion:
    question_id: str
    match_id: str
    home_team: str
    away_team: str
    raw_question: str
    event_type: str
    selection: str = ""
    threshold: float | None = None
    player: str = ""
    p_manual: float | None = None
    manual_weight: float | None = None
    notes: str = ""

    @property
    def is_model_supported(self) -> bool:
        return self.event_type in GOAL_EVENT_TYPES


def validate_event_type(event_type: str) -> str:
    value = str(event_type).strip()
    if value not in ALL_EVENT_TYPES:
        raise ValueError(f"unsupported event_type {event_type!r}; use unsupported_market_only if needed")
    return value
