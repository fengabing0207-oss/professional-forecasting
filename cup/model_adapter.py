"""Optional bridge from score-matrix models to Probability Cup goal events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.question_mapper import probability_from_score_matrix
from cup.schema import GOAL_EVENT_TYPES


@dataclass(frozen=True)
class ModelAdapterResult:
    p_model: float | None
    status: str
    model_family: str = ""
    notes: str = ""


def goal_event_probabilities_from_score_matrix(
    score_matrix: np.ndarray,
    home_team: str,
    away_team: str,
    event_type: str,
    selection: str,
    threshold: float | str | None,
    player: str | None = None,
) -> ModelAdapterResult:
    """Compute a supported goal-event probability from a model score matrix.

    This adapter is intentionally narrow. It does not train or load a model and
    it refuses halftime, prop, player, card, foul, corner, shot, and offside
    markets. Callers that need those event types should supply market/manual
    probabilities through the existing Phase 1 path.
    """
    if player:
        return ModelAdapterResult(None, "unsupported", notes="player props are not goal-matrix events")
    if event_type not in GOAL_EVENT_TYPES:
        return ModelAdapterResult(None, "unsupported", notes="event type is market/manual-only")
    try:
        result = probability_from_score_matrix(
            score_matrix,
            event_type,
            selection=selection,
            threshold=threshold,
            home_team=home_team,
            away_team=away_team,
            model_family="score_matrix",
        )
    except Exception as exc:
        return ModelAdapterResult(None, "error", notes=str(exc))
    return ModelAdapterResult(result.p_model, "model_supported", result.model_family, result.notes)
