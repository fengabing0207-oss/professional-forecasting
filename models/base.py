"""
Shared model contract for the rolling backtest.

Every forecaster — the parametric Dixon-Coles model, the negative-binomial
variant, and the ML baselines — implements the SAME small interface so the
backtest engine and the evaluation metrics never need to know which model they
are scoring. This is what makes "all models go through one backtest, one metric
suite" true rather than aspirational.

The contract is deliberately thin:

    fit(train, as_of)            -> self      (learn from matches strictly before as_of)
    predict_proba(home, away, neutral) -> (p_home, p_draw, p_away) | None

`predict_proba` returns probabilities in the fixed order (home, draw, away),
summing to ~1.0, or ``None`` when the model cannot score the fixture (e.g. a
team it never saw in training). Returning None — rather than guessing — keeps
the no-leakage guarantee honest: an un-trainable fixture is recorded as such,
not silently faked.

Models that also produce a full scoreline distribution (the goal models) may
optionally implement ``predict_scoreline`` for the exact-score NLL metric.
"""
from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable
import pandas as pd

# Canonical outcome ordering used everywhere downstream. Do not reorder.
OUTCOMES = ("home", "draw", "away")


def outcome_of(home_goals: int, away_goals: int) -> str:
    """Map a final scoreline to its 1X2 outcome label."""
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


@runtime_checkable
class ForecastModel(Protocol):
    """Structural type every model in this project satisfies."""

    name: str

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp) -> "ForecastModel":
        """Learn from ``train`` (all matches strictly before ``as_of``).

        ``as_of`` is the prediction cutoff; models that time-weight their data
        (Dixon-Coles, negative binomial) anchor the decay at this date so the
        most recent training match — not some future match — is "now".
        """
        ...

    def predict_proba(
        self, home: str, away: str, neutral: bool
    ) -> Optional[tuple[float, float, float]]:
        """Return (p_home, p_draw, p_away) summing to ~1, or None if un-scorable."""
        ...
