"""
Dixon-Coles forecaster — adapter around the existing src/dixoncoles.py.

This does NOT reimplement the model. It wraps the MLE Dixon-Coles fit so it
satisfies the shared ForecastModel contract (fit/predict_proba) used by the
rolling backtest. Two responsibilities live here:

1. Apply the time-decay weights anchored at the prediction cutoff (`as_of`),
   so during a backtest the decay treats the latest *training* match as "now"
   rather than leaking the global dataset maximum.
2. Translate the model's scoreline distribution into the canonical
   (p_home, p_draw, p_away) tuple, and expose the full score matrix for the
   exact-score NLL metric.

Unseen teams (not present in the training fold, or filtered out for low sample)
yield ``None`` from predict_proba, which the backtest records as un-scorable.
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd

# the original model + data helpers live in src/ (kept unmodified)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dixoncoles import DixonColes          # noqa: E402
from data import add_time_weights          # noqa: E402

from models.base import OUTCOMES           # noqa: E402


class DixonColesForecaster:
    """ForecastModel adapter for the parametric Dixon-Coles goal model."""

    def __init__(self, half_life_days: float = 540.0, ridge: float = 1e-3,
                 max_goals: int = 10):
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.max_goals = max_goals
        self.name = "dixon_coles"
        self._model: Optional[DixonColes] = None

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColesForecaster":
        # decay anchored at the cutoff: newest training match ~= full weight,
        # nothing from on/after `as_of` is present in `train` by construction.
        weighted = add_time_weights(train, half_life_days=self.half_life_days, as_of=as_of)
        self._model = DixonColes(ridge=self.ridge).fit(weighted)
        return self

    def _can_predict(self, home: str, away: str) -> bool:
        return (self._model is not None
                and home in self._model.idx_
                and away in self._model.idx_)

    def predict_proba(self, home: str, away: str, neutral: bool
                      ) -> Optional[tuple[float, float, float]]:
        if not self._can_predict(home, away):
            return None
        pred = self._model.predict(home, away, neutral=neutral)
        p = (pred["p_home"], pred["p_draw"], pred["p_away"])
        return tuple(float(x) for x in p)  # type: ignore[return-value]

    def predict_scoreline(self, home: str, away: str, neutral: bool
                          ) -> Optional[np.ndarray]:
        """Full (max_goals+1, max_goals+1) score-probability matrix, or None."""
        if not self._can_predict(home, away):
            return None
        lh, la = self._model.rates(home, away, neutral=neutral)
        return self._model.score_matrix(lh, la, max_goals=self.max_goals)
