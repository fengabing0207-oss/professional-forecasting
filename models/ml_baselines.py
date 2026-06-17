"""
ML baselines for 1X2 — does a flexible model beat the white-box one?

PURPOSE
-------
Not to win at all costs, but to answer the project's real question: where does a
flexible ML model add value over the parametric Dixon-Coles model, and where
does it simply over-fit a few-thousand-row dataset? Two deliberately LIGHT
models, both predicting the 1X2 outcome directly from pre-match features:

- LogisticForecaster : multinomial logistic regression. Linear, ~handful of
  coefficients — a strong, hard-to-over-fit baseline.
- GBMForecaster      : sklearn HistGradientBoosting (gradient-boosted trees).
  The flexible model. The interesting comparison is whether its flexibility
  helps or hurts at this sample size.

NO neural networks: a few thousand matches would over-fit a net, and it would
add nothing interpretable. (LightGBM would be the natural choice but needs a
system OpenMP lib that isn't present here; HistGradientBoosting is the same
algorithm family — histogram-binned gradient-boosted trees — with no native
dependency.)

NO LEAKAGE
----------
Every feature for a match is computed from matches strictly BEFORE it:
- recent attack/defense form (mean goals for / against, last K games)
- recent points-per-game form
- rest days since each team's previous match
- neutral-venue flag

Training rows use each match's own as-of-kickoff state. At prediction time the
model contract is predict_proba(home, away, neutral) with no date, so we use
each team's state as of the fit cutoff (at most one block — ~weeks — stale, and
strictly leakage-free: never a match on/after the cutoff). Teams unseen in the
training fold yield None, matching the goal models so all are scored on the
same fixtures.
"""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import OUTCOMES, outcome_of

_K = 5                      # recency window for form
_REST_CAP = 60             # cap rest days (long international gaps are uninformative)
_FEATURES = [
    "home_gf", "home_ga", "home_ppg", "home_rest", "home_n",
    "away_gf", "away_ga", "away_ppg", "away_rest", "away_n",
    "form_diff", "atk_vs_def", "neutral",
]


def _team_form(hist: deque) -> tuple[float, float, float, int]:
    """(mean goals-for, mean goals-against, points-per-game, n) over recent games."""
    if not hist:
        return 0.0, 0.0, 1.0, 0          # neutral cold-start priors
    gf = np.mean([h["gf"] for h in hist])
    ga = np.mean([h["ga"] for h in hist])
    ppg = np.mean([h["pts"] for h in hist])
    return float(gf), float(ga), float(ppg), len(hist)


def _feature_row(home_state, away_state, h_last_date, a_last_date,
                 match_date, neutral: bool) -> list[float]:
    hgf, hga, hppg, hn = home_state
    agf, aga, appg, an = away_state
    h_rest = (match_date - h_last_date).days if h_last_date is not None else _REST_CAP
    a_rest = (match_date - a_last_date).days if a_last_date is not None else _REST_CAP
    h_rest = float(min(h_rest, _REST_CAP))
    a_rest = float(min(a_rest, _REST_CAP))
    form_diff = (hgf - hga) - (agf - aga)        # net-goal-rate edge
    atk_vs_def = hgf - aga                        # home attack vs away defence
    return [hgf, hga, hppg, h_rest, hn,
            agf, aga, appg, a_rest, an,
            form_diff, atk_vs_def, float(neutral)]


class _MLForecaster:
    """Shared feature plumbing; subclasses supply the sklearn classifier."""

    name = "ml_base"

    def _new_clf(self):
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp) -> "_MLForecaster":
        df = train.sort_values("date").reset_index(drop=True)
        hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=_K))
        last_date: dict[str, pd.Timestamp] = {}

        X, y = [], []
        for _, g in df.iterrows():
            h, a = g["home"], g["away"]
            md = g["date"]
            row = _feature_row(_team_form(hist[h]), _team_form(hist[a]),
                               last_date.get(h), last_date.get(a),
                               md, bool(g["neutral"]))
            X.append(row)
            y.append(outcome_of(int(g["hg"]), int(g["ag"])))
            # update state AFTER recording features (strictly-before guarantee)
            hg, ag = int(g["hg"]), int(g["ag"])
            hpts = 3 if hg > ag else 1 if hg == ag else 0
            apts = 3 if ag > hg else 1 if hg == ag else 0
            hist[h].append({"gf": hg, "ga": ag, "pts": hpts})
            hist[a].append({"gf": ag, "ga": hg, "pts": apts})
            last_date[h] = md
            last_date[a] = md

        self._clf = self._new_clf()
        self._clf.fit(np.array(X, dtype=float), np.array(y))
        self._classes = list(self._clf.classes_)
        # freeze team state as of the cutoff for prediction
        self._state = {t: _team_form(h) for t, h in hist.items()}
        self._last_date = dict(last_date)
        self._as_of = as_of
        self._teams = set(self._state)
        return self

    def predict_proba(self, home: str, away: str, neutral: bool
                      ) -> Optional[tuple[float, float, float]]:
        if home not in self._teams or away not in self._teams:
            return None
        row = _feature_row(self._state[home], self._state[away],
                           self._last_date.get(home), self._last_date.get(away),
                           self._as_of, neutral)
        proba = self._clf.predict_proba(np.array([row], dtype=float))[0]
        # reorder classifier output into canonical (home, draw, away)
        out = {c: float(p) for c, p in zip(self._classes, proba)}
        return tuple(out.get(o, 0.0) for o in OUTCOMES)  # type: ignore[return-value]


class LogisticForecaster(_MLForecaster):
    """Multinomial logistic regression — the linear, over-fit-resistant baseline."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.name = "logreg"

    def _new_clf(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=self.C),
        )


class GBMForecaster(_MLForecaster):
    """Gradient-boosted trees — the flexible model that may add value or over-fit."""

    def __init__(self, max_depth: int = 3, learning_rate: float = 0.05,
                 max_iter: int = 300, l2_regularization: float = 1.0,
                 min_samples_leaf: int = 40):
        self.kw = dict(max_depth=max_depth, learning_rate=learning_rate,
                       max_iter=max_iter, l2_regularization=l2_regularization,
                       min_samples_leaf=min_samples_leaf)
        self.name = "gbm"

    def _new_clf(self):
        # conservative settings (shallow trees, strong leaf/L2 regularisation)
        # precisely because the dataset is small and over-fitting is the risk.
        return HistGradientBoostingClassifier(**self.kw)
