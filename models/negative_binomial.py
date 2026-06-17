"""
Negative-Binomial goal model — relaxing Poisson's mean = variance assumption.

WHY
---
The README states the core limitation up front: Poisson forces variance = mean,
but real goal counts are over-dispersed (variance > mean). The consequence is
that the Dixon-Coles model systematically under-prices lopsided scorelines by
strong attacks (it ranked France's actual 3-1 over Senegal only ~7th). This
model is the controlled experiment that isolates that assumption.

WHAT CHANGES (and what does not)
--------------------------------
Identical mean structure to Dixon-Coles, fit by the same time-weighted MLE:

    lambda_home = exp( atk[h] - def[a] + home_adv*(1 - neutral) )
    lambda_away = exp( atk[a] - def[h] )

The ONLY change is the count distribution: each goal count is drawn from a
Negative Binomial with mean = lambda and a shared dispersion parameter r:

    Var = lambda + lambda^2 / r        (r -> infinity recovers Poisson)

So if r is estimated large, the data is ~Poisson and this model should match
Dixon-Coles; if r is moderate, the extra variance is real and this model should
price blowouts (and the exact-score NLL) better. The Dixon-Coles low-score
correction (rho) is retained so the comparison isolates the count distribution,
not the low-score handling.

This is exactly the white-box virtue the project is built on: a single, named,
testable assumption is swapped, and the backtest says whether it helped.
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dixoncoles import _tau                # reuse the DC low-score correction
from data import add_time_weights          # noqa: E402


def _nb_logpmf(k: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """log P(k goals) under NB with mean mu and dispersion r (vectorised)."""
    # parameterisation: mean=mu, var = mu + mu^2/r
    return (gammaln(k + r) - gammaln(r) - gammaln(k + 1)
            + r * (np.log(r) - np.log(r + mu))
            + k * (np.log(mu) - np.log(r + mu)))


def _nb_pmf_vector(mu: float, r: float, max_goals: int) -> np.ndarray:
    """PMF over 0..max_goals for NB(mean=mu, dispersion=r) via scipy."""
    p = r / (r + mu)                       # scipy nbinom: n failures, prob p
    k = np.arange(max_goals + 1)
    return nbinom.pmf(k, r, p)


class NegativeBinomialForecaster:
    """ForecastModel: Dixon-Coles mean structure with NB (over-dispersed) counts."""

    def __init__(self, half_life_days: float = 540.0, ridge: float = 1e-3,
                 max_goals: int = 10):
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.max_goals = max_goals
        self.name = "negative_binomial"
        self.teams_: list[str] = []
        self.idx_: dict[str, int] = {}
        self.atk_: Optional[np.ndarray] = None
        self.def_: Optional[np.ndarray] = None
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.r_: float = np.inf          # dispersion; large => ~Poisson

    # ---- training -------------------------------------------------------
    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp) -> "NegativeBinomialForecaster":
        df = add_time_weights(train, half_life_days=self.half_life_days, as_of=as_of)

        teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        hg = df["hg"].to_numpy()
        ag = df["ag"].to_numpy()
        neut = df["neutral"].to_numpy().astype(float)
        w = df["weight"].to_numpy() if "weight" in df else np.ones(len(df))

        def negll(p):
            atk = p[:n]
            dfn = p[n:2 * n]
            hadv = p[2 * n]
            rho = p[2 * n + 1]
            log_r = p[2 * n + 2]
            r = np.exp(log_r)
            lh = np.exp(atk[hi] - dfn[ai] + hadv * (1 - neut))
            la = np.exp(atk[ai] - dfn[hi])
            tau = np.clip(_tau(hg, ag, lh, la, rho), 1e-10, None)
            ll = w * (np.log(tau)
                      + _nb_logpmf(hg, lh, r)
                      + _nb_logpmf(ag, la, r))
            pen = self.ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
            return -np.sum(ll) + pen

        # init log_r at log(20): mildly over-dispersed start, free to grow toward Poisson
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.10], [np.log(20.0)]])
        # wide upper bound on r so a Poisson-preferring fit reveals itself as a
        # genuine optimum (r -> large), not an artefact of a tight ceiling.
        bounds = ([(-3, 3)] * (2 * n)
                  + [(-1.0, 1.0), (-0.2, 0.2), (np.log(0.5), np.log(5000.0))])
        res = minimize(negll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "maxfun": 150000})

        atk = res.x[:n] - res.x[:n].mean()       # center for identifiability
        self.teams_, self.idx_ = teams, idx
        self.atk_, self.def_ = atk, res.x[n:2 * n]
        self.home_adv_ = float(res.x[2 * n])
        self.rho_ = float(res.x[2 * n + 1])
        self.r_ = float(np.exp(res.x[2 * n + 2]))
        self.fit_result_ = res
        return self

    # ---- prediction -----------------------------------------------------
    def _can_predict(self, home: str, away: str) -> bool:
        return (self.atk_ is not None
                and home in self.idx_ and away in self.idx_)

    def rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        h, a = self.idx_[home], self.idx_[away]
        lh = np.exp(self.atk_[h] - self.def_[a] + self.home_adv_ * (0 if neutral else 1))
        la = np.exp(self.atk_[a] - self.def_[h])
        return float(lh), float(la)

    def predict_scoreline(self, home: str, away: str, neutral: bool
                          ) -> Optional[np.ndarray]:
        if not self._can_predict(home, away):
            return None
        lh, la = self.rates(home, away, neutral)
        ph = _nb_pmf_vector(lh, self.r_, self.max_goals)
        pa = _nb_pmf_vector(la, self.r_, self.max_goals)
        M = np.outer(ph, pa)
        for x in (0, 1):                         # DC low-score correction
            for y in (0, 1):
                M[x, y] *= _tau(np.array([x]), np.array([y]),
                                np.array([lh]), np.array([la]), self.rho_)[0]
        return M / M.sum()

    def predict_proba(self, home: str, away: str, neutral: bool
                      ) -> Optional[tuple[float, float, float]]:
        M = self.predict_scoreline(home, away, neutral)
        if M is None:
            return None
        p_home = float(np.tril(M, -1).sum())
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, 1).sum())
        return p_home, p_draw, p_away
