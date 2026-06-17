"""
Dixon-Coles bivariate-Poisson goal model.

THIS is the "fitting lambda" layer the whole project is about. Instead of
hand-typing an expected-goals number per team, we LEARN, for every team, an
attack rating and a defense rating (plus a global home-advantage term and the
Dixon-Coles low-score correlation rho) by maximum likelihood on historical
results, with exponential time-decay weighting.

Model
-----
For a match between home team h and away team a:

    lambda_home = exp( atk[h] - def[a] + home_adv * (1 - neutral) )
    lambda_away = exp( atk[a] - def[h] )

    P(score = i, j) = tau(i, j) * Poisson(i; lambda_home) * Poisson(j; lambda_away)

where higher atk = scores more, higher def = concedes less, and tau is the
Dixon-Coles adjustment that re-weights 0-0 / 1-0 / 0-1 / 1-1 (which independent
Poisson misprices). rho < 0 inflates low-score draws.

Training = maximum likelihood: choose all atk[], def[], home_adv, rho to
maximize the time-weighted log-likelihood of every historical scoreline. This
is the same "define params -> define loss -> optimize" loop as any trained
model; here the loss is the negative weighted log-likelihood.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction (vectorized)."""
    out = np.ones_like(lh, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lh * la * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lh * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + la * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


class DixonColes:
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge          # tiny L2 keeps atk/def identifiable & stable
        self.teams_: list[str] = []
        self.idx_: dict[str, int] = {}
        self.atk_: np.ndarray | None = None
        self.def_: np.ndarray | None = None
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0

    # ---- training -------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "DixonColes":
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
            lh = np.exp(atk[hi] - dfn[ai] + hadv * (1 - neut))
            la = np.exp(atk[ai] - dfn[hi])
            tau = np.clip(_tau(hg, ag, lh, la, rho), 1e-10, None)
            ll = w * (np.log(tau)
                      + hg * np.log(lh) - lh
                      + ag * np.log(la) - la)
            pen = self.ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))
            return -np.sum(ll) + pen

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.10]])
        bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
        res = minimize(negll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "maxfun": 100000})

        atk = res.x[:n]
        dfn = res.x[n:2 * n]
        atk = atk - atk.mean()      # center for interpretability (differences fixed)
        self.teams_, self.idx_, self.atk_, self.def_ = teams, idx, atk, dfn
        self.home_adv_, self.rho_ = float(res.x[2 * n]), float(res.x[2 * n + 1])
        self.fit_result_ = res
        return self

    # ---- prediction -----------------------------------------------------
    def rates(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) for a fixture."""
        h, a = self.idx_[home], self.idx_[away]
        lh = np.exp(self.atk_[h] - self.def_[a] + self.home_adv_ * (0 if neutral else 1))
        la = np.exp(self.atk_[a] - self.def_[h])
        return float(lh), float(la)

    def score_matrix(self, lh: float, la: float, max_goals: int = 10) -> np.ndarray:
        from scipy.stats import poisson
        i = np.arange(max_goals + 1)
        ph = poisson.pmf(i, lh)
        pa = poisson.pmf(i, la)
        M = np.outer(ph, pa)
        # apply DC correction to the 2x2 low-score corner
        for x in (0, 1):
            for y in (0, 1):
                M[x, y] *= _tau(np.array([x]), np.array([y]),
                                np.array([lh]), np.array([la]), self.rho_)[0]
        return M / M.sum()

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        lh, la = self.rates(home, away, neutral)
        M = self.score_matrix(lh, la)
        win = np.tril(M, -1).sum()      # home goals > away goals
        draw = np.trace(M)
        loss = np.triu(M, 1).sum()
        # top scorelines
        flat = [((i, j), M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])]
        flat.sort(key=lambda kv: kv[1], reverse=True)
        over25 = sum(p for (i, j), p in flat if i + j >= 3)
        return {
            "lambda_home": lh, "lambda_away": la,
            "p_home": float(win), "p_draw": float(draw), "p_away": float(loss),
            "over_2_5": float(over25),
            "top_scores": flat[:6],
        }

    def ratings_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"team": self.teams_, "attack": self.atk_, "defense": self.def_})
                .sort_values("attack", ascending=False).reset_index(drop=True))
