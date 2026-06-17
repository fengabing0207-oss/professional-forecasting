"""
Evaluation suite — one scoring contract for every model.

Migrated and extended from the original src/calibration.py. The project's
thesis is that *calibration*, not winner-accuracy, is the thing to measure, so
this module is deliberately rich on proper scoring rules and calibration
diagnostics. Every model — Dixon-Coles, negative binomial, the ML baselines,
and the reference baselines — is scored by exactly these functions on exactly
the rows the rolling backtest produced.

Proper scoring rules (all: lower = better)
------------------------------------------
- log_loss          : multiclass cross-entropy. Punishes confident wrong calls
                      hardest; the canonical probabilistic-forecast loss.
- brier_score       : mean squared error of the (home,draw,away) vector. 0 is
                      perfect; always-1/3 scores 0.667.
- rps               : Ranked Probability Score. Unlike log-loss and Brier it
                      respects the ORDER home > draw > away — predicting a draw
                      when the result was an away win is penalised less than
                      predicting a home win. The right metric for an ordinal
                      1X2 outcome.
- exact_score_nll   : negative log-likelihood of the realised exact scoreline,
                      for models that emit a full score distribution. Tests the
                      goal model where it is supposed to add value (and where
                      Poisson's mean=variance assumption is supposed to hurt).

Calibration diagnostics
-----------------------
- calibration_curve : per-bin predicted-vs-observed frequency for one event
                      (reliability table). Systematic gaps = over/under-confidence.

Reference baselines (the bars to clear)
---------------------------------------
- UNIFORM           : always (1/3, 1/3, 1/3). Beating it is trivial; failing to
                      is a red flag.
- de-vigged MARKET  : bookmaker odds with the overround removed. Beating the
                      market is the real bar. Interface provided; if no odds are
                      supplied the baseline is simply absent from the report.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")
_EPS = 1e-15


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _onehot(outcomes: list[str]) -> np.ndarray:
    """(N,3) indicator matrix in (home,draw,away) order."""
    oh = np.zeros((len(outcomes), 3))
    for k, o in enumerate(outcomes):
        oh[k, OUTCOMES.index(o)] = 1.0
    return oh


def _as_probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return p


# --------------------------------------------------------------------------
# proper scoring rules
# --------------------------------------------------------------------------
def log_loss(probs, outcomes: list[str]) -> float:
    """Mean multiclass cross-entropy, -log p(true class)."""
    p = np.clip(_as_probs(probs), _EPS, 1.0)
    oh = _onehot(outcomes)
    return float(np.mean(-np.sum(oh * np.log(p), axis=1)))


def brier_score(probs, outcomes: list[str]) -> float:
    """Mean squared error of the probability vector vs the one-hot outcome."""
    p = _as_probs(probs)
    oh = _onehot(outcomes)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def rps(probs, outcomes: list[str]) -> float:
    """Ranked Probability Score for the ordinal outcome home > draw > away.

    RPS = 1/(r-1) * sum_{k=1}^{r-1} (CDF_pred_k - CDF_obs_k)^2,  r = 3 categories.
    Order-aware: closer-on-the-ladder mistakes cost less.
    """
    p = _as_probs(probs)
    oh = _onehot(outcomes)
    cdf_p = np.cumsum(p, axis=1)
    cdf_o = np.cumsum(oh, axis=1)
    # last cumulative term is identically 1 for both -> drop it (r-1 terms)
    return float(np.mean(np.sum((cdf_p[:, :-1] - cdf_o[:, :-1]) ** 2, axis=1)))


def exact_score_nll(p_exact) -> float:
    """Mean -log P(realised exact scoreline). NaNs (1X2-only models) ignored."""
    pe = np.asarray(p_exact, dtype=float)
    pe = pe[~np.isnan(pe)]
    if pe.size == 0:
        return float("nan")
    return float(np.mean(-np.log(np.clip(pe, _EPS, 1.0))))


# --------------------------------------------------------------------------
# calibration diagnostics
# --------------------------------------------------------------------------
def calibration_curve(probs_for_event: np.ndarray, event_happened: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Reliability table for a single event (e.g. 'home win').

    Per probability bin: mean forecast vs observed frequency. predicted ~
    observed along the diagonal = well calibrated.
    """
    probs_for_event = np.asarray(probs_for_event, dtype=float)
    event_happened = np.asarray(event_happened, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(probs_for_event, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": float(probs_for_event[m].mean()),
            "observed_freq": float(event_happened[m].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(probs_for_event: np.ndarray,
                               event_happened: np.ndarray,
                               n_bins: int = 10) -> float:
    """Sample-weighted mean |predicted - observed| across calibration bins."""
    tbl = calibration_curve(probs_for_event, event_happened, n_bins)
    if tbl.empty:
        return float("nan")
    w = tbl["n"] / tbl["n"].sum()
    return float((w * (tbl["mean_predicted"] - tbl["observed_freq"]).abs()).sum())


# --------------------------------------------------------------------------
# reference baselines
# --------------------------------------------------------------------------
def uniform_probs(n: int) -> np.ndarray:
    """Always (1/3, 1/3, 1/3)."""
    return np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))


def devig_probs(odds: np.ndarray, method: str = "proportional") -> np.ndarray:
    """De-vig decimal bookmaker odds into a probability simplex.

    odds : (N,3) decimal odds in (home,draw,away) order.
    The raw implied probabilities 1/odds sum to > 1 (the overround / vig);
    'proportional' (a.k.a. multiplicative / normalised) de-vigging divides by
    that sum. This is the MARKET baseline — the real bar to clear. Supply odds
    via the backtest's market interface; absent odds, the baseline is omitted.
    """
    odds = np.asarray(odds, dtype=float)
    raw = 1.0 / odds
    if method != "proportional":
        raise ValueError(f"unsupported de-vig method: {method!r}")
    return raw / raw.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# top-level evaluation over a backtest predictions frame
# --------------------------------------------------------------------------
@dataclass
class ModelScores:
    model: str
    n: int
    log_loss: float
    brier: float
    rps: float
    exact_nll: float
    ece_home: float
    accuracy: float

    def as_row(self) -> dict:
        return {
            "model": self.model, "n": self.n,
            "log_loss": self.log_loss, "brier": self.brier, "rps": self.rps,
            "exact_nll": self.exact_nll, "ece_home": self.ece_home,
            "accuracy": self.accuracy,
        }


def _score_block(model: str, probs: np.ndarray, outcomes: list[str],
                 p_exact: Optional[np.ndarray]) -> ModelScores:
    home_p = probs[:, 0]
    home_hit = np.array([o == "home" for o in outcomes], dtype=float)
    pred_idx = probs.argmax(axis=1)
    actual_idx = np.array([OUTCOMES.index(o) for o in outcomes])
    return ModelScores(
        model=model, n=len(outcomes),
        log_loss=log_loss(probs, outcomes),
        brier=brier_score(probs, outcomes),
        rps=rps(probs, outcomes),
        exact_nll=exact_score_nll(p_exact) if p_exact is not None else float("nan"),
        ece_home=expected_calibration_error(home_p, home_hit),
        accuracy=float(np.mean(pred_idx == actual_idx)),
    )


def evaluate_backtest(preds: pd.DataFrame,
                      include_baselines: bool = True,
                      market_odds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Score every model in a backtest frame on the common, scorable rows.

    To compare models fairly they must be scored on the SAME fixtures. We
    therefore restrict to fixtures every model could predict (the intersection
    of predictable rows), then score each model plus the reference baselines on
    that shared set.
    """
    df = preds.copy()
    models = sorted(df["model"].unique())

    # shared fixture set: those predictable by *all* models
    key = ["date", "home", "away", "cutoff"]
    predictable = df[df["predictable"]]
    counts = predictable.groupby(key)["model"].nunique()
    shared_keys = counts[counts == len(models)].index
    shared = predictable.set_index(key).loc[shared_keys].reset_index()

    rows: list[dict] = []
    ref_outcomes: Optional[list[str]] = None
    for m in models:
        sub = shared[shared["model"] == m].sort_values(key)
        probs = sub[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = sub["actual"].tolist()
        ref_outcomes = outcomes  # identical across models (same shared set)
        p_exact = sub["p_exact"].to_numpy() if "p_exact" in sub else None
        rows.append(_score_block(m, probs, outcomes, p_exact).as_row())

    if include_baselines and ref_outcomes is not None:
        n = len(ref_outcomes)
        rows.append(_score_block("baseline_uniform", uniform_probs(n),
                                 ref_outcomes, None).as_row())
        if market_odds is not None:
            mp = devig_probs(market_odds)
            rows.append(_score_block("baseline_market", mp, ref_outcomes,
                                     None).as_row())

    out = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    out.attrs["n_shared_fixtures"] = len(shared_keys)
    return out
