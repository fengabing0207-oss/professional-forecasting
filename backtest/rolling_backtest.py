"""
Rolling-origin backtest — the no-leakage evaluation harness.

THE PROBLEM THIS FIXES
----------------------
The original pipeline fit Dixon-Coles on the *entire* dataset (2021-2026) and
then "predicted" fixtures inside that same span. Every prediction therefore saw
matches played after it — classic look-ahead leakage. Reported skill under that
scheme is optimistic and uninterpretable.

THE GUARANTEE
-------------
Walk forward in time. Pick a sequence of cutoff dates T0 < T1 < ... Between
consecutive cutoffs lies one *test block*. For block [T, T_next):

    train  = matches with date  <  T          (optionally a rolling lookback)
    test   = matches with date in [T, T_next)
    model  = fresh fit on `train`, decay anchored at T

Because training is strictly ``date < T`` and every test match has
``date >= T``, no match is ever scored using information from its own match day
or later. The model is refit once per block (walk-forward / rolling-origin),
which is both leakage-safe and the standard time-series CV scheme.

OUTPUT
------
A tidy DataFrame, one row per (model, fixture), with the predicted W/D/L
probabilities, the realised outcome, and bookkeeping (train size, cutoff,
whether the model could score the fixture). This frame is the single input the
evaluation suite consumes — every model is compared on exactly these rows.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.base import OUTCOMES, outcome_of, ForecastModel   # noqa: E402
from data import filter_min_matches                            # noqa: E402  (src on path)

# A factory returns a FRESH, unfitted model for each block (no state bleed).
ModelFactory = Callable[[], ForecastModel]


@dataclass
class BacktestConfig:
    test_start: str = "2024-01-01"   # first cutoff; nothing before this is scored
    test_end: Optional[str] = None   # last date to score (default: data max)
    step_days: int = 30              # block width = refit cadence
    min_train_matches: int = 300     # skip a block if training fold is too thin
    train_window_days: Optional[int] = None  # None = expanding; else rolling lookback
    min_matches_per_team: int = 15   # team universe threshold, re-derived per cutoff
    max_goals: int = 10


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    config: BacktestConfig
    models: list[str] = field(default_factory=list)

    def save(self, path: str) -> str:
        self.predictions.to_csv(path, index=False)
        return path


def _exact_score_prob(model, game, max_goals: int) -> float:
    """Probability the model assigned to the realised exact scoreline.

    Only defined for goal models exposing ``predict_scoreline``; returns NaN for
    1X2-only models (the ML baselines). Goals beyond the matrix are clipped to
    the last cell so the metric stays defined for blowouts.
    """
    if not hasattr(model, "predict_scoreline"):
        return np.nan
    M = model.predict_scoreline(game["home"], game["away"], bool(game["neutral"]))
    if M is None:
        return np.nan
    i = min(int(game["hg"]), max_goals)
    j = min(int(game["ag"]), max_goals)
    return float(M[i, j])


def _iter_blocks(dates: pd.Series, cfg: BacktestConfig):
    """Yield (cutoff_T, next_T) block boundaries across the test span."""
    t = pd.Timestamp(cfg.test_start)
    end = pd.Timestamp(cfg.test_end) if cfg.test_end else dates.max()
    step = pd.Timedelta(days=cfg.step_days)
    while t <= end:
        yield t, min(t + step, end + pd.Timedelta(days=1))
        t = t + step


def run_rolling_backtest(
    df: pd.DataFrame,
    model_factories: dict[str, ModelFactory],
    cfg: BacktestConfig = BacktestConfig(),
    verbose: bool = True,
) -> BacktestResult:
    """Run a walk-forward backtest for one or more models on the same blocks.

    Parameters
    ----------
    df : full match frame with columns home, away, hg, ag, date, neutral.
    model_factories : name -> zero-arg callable producing a fresh model.
    """
    df = df.sort_values("date").reset_index(drop=True)
    rows: list[dict] = []

    for cutoff, nxt in _iter_blocks(df["date"], cfg):
        test = df[(df["date"] >= cutoff) & (df["date"] < nxt)]
        if test.empty:
            continue

        # training fold: strictly before the cutoff (=> no leakage)
        train = df[df["date"] < cutoff]
        if cfg.train_window_days is not None:
            lo = cutoff - pd.Timedelta(days=cfg.train_window_days)
            train = train[train["date"] >= lo]
        # Build the team universe from PAST matches only — re-deriving the
        # min-matches-per-team filter here (not globally at load time) keeps
        # universe construction leak-free. Teams that fall below threshold as of
        # the cutoff are absent from training, so the models return None for
        # them and the backtest records those fixtures as un-scorable.
        train = filter_min_matches(train, cfg.min_matches_per_team)
        if len(train) < cfg.min_train_matches:
            continue

        # one fresh fit per model per block
        fitted: dict[str, ForecastModel] = {}
        for name, factory in model_factories.items():
            m = factory()
            m.fit(train.copy(), as_of=cutoff)
            fitted[name] = m

        if verbose:
            print(f"  cutoff {cutoff.date()} | train={len(train):5d} | "
                  f"test={len(test):3d} | models={list(fitted)}")

        for _, g in test.iterrows():
            actual = outcome_of(int(g["hg"]), int(g["ag"]))
            base = {
                "date": g["date"], "home": g["home"], "away": g["away"],
                "neutral": bool(g["neutral"]), "hg": int(g["hg"]), "ag": int(g["ag"]),
                "actual": actual, "cutoff": cutoff, "train_n": len(train),
            }
            for name, model in fitted.items():
                probs = model.predict_proba(g["home"], g["away"], bool(g["neutral"]))
                row = dict(base, model=name)
                if probs is None:
                    row.update(p_home=np.nan, p_draw=np.nan, p_away=np.nan,
                               p_exact=np.nan, predictable=False)
                else:
                    ph, pd_, pa = probs
                    row.update(p_home=ph, p_draw=pd_, p_away=pa, predictable=True,
                               p_exact=_exact_score_prob(model, g, cfg.max_goals))
                rows.append(row)

    preds = pd.DataFrame(rows)
    return BacktestResult(predictions=preds, config=cfg,
                          models=list(model_factories))


# --------------------------------------------------------------------------
# Smoke test: run Dixon-Coles through the harness and sanity-check the output.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from data import load_matches                         # noqa: E402  (src on path)
    from models.dixon_coles import DixonColesForecaster   # noqa: E402

    DATA = os.path.join(_ROOT, "data", "results.csv")
    # load raw (no global team filter); the backtest re-derives the universe
    # per cutoff so universe construction is leak-free.
    matches = load_matches(DATA, since="2021-01-01", min_matches_per_team=0)

    cfg = BacktestConfig(test_start="2025-01-01", step_days=45, min_train_matches=300,
                         min_matches_per_team=15)
    print(f"Loaded {len(matches):,} matches "
          f"({matches['date'].min().date()} -> {matches['date'].max().date()})")
    print(f"Backtest: test_start={cfg.test_start} step={cfg.step_days}d "
          f"window={'expanding' if cfg.train_window_days is None else cfg.train_window_days}")

    res = run_rolling_backtest(
        matches,
        {"dixon_coles": lambda: DixonColesForecaster(half_life_days=540)},
        cfg,
    )
    p = res.predictions
    scored = p[p["predictable"]]
    print(f"\nRows: {len(p)} | scorable: {len(scored)} | "
          f"un-scorable (unseen team): {len(p) - len(scored)}")
    # leakage assertion: every prediction trained strictly before kickoff
    assert (p["cutoff"] <= p["date"]).all(), "LEAKAGE: cutoff after kickoff!"
    print("Leakage check PASSED: all cutoffs <= match date.")
    # probabilities well-formed
    s = scored[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6), "probabilities do not sum to 1"
    print(f"Probability check PASSED: W/D/L sums in [{s.min():.4f}, {s.max():.4f}].")
    out = res.save(os.path.join(_ROOT, "backtest_results", "smoke_dixon_coles.csv"))
    print(f"Saved -> {out}")
    print(scored.head(6)[["date", "home", "away", "p_home", "p_draw", "p_away",
                          "actual"]].to_string(index=False))
