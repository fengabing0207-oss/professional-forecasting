"""
Calibration layer.

The point the whole project is built around: a model that picks the right
WINNER is not the same as a model whose PROBABILITIES are correct. Single games
can never validate a probability; a batch can. This module scores a log of
(predicted probabilities, actual outcome) pairs.

- Brier score: mean squared error of probabilistic forecasts (lower = better).
  Multiclass version for {home win, draw, away win}. 0.0 is perfect; a naive
  "always 1/3 each" baseline scores 0.667.
- Reliability bins: of the games where you said ~p, did ~p actually happen?
  Systematic gaps reveal over/under-confidence.

Usage: append a row per prediction, then call summary() once enough have
resolved. Designed to be dropped next to a JSONL log (same pattern as
ThesisBoard's journal persistence).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

OUTCOMES = ("home", "draw", "away")


def brier_multiclass(probs: np.ndarray, outcomes: list[str]) -> float:
    """probs: (N,3) in order (home,draw,away). outcomes: list of 'home'/'draw'/'away'."""
    onehot = np.zeros_like(probs)
    for k, o in enumerate(outcomes):
        onehot[k, OUTCOMES.index(o)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability(probs_for_event: np.ndarray, event_happened: np.ndarray,
                n_bins: int = 5) -> pd.DataFrame:
    """One-vs-rest reliability table for a single event (e.g. 'home win').

    For each probability bin: how often did the event actually occur vs the
    average forecast probability in that bin. predicted ~ observed = calibrated.
    """
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


def baseline_scores(outcomes: list[str], market_probs: np.ndarray | None = None) -> dict:
    """Reference points to beat.

    - uniform: always (1/3, 1/3, 1/3)
    - market: the de-vigged bookmaker probabilities, if supplied. Beating the
      market on Brier is the real bar; beating uniform is trivial.
    """
    n = len(outcomes)
    uni = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    out = {"uniform": brier_multiclass(uni, outcomes)}
    if market_probs is not None:
        out["market"] = brier_multiclass(market_probs, outcomes)
    return out
