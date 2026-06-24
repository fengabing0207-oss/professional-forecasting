"""Odds conversion and no-vig normalization for manual market inputs."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.metrics import devig_probs


SUPPORTED_ODDS_FORMATS = {"decimal", "american", "probability", "direct", "direct_probability"}


def _is_missing(value: object) -> bool:
    return value is None or bool(pd.isna(value)) or value == ""


def decimal_to_implied_probability(odds: float) -> float:
    """Convert positive decimal odds to raw implied probability."""
    value = float(odds)
    if value <= 1.0:
        raise ValueError(f"decimal odds must be greater than 1.0, got {odds!r}")
    return 1.0 / value


def american_to_implied_probability(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    value = float(odds)
    if value == 0:
        raise ValueError("American odds cannot be zero")
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def direct_probability(probability: float) -> float:
    """Validate and return a direct probability input."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability!r}")
    return value


def implied_probability(odds_format: str, odds_value: object = None,
                        direct_probability_value: object = None) -> float:
    """Convert one odds row to a raw probability before market normalization."""
    fmt = str(odds_format).strip().lower()
    if fmt not in SUPPORTED_ODDS_FORMATS:
        raise ValueError(f"unsupported odds_format: {odds_format!r}")
    if fmt == "decimal":
        if _is_missing(odds_value):
            raise ValueError("decimal odds require odds_value")
        return decimal_to_implied_probability(float(odds_value))
    if fmt == "american":
        if _is_missing(odds_value):
            raise ValueError("American odds require odds_value")
        return american_to_implied_probability(float(odds_value))
    if _is_missing(direct_probability_value):
        raise ValueError("direct probability rows require direct_probability")
    return direct_probability(float(direct_probability_value))


def no_vig_normalize(probabilities: Iterable[float]) -> list[float]:
    """Normalize a 2-way or 3-way market to a probability simplex.

    For decimal odds groups this reuses the project's existing ``devig_probs``
    logic. For already-converted raw probabilities, proportional normalization
    is equivalent to the existing implementation.
    """
    raw = np.asarray(list(probabilities), dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional sequence")
    if raw.size not in (2, 3):
        raise ValueError("no-vig normalization is supported for 2-way and 3-way markets")
    if np.any(raw < 0) or not np.all(np.isfinite(raw)):
        raise ValueError("probabilities must be finite and non-negative")
    total = raw.sum()
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    if raw.size == 3 and np.all(raw > 0):
        synthetic_decimal = 1.0 / raw.reshape(1, -1)
        return [float(x) for x in devig_probs(synthetic_decimal)[0]]
    return [float(x) for x in raw / total]


def market_probabilities_from_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Return question-level market probabilities from a manual odds table.

    Rows are grouped by ``market_id`` when present, otherwise by ``question_id``.
    Two- and three-outcome groups are no-vig normalized. Single rows keep their
    raw implied/direct probability because there is no companion outcome to
    remove overround against.
    """
    required = {
        "question_id", "market_id", "outcome_key", "odds_format",
        "odds_value", "direct_probability",
    }
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"manual odds file missing columns: {sorted(missing)}")
    if odds.empty:
        return pd.DataFrame(columns=["question_id", "p_market"])

    rows: list[dict[str, object]] = []
    work = odds.copy()
    work["question_id"] = work["question_id"].astype(str)
    group_key = work["market_id"].where(work["market_id"].notna() & (work["market_id"] != ""),
                                        work["question_id"])
    for _, group in work.groupby(group_key, dropna=False):
        raw_probs = [
            implied_probability(row["odds_format"], row["odds_value"], row["direct_probability"])
            for _, row in group.iterrows()
        ]
        if len(raw_probs) in (2, 3):
            probs = no_vig_normalize(raw_probs)
        elif len(raw_probs) == 1:
            probs = [direct_probability(raw_probs[0])]
        else:
            raise ValueError("market groups must contain one, two, or three outcomes")
        for (_, row), prob in zip(group.iterrows(), probs):
            rows.append({"question_id": str(row["question_id"]), "p_market": float(prob)})
    return pd.DataFrame(rows)
