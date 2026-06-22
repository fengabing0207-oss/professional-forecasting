"""Score resolved Probability Cup predictions."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from cup.schema import RESULT_COLUMNS


def _missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value)) or value == ""


def _probability(value: Any, name: str) -> float:
    if _missing(value):
        raise ValueError(f"{name} is required")
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return p


def brier_score(p: float, actual: float) -> float:
    """Binary Brier score for a single Probability Cup question."""
    prob = _probability(p, "p")
    outcome = _probability(actual, "actual")
    return float((prob - outcome) ** 2)


def relative_brier_points(user_brier: float, crowd_brier: float | None) -> float | None:
    """Relative Brier Points: positive means beating the crowd benchmark."""
    if _missing(crowd_brier):
        return None
    return float((float(crowd_brier) - float(user_brier)) * 100.0)


def score_predictions(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"results file missing columns: {sorted(missing)}")
    if "question_id" not in predictions.columns or "p_final" not in predictions.columns:
        raise ValueError("predictions file must include question_id and p_final")

    preds = predictions.copy()
    res = results.copy()
    preds["question_id"] = preds["question_id"].astype(str)
    res["question_id"] = res["question_id"].astype(str)
    scored = preds.merge(res[RESULT_COLUMNS], on="question_id", how="left",
                         suffixes=("", "_result"))
    user_brier: list[float | None] = []
    rbp: list[float | None] = []
    for _, row in scored.iterrows():
        if _missing(row.get("p_final")) or _missing(row.get("actual_result")):
            user_brier.append(None)
            rbp.append(None)
            continue
        brier = brier_score(row["p_final"], row["actual_result"])
        user_brier.append(brier)
        rbp.append(relative_brier_points(brier, row.get("crowd_brier")))
    scored["user_brier"] = user_brier
    scored["rbp"] = rbp
    scored["scored_at"] = datetime.now(timezone.utc).isoformat()
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Probability Cup predictions")
    parser.add_argument("--predictions", required=True, help="predictions CSV")
    parser.add_argument("--results", required=True, help="resolved results CSV")
    parser.add_argument("--output", required=True, help="scored output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    results = pd.read_csv(args.results)
    scored = score_predictions(predictions, results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scored.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
