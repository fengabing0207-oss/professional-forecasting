"""CLI for exporting Phase 1 Probability Cup predictions."""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from cup.schema import GOAL_EVENT_TYPES, QUESTION_COLUMNS, validate_event_type
from market.anchor import blend_probability
from market.odds import market_probabilities_from_odds


OUTPUT_COLUMNS = [
    "question_id",
    "match_id",
    "raw_question",
    "event_type",
    "selection",
    "p_market",
    "p_model",
    "p_manual",
    "p_final",
    "status",
    "model_family",
    "notes",
]


def _missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value)) or value == ""


def _optional_probability(value: Any, name: str) -> float | None:
    if _missing(value):
        return None
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return p


def _read_csv(path: str, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _read_optional_model_probs(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["question_id", "p_model", "model_family", "notes"])
    df = pd.read_csv(path)
    required = {"question_id", "p_model", "model_family", "notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def build_predictions(
    questions: pd.DataFrame,
    odds: pd.DataFrame,
    model_probs: pd.DataFrame | None = None,
    *,
    market_weight: float = 0.70,
    model_weight: float = 0.30,
    manual_weight: float = 1.0,
    min_prob: float = 0.01,
    max_prob: float = 0.99,
) -> pd.DataFrame:
    """Build one prediction row per question."""
    missing = set(QUESTION_COLUMNS) - set(questions.columns)
    if missing:
        raise ValueError(f"question bank missing columns: {sorted(missing)}")

    market = market_probabilities_from_odds(odds)
    model = model_probs if model_probs is not None else pd.DataFrame()
    if model.empty:
        model = pd.DataFrame(columns=["question_id", "p_model", "model_family", "notes"])

    work = questions.copy()
    work["question_id"] = work["question_id"].astype(str)
    market["question_id"] = market["question_id"].astype(str)
    model["question_id"] = model["question_id"].astype(str)
    work = work.merge(market, on="question_id", how="left")
    work = work.merge(model, on="question_id", how="left", suffixes=("", "_model"))

    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        try:
            event_type = validate_event_type(row["event_type"])
            model_supported = event_type in GOAL_EVENT_TYPES
            p_market = _optional_probability(row.get("p_market"), "p_market")
            p_model = _optional_probability(row.get("p_model"), "p_model")
            p_manual = _optional_probability(row.get("p_manual"), "p_manual")
            result = blend_probability(
                p_market,
                p_model,
                p_manual,
                model_supported=model_supported,
                market_weight=market_weight,
                model_weight=model_weight,
                manual_weight=manual_weight,
                min_prob=min_prob,
                max_prob=max_prob,
            )
            notes = "; ".join(
                str(x) for x in [row.get("notes", ""), row.get("notes_model", ""), result.notes]
                if not _missing(x)
            )
            rows.append({
                "question_id": row["question_id"],
                "match_id": row.get("match_id", ""),
                "raw_question": row.get("raw_question", ""),
                "event_type": event_type,
                "selection": row.get("selection", ""),
                "p_market": p_market,
                "p_model": p_model if model_supported else None,
                "p_manual": p_manual,
                "p_final": result.p_final,
                "status": result.status,
                "model_family": row.get("model_family", "") if model_supported else "",
                "notes": notes,
            })
        except Exception as exc:  # keep batch exports inspectable
            rows.append({
                "question_id": row.get("question_id", ""),
                "match_id": row.get("match_id", ""),
                "raw_question": row.get("raw_question", ""),
                "event_type": row.get("event_type", ""),
                "selection": row.get("selection", ""),
                "p_market": row.get("p_market"),
                "p_model": row.get("p_model"),
                "p_manual": row.get("p_manual"),
                "p_final": None,
                "status": "error",
                "model_family": row.get("model_family", ""),
                "notes": str(exc),
            })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Probability Cup predictions")
    parser.add_argument("--questions", required=True, help="question bank CSV")
    parser.add_argument("--odds", required=True, help="manual odds CSV")
    parser.add_argument("--model-probs", help="optional model probability CSV")
    parser.add_argument("--output", required=True, help="output predictions CSV")
    parser.add_argument("--market-weight", type=float, default=0.70)
    parser.add_argument("--model-weight", type=float, default=0.30)
    parser.add_argument("--manual-weight", type=float, default=1.0)
    parser.add_argument("--min-prob", type=float, default=0.01)
    parser.add_argument("--max-prob", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = _read_csv(args.questions, QUESTION_COLUMNS)
    odds = _read_csv(args.odds, [
        "question_id", "market_id", "outcome_key", "odds_format",
        "odds_value", "direct_probability", "bookmaker", "retrieved_at", "notes",
    ])
    model_probs = _read_optional_model_probs(args.model_probs)
    predictions = build_predictions(
        questions,
        odds,
        model_probs,
        market_weight=args.market_weight,
        model_weight=args.model_weight,
        manual_weight=args.manual_weight,
        min_prob=args.min_prob,
        max_prob=args.max_prob,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    predictions.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
