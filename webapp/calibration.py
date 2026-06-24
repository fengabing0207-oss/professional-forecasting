"""Settled-history calibration helpers for local Probability Cup review."""
from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd


SETTLED_HISTORY_COLUMNS = [
    "session_id",
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "question_id",
    "raw_question",
    "event_type",
    "selection",
    "user_prob",
    "crowd_prob",
    "actual_result",
    "platform_rbp",
    "notes",
]

PROBABILITY_BUCKETS = [
    (0.0, 0.2, "0-20%"),
    (0.2, 0.4, "20-40%"),
    (0.4, 0.6, "40-60%"),
    (0.6, 0.8, "60-80%"),
    (0.8, 1.0000001, "80-100%"),
]


def load_settled_history_csv(csv_text: str) -> pd.DataFrame:
    """Load manually copied settled history CSV into the normalized schema."""
    if not csv_text.strip():
        return pd.DataFrame(columns=SETTLED_HISTORY_COLUMNS)

    df = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    for column in SETTLED_HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[SETTLED_HISTORY_COLUMNS].copy()

    for column in ["user_prob", "crowd_prob", "actual_result"]:
        df[column] = pd.to_numeric(df[column].replace("", pd.NA), errors="coerce")

    invalid_user_prob = df["user_prob"].notna() & ~df["user_prob"].between(0, 1)
    invalid_crowd_prob = df["crowd_prob"].notna() & ~df["crowd_prob"].between(0, 1)
    invalid_actual = df["actual_result"].notna() & ~df["actual_result"].isin([0, 1])
    if invalid_user_prob.any() or invalid_crowd_prob.any() or invalid_actual.any():
        raise ValueError("user_prob/crowd_prob must be decimals 0..1 and actual_result must be 0 or 1.")

    return df


def compute_brier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric analysis columns while preserving raw platform_rbp text."""
    out = _with_schema(df)
    actual = pd.to_numeric(out["actual_result"], errors="coerce")
    user_prob = pd.to_numeric(out["user_prob"], errors="coerce")
    crowd_prob = pd.to_numeric(out["crowd_prob"], errors="coerce")

    out["platform_rbp_numeric"] = _numeric_platform_rbp(out["platform_rbp"])
    out["user_brier"] = (user_prob - actual) ** 2
    out["crowd_brier"] = (crowd_prob - actual) ** 2
    out["brier_edge"] = out["crowd_brier"] - out["user_brier"]
    directional = ((user_prob >= 0.5) == (actual == 1)).astype("boolean")
    directional.loc[user_prob.isna() | actual.isna()] = pd.NA
    out["directionally_correct"] = directional
    out["abs_user_crowd_deviation"] = (user_prob - crowd_prob).abs()
    out["probability_bucket"] = user_prob.apply(_bucket_label)
    return out


def summarize_settled_performance(df: pd.DataFrame) -> dict[str, Any]:
    scored = compute_brier_columns(df)
    return {
        "total_questions": int(len(scored)),
        "total_platform_rbp": _sum(scored["platform_rbp_numeric"]),
        "average_platform_rbp": _mean(scored["platform_rbp_numeric"]),
        "beat_crowd_count": int((scored["platform_rbp_numeric"] > 0).sum()),
        "below_crowd_count": int((scored["platform_rbp_numeric"] < 0).sum()),
        "mean_user_brier": _mean(scored["user_brier"]),
        "mean_crowd_brier": _mean(scored["crowd_brier"]),
        "mean_brier_edge": _mean(scored["brier_edge"]),
        "directional_correctness_count": int(scored["directionally_correct"].sum(skipna=True)),
        "average_abs_user_crowd_deviation": _mean(scored["abs_user_crowd_deviation"]),
    }


def summarize_by_event_type(df: pd.DataFrame) -> pd.DataFrame:
    scored = compute_brier_columns(df)
    if scored.empty:
        return pd.DataFrame(
            columns=["event_type", "questions", "total_platform_rbp", "average_platform_rbp", "mean_user_brier", "mean_crowd_brier", "mean_brier_edge"]
        )
    grouped = scored.groupby("event_type", dropna=False)
    return grouped.agg(
        questions=("question_id", "count"),
        total_platform_rbp=("platform_rbp_numeric", "sum"),
        average_platform_rbp=("platform_rbp_numeric", "mean"),
        mean_user_brier=("user_brier", "mean"),
        mean_crowd_brier=("crowd_brier", "mean"),
        mean_brier_edge=("brier_edge", "mean"),
    ).reset_index()


def summarize_by_probability_bucket(df: pd.DataFrame) -> pd.DataFrame:
    scored = compute_brier_columns(df)
    rows = []
    for _, _, label in PROBABILITY_BUCKETS:
        bucket_df = scored[scored["probability_bucket"] == label]
        rows.append(
            {
                "probability_bucket": label,
                "questions": int(len(bucket_df)),
                "total_platform_rbp": _sum(bucket_df["platform_rbp_numeric"]),
                "average_platform_rbp": _mean(bucket_df["platform_rbp_numeric"]),
                "mean_user_brier": _mean(bucket_df["user_brier"]),
                "mean_crowd_brier": _mean(bucket_df["crowd_brier"]),
                "mean_brier_edge": _mean(bucket_df["brier_edge"]),
            }
        )
    return pd.DataFrame(rows)


def find_largest_rbp_losses(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    scored = compute_brier_columns(df)
    return scored.sort_values("platform_rbp_numeric", ascending=True, na_position="last").head(n)


def find_largest_rbp_wins(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    scored = compute_brier_columns(df)
    return scored.sort_values("platform_rbp_numeric", ascending=False, na_position="last").head(n)


def find_largest_crowd_deviations(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    scored = compute_brier_columns(df)
    return scored.sort_values("abs_user_crowd_deviation", ascending=False, na_position="last").head(n)


def generate_guardrail_suggestions(df: pd.DataFrame) -> list[str]:
    scored = compute_brier_columns(df)
    if scored.empty:
        return []

    suggestions: list[str] = []
    event = scored["event_type"].fillna("").str.lower()
    question = scored["raw_question"].fillna("").str.lower()
    user_prob = pd.to_numeric(scored["user_prob"], errors="coerce")
    crowd_prob = pd.to_numeric(scored["crowd_prob"], errors="coerce")
    rbp = scored["platform_rbp_numeric"]
    actual = pd.to_numeric(scored["actual_result"], errors="coerce")

    if ((event.str.contains("foul") | question.str.contains("foul")) & (user_prob > 0.60) & (rbp < 0)).any():
        suggestions.append("Review fouls questions above 60% after RBP losses; consider requiring stronger evidence before taking high-confidence foul positions.")

    player_second_half_sot = (
        (event.str.contains("player") | question.str.contains("player") | question.str.contains("sadio") | question.str.contains("mane"))
        & (question.str.contains("second half") | question.str.contains("2nd half") | event.str.contains("second_half"))
        & (question.str.contains("shot") | question.str.contains("sot") | event.str.contains("shot"))
    )
    if (player_second_half_sot & (user_prob > 0.55) & (rbp < 0)).any():
        suggestions.append("Review player second-half shot-on-target questions above 55%; these may need a lower cap until supported by prop-specific evidence.")

    if ((event.str.contains("corner") | question.str.contains("corner")) & ((crowd_prob - user_prob) > 0.15) & (rbp < 0)).any():
        suggestions.append("Review corners questions where your probability is far below crowd after losses; check whether market/crowd baselines are being underweighted.")

    compound_btts_total = (
        (question.str.contains("both teams") | question.str.contains("btts") | event.str.contains("both_teams"))
        & (question.str.contains("total") | question.str.contains("goal") | question.str.contains("over") | event.str.contains("total_goals"))
    )
    if (compound_btts_total & (user_prob > 0.50) & (rbp < 0)).any():
        suggestions.append("Review compound BTTS plus total-goals questions above 50%; correlated conditions may be overstated.")

    favorite_win = (event.str.contains("team_win") | event.str.contains("home_win") | event.str.contains("away_win") | question.str.contains("win"))
    if (favorite_win & ((crowd_prob - user_prob) > 0.15) & (actual == 1)).any():
        suggestions.append("Review favorite-win questions where your probability is well below crowd and the outcome is yes; avoid underpricing clear favorites.")

    negative_event_totals = summarize_by_event_type(scored)
    for _, row in negative_event_totals.iterrows():
        event_type = str(row["event_type"] or "unknown")
        total_rbp = row["total_platform_rbp"]
        if pd.notna(total_rbp) and float(total_rbp) < 0:
            suggestions.append(f"Event type '{event_type}' has negative total platform RBP; review thresholds or confidence caps before the next dry run.")

    return suggestions


def to_normalized_csv(df: pd.DataFrame) -> str:
    normalized = _with_schema(df)[SETTLED_HISTORY_COLUMNS]
    output = io.StringIO()
    normalized.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return output.getvalue()


def _with_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in SETTLED_HISTORY_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out


def _numeric_platform_rbp(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace("%", "", regex=False).replace("", pd.NA),
        errors="coerce",
    )


def _bucket_label(probability: Any) -> str:
    if pd.isna(probability):
        return "missing"
    value = float(probability)
    for low, high, label in PROBABILITY_BUCKETS:
        if low <= value < high:
            return label
    if value == 1.0:
        return "80-100%"
    return "missing"


def _mean(series: pd.Series) -> float | None:
    value = series.mean(skipna=True)
    if pd.isna(value):
        return None
    return float(value)


def _sum(series: pd.Series) -> float:
    value = series.sum(skipna=True)
    if pd.isna(value):
        return 0.0
    return float(value)
