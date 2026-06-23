"""Bridge between Flask routes and the existing Probability Cup engine."""
from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from cup.schema import GOAL_EVENT_TYPES, MARKET_ONLY_EVENT_TYPES, ODDS_COLUMNS
from cup.import_questions import IMPORT_COLUMNS, import_raw_text
from cup.predict_cup import OUTPUT_COLUMNS, build_predictions
from cup.scoring import score_predictions


def _read_csv_text(csv_text: str) -> pd.DataFrame:
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


def _to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def parse_raw_questions_to_csv(raw_text: str, home_team: str, away_team: str,
                               match_id: str, match_date: str | None = None) -> str:
    """Parse pasted questions without inventing odds, probabilities, or results."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="prob-cup-import-") as tmp:
        path = Path(tmp) / "raw_questions.txt"
        path.write_text(raw_text or "", encoding="utf-8")
        df = import_raw_text(str(path), match_id, match_date or "", home_team, away_team)
    return _to_csv_text(df[IMPORT_COLUMNS])


def run_prediction_csv(question_csv_text: str, odds_csv_text: str | None = None,
                       model_probs_csv_text: str | None = None,
                       options: dict[str, Any] | None = None) -> str:
    questions = _read_csv_text(question_csv_text)
    odds = _read_csv_text(odds_csv_text or "")
    if odds.empty:
        odds = pd.DataFrame(columns=[
            "question_id", "market_id", "outcome_key", "odds_format",
            "odds_value", "direct_probability", "bookmaker", "retrieved_at", "notes",
        ])
    model_probs = _read_csv_text(model_probs_csv_text or "")
    if model_probs.empty:
        model_probs = pd.DataFrame(columns=["question_id", "p_model", "model_family", "notes"])
    opts = options or {}
    predictions = build_predictions(
        questions,
        odds,
        model_probs,
        market_weight=float(opts.get("market_weight", 0.70)),
        model_weight=float(opts.get("model_weight", 0.30)),
        manual_weight=float(opts.get("manual_weight", 1.0)),
        min_prob=float(opts.get("min_prob", 0.01)),
        max_prob=float(opts.get("max_prob", 0.99)),
    )
    return _to_csv_text(predictions[OUTPUT_COLUMNS])


def question_csv_to_manual_probability_rows(question_csv_text: str) -> list[dict[str, Any]]:
    """Return review rows for entering manual probabilities without inventing them."""
    df = _read_csv_text(question_csv_text)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        event_type = _string_value(row.get("event_type"))
        status = _string_value(row.get("status"))
        p_manual = _string_value(row.get("p_manual"))
        reasons = []
        if event_type in MARKET_ONLY_EVENT_TYPES or event_type not in GOAL_EVENT_TYPES:
            reasons.append("market/manual-only")
        if status == "needs_review":
            reasons.append("needs_review")
        if not p_manual:
            reasons.append("p_manual blank")
        if not p_manual and (event_type in MARKET_ONLY_EVENT_TYPES or event_type not in GOAL_EVENT_TYPES):
            reasons.append("would be missing_probability without market/manual input")
        rows.append({
            "row_index": idx,
            "question_id": _string_value(row.get("question_id")),
            "raw_question": _string_value(row.get("raw_question")),
            "event_type": event_type,
            "selection": _string_value(row.get("selection")),
            "p_manual": p_manual,
            "status": status,
            "notes": _string_value(row.get("notes")),
            "manual_probability_percent": "",
            "highlight": bool(reasons),
            "highlight_reasons": "; ".join(dict.fromkeys(reasons)),
        })
    return rows


def normalize_manual_probability_percent(value: Any) -> float | None:
    """Convert percent-mode input to a probability decimal."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("%", "")
    try:
        percent = float(text)
    except ValueError as exc:
        raise ValueError(f"manual probability must be a percent between 0 and 100, got {value!r}") from exc
    if 0 < percent < 1:
        raise ValueError("manual probability uses percent mode; enter 51 for 51%, not 0.51")
    if not 0 <= percent <= 100:
        raise ValueError(f"manual probability must be between 0 and 100, got {value!r}")
    return percent / 100.0


def manual_probability_rows_to_odds_csv(rows: list[dict[str, Any]], match_id: str | None = None) -> str:
    """Build direct-probability manual odds CSV from workbench rows."""
    odds_rows: list[dict[str, Any]] = []
    prefix = _safe_market_id_part(match_id or "manual")
    for row in rows:
        probability = normalize_manual_probability_percent(row.get("manual_probability_percent"))
        if probability is None:
            continue
        question_id = _string_value(row.get("question_id"))
        odds_rows.append({
            "question_id": question_id,
            "market_id": f"{prefix}_{_safe_market_id_part(question_id)}",
            "outcome_key": "yes",
            "odds_format": "direct_probability",
            "odds_value": "",
            "direct_probability": probability,
            "bookmaker": "manual",
            "retrieved_at": "",
            "notes": "manual probability workbench",
        })
    return _to_csv_text(pd.DataFrame(odds_rows, columns=ODDS_COLUMNS))


def run_scoring_csv(predictions_csv_text: str, results_csv_text: str) -> str:
    predictions = _read_csv_text(predictions_csv_text)
    results = _read_csv_text(results_csv_text)
    scored = score_predictions(predictions, results)
    return _to_csv_text(scored)


def summarize_predictions_csv(predictions_csv_text: str) -> dict[str, Any]:
    df = _read_csv_text(predictions_csv_text)
    if df.empty:
        return {}
    numeric = pd.to_numeric(df.get("p_final"), errors="coerce")
    return {
        "rows": int(len(df)),
        "status_counts": _value_counts(df, "status"),
        "event_type_counts": _value_counts(df, "event_type"),
        "missing_probability": int((df.get("status") == "missing_probability").sum()),
        "extreme_high": int((numeric >= 0.80).sum()),
        "extreme_low": int((numeric <= 0.20).sum()),
        "average_p_final_by_event_type": _group_mean(df, "event_type", numeric),
        "average_p_final_by_status": _group_mean(df, "status", numeric),
    }


def summarize_scoring_csv(scoring_csv_text: str) -> dict[str, Any]:
    df = _read_csv_text(scoring_csv_text)
    if df.empty:
        return {}
    user_brier = pd.to_numeric(df.get("user_brier"), errors="coerce")
    rbp = pd.to_numeric(df.get("rbp"), errors="coerce")
    crowd = pd.to_numeric(df.get("crowd_brier"), errors="coerce")
    actual = pd.to_numeric(df.get("actual_result"), errors="coerce")
    worst = df.assign(_user_brier=user_brier).sort_values("_user_brier", ascending=False).head(5)
    best = df.assign(_user_brier=user_brier).sort_values("_user_brier", ascending=True).head(5)
    return {
        "rows": int(len(df)),
        "average_user_brier": _maybe_float(user_brier.mean()),
        "sum_rbp": _maybe_float(rbp.sum()) if rbp.notna().any() else None,
        "average_user_brier_by_event_type": _group_mean(df, "event_type", user_brier),
        "average_user_brier_by_status": _group_mean(df, "status", user_brier),
        "worst_5": _compact_questions(worst),
        "best_5": _compact_questions(best),
        "missing_actual_result": int(actual.isna().sum()),
        "missing_crowd_brier": int(crowd.isna().sum()),
    }


def flag_prediction_risks(predictions_csv_text: str) -> list[dict[str, Any]]:
    df = _read_csv_text(predictions_csv_text)
    if df.empty:
        return []
    risks: list[dict[str, Any]] = []
    goal_events = {
        "home_win", "away_win", "draw", "team_win", "team_not_lose",
        "total_goals_over", "total_goals_under", "team_goals_over",
        "team_goals_under", "both_teams_score_yes", "both_teams_score_no",
        "clean_sheet", "exact_score", "win_by_margin",
    }
    prop_events = set(df["event_type"].dropna()) - goal_events
    for _, row in df.iterrows():
        qid = row.get("question_id", "")
        event_type = row.get("event_type", "")
        status = row.get("status", "")
        p_final = pd.to_numeric(pd.Series([row.get("p_final")]), errors="coerce").iloc[0]
        p_model = pd.to_numeric(pd.Series([row.get("p_model")]), errors="coerce").iloc[0]
        if status == "missing_probability":
            risks.append(_risk(qid, "missing_probability", "No final probability"))
        if pd.isna(p_final):
            risks.append(_risk(qid, "blank_p_final", "p_final is blank"))
        elif p_final >= 0.80:
            risks.append(_risk(qid, "extreme_high", "p_final >= 0.80"))
        elif p_final <= 0.20:
            risks.append(_risk(qid, "extreme_low", "p_final <= 0.20"))
        if event_type in prop_events and not pd.isna(p_model):
            risks.append(_risk(qid, "unsupported_prop_with_model", "Unsupported prop has p_model"))
        if event_type in goal_events and pd.isna(p_model):
            risks.append(_risk(qid, "goal_event_without_model", "Model-supported goal event has no p_model"))
    return risks


def summary_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("").value_counts().to_dict().items()}


def _group_mean(df: pd.DataFrame, group_col: str, values: pd.Series) -> dict[str, float | None]:
    if group_col not in df:
        return {}
    tmp = pd.DataFrame({"group": df[group_col].fillna(""), "value": values})
    return {str(k): _maybe_float(v) for k, v in tmp.groupby("group")["value"].mean().to_dict().items()}


def _maybe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _compact_questions(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "question_id": row.get("question_id", ""),
            "raw_question": row.get("raw_question", ""),
            "user_brier": _maybe_float(row.get("_user_brier")),
        })
    return rows


def _risk(question_id: Any, kind: str, message: str) -> dict[str, Any]:
    return {"question_id": str(question_id), "kind": kind, "message": message}


def _string_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _safe_market_id_part(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    return "_".join(part for part in cleaned.split("_") if part) or "manual"
