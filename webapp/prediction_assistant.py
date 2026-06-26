"""Deterministic local assistant for live Probability Cup predictions."""
from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd

from cup.schema import GOAL_EVENT_TYPES, ODDS_COLUMNS


EXPOSURE_WARNING = "You may be overexposed to the same match script."
EVENT_TYPE_ALIASES = {
    "player_second_half_shots_on_target": "player_second_half_shot_on_target",
    "player_second_half_sot": "player_second_half_shot_on_target",
    "player_shots_on_target": "player_shot_on_target",
}


def suggest_probability_for_question(
    row: Any,
    match_context: dict[str, Any] | None = None,
    calibration_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a transparent heuristic suggestion for one question."""
    data = _row_to_dict(row)
    context = match_context or {}
    raw_question = _text(data.get("raw_question"))
    event_type = _text(data.get("event_type"))
    normalized_event_type = normalize_event_type(event_type, raw_question)
    selection = _text(data.get("selection"))
    threshold = _optional_float(data.get("threshold"))
    player = _text(data.get("player"))
    period = _period(raw_question, normalized_event_type, selection)
    final_probability_percent = _text(data.get("final_probability_percent"))

    out = {
        "question_id": _text(data.get("question_id")),
        "raw_question": raw_question,
        "event_type": event_type,
        "normalized_event_type": normalized_event_type,
        "selection": selection,
        "threshold": "" if threshold is None else threshold,
        "player": player,
        "period": period,
        "parser_status": _text(data.get("status") or data.get("parser_status")),
        "model_support_status": _model_support_status(normalized_event_type),
        "suggested_probability": None,
        "suggested_probability_percent": "",
        "suggested_range_low": None,
        "suggested_range_high": None,
        "suggested_range": "",
        "confidence": "low",
        "reasoning": "No deterministic heuristic matched; user judgment is required.",
        "risk_flags": ["needs_manual"],
        "exposure_warnings": [],
        "final_probability_percent": final_probability_percent,
    }

    normalized_data = {**data, "event_type": normalized_event_type}
    suggestion = _base_suggestion(normalized_data, context)
    if suggestion is not None:
        probability, low, high, confidence, reason, flags = suggestion
        out.update({
            "suggested_probability": round(probability, 4),
            "suggested_probability_percent": _percent(probability),
            "suggested_range_low": round(low, 4),
            "suggested_range_high": round(high, 4),
            "suggested_range": f"{_percent(low)}-{_percent(high)}%",
            "confidence": confidence,
            "reasoning": reason,
            "risk_flags": flags,
        })

    final_probability = normalize_final_probability_percent(final_probability_percent)
    if final_probability is not None:
        out["risk_flags"] = _dedupe(
            out["risk_flags"] + _final_probability_risks(normalized_data, context, final_probability)
        )
    return out


def suggest_probabilities_for_question_csv(
    question_csv_text: str,
    match_context: dict[str, Any] | None = None,
    calibration_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if not question_csv_text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(question_csv_text)).head(10)
    rows = [
        suggest_probability_for_question(row, match_context, calibration_context)
        for _, row in df.iterrows()
    ]
    warnings = detect_match_script_exposure(rows)
    if warnings:
        for row in rows:
            row["exposure_warnings"] = warnings
    return pd.DataFrame(rows)


def assistant_rows_to_manual_odds_csv(rows: Any, match_id: str | None = None) -> str:
    odds_rows: list[dict[str, Any]] = []
    prefix = _safe_id(match_id or "manual")
    for row in _rows_to_dicts(rows):
        probability = normalize_final_probability_percent(row.get("final_probability_percent"))
        if probability is None:
            continue
        question_id = _text(row.get("question_id"))
        odds_rows.append({
            "question_id": question_id,
            "market_id": f"{prefix}_{_safe_id(question_id)}",
            "outcome_key": "yes",
            "odds_format": "direct_probability",
            "odds_value": "",
            "direct_probability": probability,
            "bookmaker": "manual",
            "retrieved_at": "",
            "notes": "live prediction assistant",
        })
    output = io.StringIO()
    pd.DataFrame(odds_rows, columns=ODDS_COLUMNS).to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return output.getvalue()


def normalize_final_probability_percent(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("%", "")
    try:
        percent = float(text)
    except ValueError as exc:
        raise ValueError(f"final probability must be a percent between 0 and 100, got {value!r}") from exc
    if 0 < percent < 1:
        raise ValueError("final probability uses percent mode; enter 51 for 51%, not 0.51")
    if not 0 <= percent <= 100:
        raise ValueError(f"final probability must be between 0 and 100, got {value!r}")
    return percent / 100.0


def normalize_event_type(event_type: str, raw_question: str = "") -> str:
    """Normalize known importer/user aliases without inventing unsupported model coverage."""
    normalized = EVENT_TYPE_ALIASES.get(_text(event_type), _text(event_type))
    raw_low = _text(raw_question).lower()
    if _is_compound_btts_total(normalized, raw_low):
        return "both_teams_score_and_total_goals_over"
    return normalized


def detect_match_script_exposure(rows: Any) -> list[str]:
    work = _rows_to_dicts(rows)
    favorite_rows = [
        row for row in work
        if _normalized_row_event_type(row) == "team_win"
        and _final_probability(row) >= 0.60
        and _text(row.get("selection"))
    ]
    favorite_team = _text(favorite_rows[0].get("selection")) if favorite_rows else ""
    btts_high = any("both_teams" in _normalized_row_event_type(row) and _final_probability(row) >= 0.50 for row in work)
    underdog_sot_high = any(
        _normalized_row_event_type(row) in {"team_shots_on_target_threshold", "shots_on_target_threshold"}
        and _is_underdog(_text(row.get("selection")), favorite_team)
        and _final_probability(row) >= 0.50
        for row in work
    )
    underdog_corners_high = any(
        _normalized_row_event_type(row) in {"team_corners_threshold", "corners_threshold"}
        and _is_underdog(_text(row.get("selection")), favorite_team)
        and _final_probability(row) >= 0.45
        for row in work
    )
    second_half_high = sum(
        1 for row in work
        if _period(_text(row.get("raw_question")), _normalized_row_event_type(row), _text(row.get("selection"))) == "second_half"
        and _final_probability(row) > 0.55
    )
    player_high = sum(
        1 for row in work
        if _normalized_row_event_type(row).startswith("player_") and _final_probability(row) > 0.60
    )
    underdog_output_overs = sum(
        1 for row in work
        if _is_underdog(_text(row.get("selection")), favorite_team)
        and _normalized_row_event_type(row) in {"team_shots_on_target_threshold", "team_corners_threshold", "both_teams_score_yes"}
        and _final_probability(row) >= 0.45
    )
    if favorite_rows and (underdog_sot_high or btts_high or underdog_corners_high):
        return [EXPOSURE_WARNING]
    if second_half_high >= 2 or player_high >= 2 or underdog_output_overs >= 2:
        return [EXPOSURE_WARNING]
    return []


def _base_suggestion(data: dict[str, Any], context: dict[str, Any]) -> tuple[float, float, float, str, str, list[str]] | None:
    event_type = _text(data.get("event_type"))
    raw = _text(data.get("raw_question"))
    raw_low = raw.lower()
    selection = _text(data.get("selection"))
    threshold = _optional_float(data.get("threshold"))
    player = _text(data.get("player"))
    favorite = _text(context.get("favorite_team"))
    script = _context_text(context)
    adjustments: list[tuple[str, float]] = []
    flags: list[str] = []
    confidence = "low"
    base_reason = ""
    probability: float | None = None
    low: float | None = None
    high: float | None = None

    if event_type == "penalty_or_red_card":
        probability, low, high, confidence = 0.28, 0.26, 0.32, "medium"
        base_reason = "Base prior 28% for penalty/red-card events."
        if _has_any(script, ["rivalry", "knockout", "must-win", "must win", "high-intensity"]):
            adjustments.append(("increased for high-intensity or knockout context", 0.03))
        if _has_any(script, ["low stakes", "dead rubber", "friendly"]):
            adjustments.append(("decreased for low-stakes context", -0.03))
    elif event_type in {"halftime_draw", "halftime_result"}:
        probability, low, high, confidence = 0.41, 0.38, 0.45, "medium"
        base_reason = "Base prior 41% for halftime draw-style markets."
        if _has_any(script, ["even", "balanced", "tight", "cagey"]):
            adjustments.append(("increased for evenly matched or cagey context", 0.03))
        if _has_any(script, ["strong favorite", "one-sided", "mismatch"]):
            adjustments.append(("decreased for strong favorite or one-sided context", -0.04))
    elif _is_compound_btts_total(event_type, raw_low):
        probability, low, high, confidence = 0.44, 0.40, 0.48, "low"
        base_reason = "Base prior 44% for compound BTTS plus total-goals conditions."
        flags.append("compound_condition")
        if _has_any(script, ["attacking", "open", "weak defense", "weak defences", "both need"]):
            adjustments.append(("increased for open attacking context", 0.04))
        if _has_any(script, ["one-sided", "defensive", "low tempo", "favorite can win 2-0"]):
            adjustments.append(("decreased for one-sided or defensive context", -0.05))
    elif event_type == "player_second_half_shot_on_target":
        probability, low, high, confidence = 0.48, 0.40, 0.55, "low"
        base_reason = "Base prior 48% for second-half-only player shot-on-target props."
        flags.append("high_variance_prop")
        if _has_any(f"{player} {script}", ["primary attacker", "star", "full match", "full 90"]):
            adjustments.append(("increased for primary attacker or minutes context", 0.05))
        if _has_any(script, ["rotation", "minutes uncertainty", "limited minutes", "sub risk"]):
            adjustments.append(("decreased for rotation or minutes uncertainty", -0.07))
    elif event_type == "player_shot_on_target":
        probability, low, high, confidence = 0.52, 0.40, 0.60, "low"
        base_reason = "Base prior 52% for player shot-on-target props."
        if _has_any(f"{player} {script}", ["striker", "primary attacker", "star"]):
            adjustments.append(("increased for striker/star role", 0.05))
        if _has_any(f"{player} {script}", ["winger"]):
            adjustments.append(("slightly increased for winger attacking role", 0.02))
        if _has_any(f"{player} {script}", ["midfielder"]):
            adjustments.append(("decreased for midfield role", -0.04))
        if _has_any(script, ["rotation", "minutes uncertainty", "limited minutes", "sub risk"]):
            adjustments.append(("decreased for minutes uncertainty", -0.06))
    elif event_type == "fouls_more_than_opponent":
        probability, low, high, confidence = 0.55, 0.52, 0.59, "medium"
        base_reason = "Base prior 55% for fouls-more-than-opponent props."
        flags.append("fouls_guardrail")
        if _is_underdog(selection, favorite):
            adjustments.append(("increased because selected team appears to be underdog/defending more", 0.03))
        if selection and favorite and selection.lower() == favorite.lower():
            adjustments.append(("decreased because selected team appears likely to have more possession", -0.04))
    elif event_type in {"corners_threshold", "team_corners_threshold"}:
        total_market = selection.lower() == "total" or "total" in raw_low
        if total_market and (threshold is None or threshold >= 9):
            probability, low, high, confidence = 0.50, 0.45, 0.55, "low"
            base_reason = "Base prior 50% for total 9+ corners."
            if _has_any(script, ["wide", "high tempo", "attacking", "crosses"]):
                adjustments.append(("increased for wide/high-tempo attacking context", 0.04))
        elif threshold is not None and threshold >= 5:
            probability, low, high, confidence = 0.35, 0.30, 0.40, "low"
            base_reason = "Base prior 35% for team/underdog 5+ corners."
            if _has_any(script, ["chase", "trailing", "need result"]):
                adjustments.append(("increased if underdog may chase the game", 0.04))
            if _has_any(script, ["pinned deep", "low possession"]):
                adjustments.append(("decreased if underdog likely pinned deep", -0.05))
        else:
            probability, low, high, confidence = 0.42, 0.35, 0.50, "low"
            base_reason = "Fallback prior for corners threshold; market context recommended."
    elif event_type in {"shots_on_target_threshold", "team_shots_on_target_threshold", "second_half_total_shots_on_target_threshold"}:
        if "both teams" in raw_low and _period(raw, event_type, selection) == "second_half":
            probability, low, high, confidence = 0.53, 0.50, 0.57, "low"
            base_reason = "Base prior 53% for both teams second-half 1+ SOT."
            if _has_any(script, ["both need", "must-win", "must win"]):
                adjustments.append(("increased because both teams may need the result", 0.04))
        elif _period(raw, event_type, selection) == "second_half" and (threshold is None or threshold >= 4):
            probability, low, high, confidence = 0.56, 0.52, 0.60, "medium"
            base_reason = "Base prior 56% for second-half total 4+ SOT."
            if _has_any(script, ["open late", "chase", "must-win", "must win"]):
                adjustments.append(("increased for likely open late game state", 0.04))
        elif _is_underdog(selection, favorite) and threshold is not None and threshold >= 4:
            probability, low, high, confidence = 0.36, 0.30, 0.42, "low"
            base_reason = "Base prior 36% for underdog 4+ SOT."
            if _has_any(script, ["attacking underdog", "transition", "counter"]):
                adjustments.append(("increased for attacking/transition underdog style", 0.04))
        else:
            probability, low, high, confidence = 0.45, 0.38, 0.52, "low"
            base_reason = "Fallback prior for shots-on-target threshold; match context recommended."
    elif event_type == "offsides_threshold":
        probability, low, high, confidence = 0.50, 0.45, 0.58, "low"
        base_reason = "Base prior 50% for offsides thresholds."
        if _has_any(script, ["high line", "through ball", "transition", "pace"]):
            adjustments.append(("increased for high defensive line or transition context", 0.05))
    elif event_type == "team_win":
        if not favorite and not _has_any(script, ["strong favorite", "even", "must-win", "must win"]):
            return None
        probability, low, high, confidence = 0.56, 0.50, 0.64, "low"
        base_reason = "Team win has no fixed prior; this conservative value only appears because context was provided."
        if selection and favorite and selection.lower() == favorite.lower():
            adjustments.append(("increased because selected team is marked as favorite", 0.05))
        if _has_any(script, ["strong favorite"]):
            adjustments.append(("increased for strong favorite context", 0.04))
        if _has_any(script, ["even", "balanced"]):
            adjustments.append(("decreased for evenly matched context", -0.04))

    if probability is None or low is None or high is None:
        return None

    total_adjustment = sum(delta for _, delta in adjustments)
    probability = _clamp(probability + total_adjustment, 0.01, 0.99)
    low = _clamp(low + total_adjustment, 0.01, 0.99)
    high = _clamp(high + total_adjustment, 0.01, 0.99)
    reason_parts = [base_reason]
    if adjustments:
        reason_parts.extend(f"{label} ({delta:+.0%})" for label, delta in adjustments)
    else:
        reason_parts.append("No context adjustment applied.")
    reason_parts.append("Review before submitting; this is a deterministic heuristic, not a trained model.")
    return probability, low, high, confidence, " ".join(reason_parts), _dedupe(flags)


def _final_probability_risks(data: dict[str, Any], context: dict[str, Any], probability: float) -> list[str]:
    event_type = _text(data.get("event_type"))
    selection = _text(data.get("selection"))
    threshold = _optional_float(data.get("threshold"))
    favorite = _text(context.get("favorite_team"))
    flags: list[str] = []
    if event_type == "penalty_or_red_card" and probability > 0.40:
        flags.append("penalty_red_card_above_40")
    if event_type == "player_second_half_shot_on_target" and probability > 0.55:
        flags.append("player_2h_sot_above_55")
    if event_type == "player_shot_on_target" and probability > 0.60:
        flags.append("player_sot_above_60")
    if event_type == "fouls_more_than_opponent" and probability > 0.60:
        flags.append("fouls_above_60")
    if event_type in {"corners_threshold", "team_corners_threshold"}:
        if threshold is not None and threshold >= 5 and _is_underdog(selection, favorite) and probability > 0.45:
            flags.append("underdog_5_plus_corners_above_45")
    return flags


def _model_support_status(event_type: str) -> str:
    event_type = normalize_event_type(event_type)
    if event_type in GOAL_EVENT_TYPES:
        return "goal-model supported"
    if event_type == "unsupported_market_only":
        return "unsupported / needs review"
    if event_type:
        return "market/manual-only"
    return "needs manual"


def _is_compound_btts_total(event_type: str, raw_low: str) -> bool:
    return (
        "both_teams" in event_type and "total" in event_type
    ) or (
        ("both teams" in raw_low or "btts" in raw_low)
        and ("3+" in raw_low or "3 or more" in raw_low or "total" in raw_low or "over" in raw_low)
    )


def _period(raw_question: str, event_type: str, selection: str) -> str:
    low = f"{raw_question} {event_type} {selection}".lower()
    if "second half" in low or "2h" in low or "2nd half" in low or "second_half" in low:
        return "second_half"
    if "halftime" in low or "half-time" in low or "half time" in low:
        return "halftime"
    return ""


def _context_text(context: dict[str, Any]) -> str:
    return " ".join(_text(v).lower() for v in context.values())


def _has_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(needle in low for needle in needles)


def _is_underdog(selection: str, favorite: str) -> bool:
    if not selection or not favorite:
        return False
    return selection.lower() not in {favorite.lower(), "yes", "total", "match", "draw", "draw_halftime"}


def _final_probability(row: dict[str, Any]) -> float:
    value = row.get("final_probability")
    if value not in (None, ""):
        return float(value)
    probability = normalize_final_probability_percent(row.get("final_probability_percent"))
    return -1.0 if probability is None else probability


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _rows_to_dicts(rows: Any) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    return [_row_to_dict(row) for row in rows]


def _normalized_row_event_type(row: dict[str, Any]) -> str:
    return normalize_event_type(
        _text(row.get("normalized_event_type") or row.get("event_type")),
        _text(row.get("raw_question")),
    )


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _percent(probability: float) -> float:
    value = round(probability * 100, 1)
    return int(value) if float(value).is_integer() else value


def _safe_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    return "_".join(part for part in cleaned.split("_") if part) or "manual"


def _dedupe(flags: list[str]) -> list[str]:
    return list(dict.fromkeys(flag for flag in flags if flag))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
