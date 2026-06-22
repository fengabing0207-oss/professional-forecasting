"""Import copied/manual Probability Cup questions into a reviewable CSV draft."""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from cup.schema import QUESTION_COLUMNS, validate_event_type


IMPORT_COLUMNS = [
    "question_id",
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "raw_question",
    "event_type",
    "selection",
    "threshold",
    "player",
    "p_manual",
    "manual_weight",
    "parser_confidence",
    "status",
    "notes",
]

PARSER_STATUSES = {"parsed", "needs_review", "unsupported", "error"}


@dataclass(frozen=True)
class ParsedQuestion:
    raw_question: str
    event_type: str
    selection: str = ""
    threshold: float | None = None
    player: str = ""
    parser_confidence: float = 0.0
    status: str = "needs_review"
    notes: str = ""


def _clean_question(text: str) -> str:
    text = re.sub(r"^\s*(?:Q\d+|\d+)[\).:\-]\s*", "", text.strip(), flags=re.I)
    return text.strip()


def _question_id(text: str, index: int) -> str:
    match = re.match(r"^\s*(?:Q)?(\d+)[\).:\-]\s*", text.strip(), flags=re.I)
    return f"q{match.group(1)}" if match else f"q{index}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("–", "-").replace("—", "-")).strip()


def _team_in_text(text: str, home_team: str, away_team: str) -> str | None:
    low = _norm(text)
    for team in (home_team, away_team):
        if team and re.search(rf"\b{re.escape(team.lower())}\b", low):
            return team
    return None


def _opponent(team: str, home_team: str, away_team: str) -> str:
    if team == home_team:
        return away_team
    if team == away_team:
        return home_team
    return ""


def _number_before_plus(text: str, phrase: str) -> float | None:
    match = re.search(rf"\b(\d+(?:\.\d+)?)\s*\+\s+{phrase}", text)
    return float(match.group(1)) if match else None


def _number_after_keyword(text: str, keyword: str) -> float | None:
    match = re.search(rf"\b{keyword}\s+(\d+(?:\.\d+)?)\b", text)
    return float(match.group(1)) if match else None


def parse_raw_question(question: str, home_team: str, away_team: str) -> ParsedQuestion:
    """Conservatively map one copied question to the draft schema."""
    raw = _clean_question(question)
    low = _norm(raw)
    home = home_team.lower()
    away = away_team.lower()

    try:
        if re.search(r"\b(penalty|spot kick)\b", low) and "red card" in low:
            return ParsedQuestion(raw, "penalty_or_red_card", "match", parser_confidence=0.90,
                                  status="parsed", notes="market/manual-only prop")

        if re.search(r"\b(tied|level|draw)\b", low) and re.search(r"\b(halftime|half-time|half time)\b", low):
            return ParsedQuestion(raw, "halftime_draw", "draw_halftime", parser_confidence=0.90,
                                  status="parsed", notes="halftime markets are market/manual-only")

        if "both teams" in low and re.search(r"\bscore\b", low):
            no = bool(re.search(r"\b(no|not)\b", low))
            event = "both_teams_score_no" if no else "both_teams_score_yes"
            return ParsedQuestion(raw, event, "yes" if not no else "no", parser_confidence=0.90,
                                  status="parsed")

        if "clean sheet" in low:
            team = _team_in_text(raw, home_team, away_team)
            if team:
                return ParsedQuestion(raw, "clean_sheet", team, parser_confidence=0.82,
                                      status="parsed")
            return ParsedQuestion(raw, "clean_sheet", parser_confidence=0.35,
                                  status="needs_review", notes="clean sheet team unclear")

        if "corner" in low:
            total = _number_before_plus(low, r"(?:total\s+)?corners?")
            team = _team_in_text(raw, home_team, away_team)
            if "total" in low or team is None:
                return ParsedQuestion(raw, "corners_threshold", "total", total,
                                      parser_confidence=0.82 if total is not None else 0.55,
                                      status="parsed" if total is not None else "needs_review",
                                      notes="market/manual-only prop")
            return ParsedQuestion(raw, "team_corners_threshold", team, total,
                                  parser_confidence=0.80 if total is not None else 0.50,
                                  status="parsed" if total is not None else "needs_review",
                                  notes="market/manual-only prop")

        if "shot" in low and "target" in low:
            total = _number_before_plus(low, r"(?:total\s+)?shots? on target")
            if total is None:
                total = _number_before_plus(low, r"shots? on target")
            team = _team_in_text(raw, home_team, away_team)
            if "total" in low or team is None:
                return ParsedQuestion(raw, "shots_on_target_threshold",
                                      "second_half_total" if "second half" in low else "total",
                                      total, parser_confidence=0.78 if total is not None else 0.45,
                                      status="parsed" if total is not None else "needs_review",
                                      notes="market/manual-only prop")
            return ParsedQuestion(raw, "team_shots_on_target_threshold", team, total,
                                  parser_confidence=0.82 if total is not None else 0.50,
                                  status="parsed" if total is not None else "needs_review",
                                  notes="market/manual-only prop")

        if "offside" in low:
            threshold = _number_before_plus(low, r"(?:times?\s+)?(?:offside|offsides)")
            if threshold is None:
                threshold = _number_after_keyword(low, "offside")
            team = _team_in_text(raw, home_team, away_team)
            return ParsedQuestion(raw, "offsides_threshold", team or "", threshold,
                                  parser_confidence=0.80 if team and threshold is not None else 0.45,
                                  status="parsed" if team and threshold is not None else "needs_review",
                                  notes="market/manual-only prop")

        if "foul" in low and "more" in low:
            team = _team_in_text(raw, home_team, away_team)
            return ParsedQuestion(raw, "fouls_more_than_opponent", team or "",
                                  parser_confidence=0.82 if team else 0.45,
                                  status="parsed" if team else "needs_review",
                                  notes="market/manual-only prop")

        if re.search(r"\b(win|wins)\b", low) and "match" in low:
            team = _team_in_text(raw, home_team, away_team)
            if team:
                return ParsedQuestion(raw, "team_win", team, parser_confidence=0.88,
                                      status="parsed")
            return ParsedQuestion(raw, "team_win", parser_confidence=0.35,
                                  status="needs_review", notes="winning team unclear")

        if re.search(r"\b(draw|tie|tied)\b", low) and "match" in low:
            return ParsedQuestion(raw, "draw", "draw", parser_confidence=0.82, status="parsed")

        if "goal" in low and ("over" in low or "under" in low or "+" in low):
            team = _team_in_text(raw, home_team, away_team)
            direction = "over" if ("over" in low or "+" in low) else "under"
            threshold = _number_after_keyword(low, direction)
            if threshold is None:
                threshold = _number_before_plus(low, r"(?:total\s+)?goals?")
            if team:
                event = f"team_goals_{direction}"
                return ParsedQuestion(raw, event, team, threshold,
                                      parser_confidence=0.80 if threshold is not None else 0.45,
                                      status="parsed" if threshold is not None else "needs_review")
            event = f"total_goals_{direction}"
            return ParsedQuestion(raw, event, "total", threshold,
                                  parser_confidence=0.78 if threshold is not None else 0.45,
                                  status="parsed" if threshold is not None else "needs_review")

        scorer = re.search(r"\bwill\s+(.+?)\s+score\b", low)
        if scorer:
            name = scorer.group(1).strip()
            if name in {home, away, "home", "away"}:
                return ParsedQuestion(raw, "team_goals_over",
                                      home_team if name == home else away_team if name == away else name,
                                      0.5, parser_confidence=0.45, status="needs_review",
                                      notes="team scoring wording is ambiguous")
            player = raw[raw.lower().find(name): raw.lower().find(name) + len(name)].strip()
            return ParsedQuestion(raw, "player_anytime_scorer", player=player,
                                  parser_confidence=0.55, status="needs_review",
                                  notes="player team must be reviewed; market/manual-only prop")

        return ParsedQuestion(raw, "unsupported_market_only", parser_confidence=0.10,
                              status="needs_review", notes="no conservative parser match")
    except Exception as exc:
        return ParsedQuestion(raw, "unsupported_market_only", parser_confidence=0.0,
                              status="error", notes=str(exc))


def _raw_text_questions(path: str) -> list[tuple[str, str]]:
    questions: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for index, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            questions.append((_question_id(line, index), _clean_question(line)))
    return questions


def _base_row(question_id: str, raw_question: str, match_id: str, match_date: str,
              home_team: str, away_team: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "match_id": match_id,
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "raw_question": raw_question,
        "event_type": "unsupported_market_only",
        "selection": "",
        "threshold": "",
        "player": "",
        "p_manual": "",
        "manual_weight": "",
        "parser_confidence": 0.0,
        "status": "needs_review",
        "notes": "",
    }


def import_raw_text(path: str, match_id: str, match_date: str, home_team: str,
                    away_team: str) -> pd.DataFrame:
    rows = []
    for question_id, question in _raw_text_questions(path):
        parsed = parse_raw_question(question, home_team, away_team)
        row = _base_row(question_id, parsed.raw_question, match_id, match_date, home_team, away_team)
        row.update({
            "event_type": parsed.event_type,
            "selection": parsed.selection,
            "threshold": "" if parsed.threshold is None else parsed.threshold,
            "player": parsed.player,
            "parser_confidence": parsed.parser_confidence,
            "status": parsed.status,
            "notes": parsed.notes,
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=IMPORT_COLUMNS)


def _json_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("questions", "items", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError("JSON input must be an object, a list, or contain a questions/items/data list")


def import_json(path: str, match_id: str, match_date: str, home_team: str,
                away_team: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for index, item in enumerate(_json_records(payload), start=1):
        raw = str(item.get("raw_question") or item.get("question") or item.get("text") or "").strip()
        if not raw:
            raw = json.dumps(item, sort_keys=True)
        question_id = str(item.get("question_id") or item.get("id") or f"q{index}")
        row = _base_row(
            question_id,
            raw,
            str(item.get("match_id") or match_id),
            str(item.get("match_date") or match_date),
            str(item.get("home_team") or home_team),
            str(item.get("away_team") or away_team),
        )
        if item.get("event_type"):
            event_type = validate_event_type(str(item["event_type"]))
            row.update({
                "event_type": event_type,
                "selection": item.get("selection", ""),
                "threshold": item.get("threshold", ""),
                "player": item.get("player", ""),
                "p_manual": item.get("p_manual", ""),
                "manual_weight": item.get("manual_weight", ""),
                "parser_confidence": item.get("parser_confidence", 1.0),
                "status": item.get("status", "parsed"),
                "notes": item.get("notes", "structured JSON input"),
            })
        else:
            parsed = parse_raw_question(raw, row["home_team"], row["away_team"])
            row.update({
                "event_type": parsed.event_type,
                "selection": parsed.selection,
                "threshold": "" if parsed.threshold is None else parsed.threshold,
                "player": parsed.player,
                "parser_confidence": parsed.parser_confidence,
                "status": parsed.status,
                "notes": parsed.notes,
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=IMPORT_COLUMNS)


def import_csv(path: str, match_id: str, match_date: str, home_team: str,
               away_team: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in QUESTION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    for column, value in (("match_id", match_id), ("match_date", match_date),
                          ("home_team", home_team), ("away_team", away_team)):
        df[column] = df[column].replace("", pd.NA).fillna(value)
    if "parser_confidence" not in df.columns:
        df["parser_confidence"] = df["event_type"].apply(lambda x: 1.0 if str(x).strip() else 0.0)
    if "status" not in df.columns:
        df["status"] = df["event_type"].apply(lambda x: "parsed" if str(x).strip() else "needs_review")
    return df[IMPORT_COLUMNS]


def detect_mode(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    return "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import copied/manual Probability Cup questions")
    parser.add_argument("--input", required=True, help="CSV, JSON, or raw text file")
    parser.add_argument("--input-format", choices=["auto", "csv", "json", "text"], default="auto")
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--match-date", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = detect_mode(args.input, args.input_format)
    if mode == "csv":
        out = import_csv(args.input, args.match_id, args.match_date, args.home_team, args.away_team)
    elif mode == "json":
        out = import_json(args.input, args.match_id, args.match_date, args.home_team, args.away_team)
    else:
        out = import_raw_text(args.input, args.match_id, args.match_date, args.home_team, args.away_team)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
