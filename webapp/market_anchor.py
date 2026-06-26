"""Manual market-anchor helpers for local live prediction workflows."""
from __future__ import annotations

from typing import Any


def normalize_probability_percent(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("%", "")
    try:
        percent = float(text)
    except ValueError as exc:
        raise ValueError(f"market anchor probability must be a percent between 0 and 100, got {value!r}") from exc
    if 0 < percent < 1:
        raise ValueError("market anchor uses percent mode; enter 51 for 51%, not 0.51")
    if not 0 <= percent <= 100:
        raise ValueError(f"market anchor probability must be between 0 and 100, got {value!r}")
    return percent / 100.0


def implied_probability_from_odds(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        odds = float(text)
    except ValueError as exc:
        raise ValueError(f"market odds must be American or decimal odds, got {value!r}") from exc
    if text.startswith("+") or text.startswith("-"):
        if odds == 0:
            raise ValueError("American odds cannot be 0")
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return 100.0 / (odds + 100.0)
    if odds <= 1:
        raise ValueError(f"decimal odds must be greater than 1, got {value!r}")
    return 1.0 / odds


def market_anchor_probability(percent_value: Any, odds_value: Any) -> float | None:
    percent = normalize_probability_percent(percent_value)
    if percent is not None:
        return percent
    return implied_probability_from_odds(odds_value)


def odds_format(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    text = str(value).strip()
    if text.startswith("+") or text.startswith("-"):
        return "american"
    return "decimal"
