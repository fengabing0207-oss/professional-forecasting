"""Small form helpers for the local Flask app."""
from __future__ import annotations

from typing import Any


def form_value(form: Any, name: str, default: str = "") -> str:
    value = form.get(name, default)
    return "" if value is None else str(value).strip()


def float_option(form: Any, name: str, default: float) -> float:
    value = form.get(name, "")
    if value in (None, ""):
        return default
    return float(value)


def require_fields(values: dict[str, str], fields: list[str]) -> None:
    missing = [field for field in fields if not values.get(field, "").strip()]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

