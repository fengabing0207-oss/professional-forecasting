"""Blend model, market, and manual probabilities into a final forecast."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _present(value: float | None) -> bool:
    return value is not None and isfinite(float(value))


def _validate_probability(value: float, name: str) -> float:
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return p


@dataclass(frozen=True)
class BlendResult:
    p_final: float | None
    status: str
    notes: str = ""


def blend_probability(
    p_market: float | None = None,
    p_model: float | None = None,
    p_manual: float | None = None,
    *,
    model_supported: bool = True,
    market_weight: float = 0.70,
    model_weight: float = 0.30,
    manual_weight: float = 1.0,
    min_prob: float = 0.01,
    max_prob: float = 0.99,
) -> BlendResult:
    """Blend available probabilities using conservative Phase 1 defaults."""
    if min_prob < 0 or max_prob > 1 or min_prob >= max_prob:
        raise ValueError("min_prob and max_prob must satisfy 0 <= min < max <= 1")
    weights = {"market_weight": market_weight, "model_weight": model_weight,
               "manual_weight": manual_weight}
    for name, value in weights.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    market = _validate_probability(p_market, "p_market") if _present(p_market) else None
    model = _validate_probability(p_model, "p_model") if _present(p_model) else None
    manual = _validate_probability(p_manual, "p_manual") if _present(p_manual) else None

    if not model_supported:
        model = None
        if market is not None:
            return BlendResult(_clamp(market, min_prob, max_prob),
                               "unsupported_market_only",
                               "unsupported event type: using market probability only")
        if manual is not None:
            return BlendResult(_clamp(manual * manual_weight, min_prob, max_prob),
                               "manual_only",
                               "unsupported event type: using manual probability only")
        return BlendResult(None, "missing_probability",
                           "unsupported event type and no market/manual probability")

    if market is not None and model is not None:
        denom = market_weight + model_weight
        if denom <= 0:
            raise ValueError("market_weight + model_weight must be positive")
        p = (market_weight * market + model_weight * model) / denom
        return BlendResult(_clamp(p, min_prob, max_prob), "model_and_market")
    if market is not None:
        return BlendResult(_clamp(market, min_prob, max_prob), "market_only")
    if model is not None:
        return BlendResult(_clamp(model, min_prob, max_prob), "model_only")
    if manual is not None:
        return BlendResult(_clamp(manual * manual_weight, min_prob, max_prob),
                           "manual_only")
    return BlendResult(None, "missing_probability")


def _clamp(value: float, min_prob: float, max_prob: float) -> float:
    return min(max(float(value), min_prob), max_prob)
