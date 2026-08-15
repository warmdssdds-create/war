"""Confluence scoring for entries.

The strategy uses separate scoring maps for trending and ranging regimes.  This
module exposes both a high-level compatibility function and lower-level helpers
that explain why a score passed or failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from src.strategy.regime_filter import Regime


TRENDING_WEIGHTS = {
    "ema_alignment": 0.25,
    "adx_strength": 0.20,
    "macd_momentum": 0.15,
    "rsi_bias": 0.10,
    "vwap_position": 0.10,
    "obv_trend": 0.10,
    "volume_confirmation": 0.10,
}

RANGING_WEIGHTS = {
    "bb_reversion": 0.25,
    "stochrsi_extreme": 0.20,
    "rsi_reversion": 0.15,
    "vwap_revert": 0.15,
    "atr_compression": 0.15,
    "volume_confirmation": 0.10,
}


@dataclass
class ConfluenceResult:
    """Scoring result compatible with the original interface."""

    score: float
    passed: bool


@dataclass
class ConfluenceDetail:
    """Detailed explanation of the scoring inputs and weighted outcome."""

    regime: Regime
    side: str
    threshold: float
    score: float
    passed: bool
    weights_used: dict[str, float]
    condition_values: dict[str, float] = field(default_factory=dict)
    weighted_components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def strongest_components(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the largest weighted components."""
        ordered = sorted(self.weighted_components.items(), key=lambda item: item[1], reverse=True)
        return ordered[:n]

    def as_dict(self) -> dict[str, object]:
        """Serialize the detail object for logging or alerts."""
        return {
            "regime": self.regime.value,
            "side": self.side,
            "threshold": self.threshold,
            "score": self.score,
            "passed": self.passed,
            "weights_used": self.weights_used,
            "condition_values": self.condition_values,
            "weighted_components": self.weighted_components,
            "notes": self.notes,
        }


def _resolve_weights(regime: Regime, weights: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, float]:
    """Merge caller-provided weights onto sensible defaults."""
    if regime == Regime.TRENDING:
        base = TRENDING_WEIGHTS.copy()
        if weights:
            base.update(weights.get("trending", {}))
        return base
    if regime == Regime.RANGING:
        base = RANGING_WEIGHTS.copy()
        if weights:
            base.update(weights.get("ranging", {}))
        return base
    return {}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _directional_conditions(frame: pd.DataFrame, side: str) -> dict[str, float]:
    """Compute directional condition values for long or short entries."""
    row = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) > 1 else row
    is_long = side == "Buy"

    ema_long = float(row["ema21"] > row["ema55"] > row["ema200"])
    ema_short = float(row["ema21"] < row["ema55"] < row["ema200"])
    adx_strength = min(max((_float(row.get("adx14")) - 20.0) / 20.0, 0.0), 1.0)
    plus_di = _float(row.get("plus_di"))
    minus_di = _float(row.get("minus_di"))
    di_bias = float(plus_di > minus_di) if is_long else float(minus_di > plus_di)
    macd_hist = _float(row.get("macd_hist"))
    macd_signal = min(abs(macd_hist) / max(abs(_float(row.get("close"))), 1.0) * 1000.0, 1.0)
    macd_momentum = macd_signal if (macd_hist > 0 if is_long else macd_hist < 0) else 0.0
    rsi = _float(row.get("rsi14"), 50.0)
    rsi_bias = min(max((rsi - 50.0) / 20.0, 0.0), 1.0) if is_long else min(max((50.0 - rsi) / 20.0, 0.0), 1.0)
    vwap_position = float(_float(row.get("close")) >= _float(row.get("vwap"))) if is_long else float(_float(row.get("close")) <= _float(row.get("vwap")))
    obv_trend = float(_float(row.get("obv")) >= _float(prev.get("obv"))) if is_long else float(_float(row.get("obv")) <= _float(prev.get("obv")))
    volume_confirmation = min(max((_float(row.get("volume_zscore")) + 1.0) / 2.0, 0.0), 1.0)

    bb_pct = _float(row.get("bb_pct"), 0.5)
    bb_reversion = (1.0 - bb_pct) if is_long else bb_pct
    stoch = _float(row.get("stochrsi_k"), 50.0)
    stochrsi_extreme = min(max((20.0 - stoch) / 20.0, 0.0), 1.0) if is_long else min(max((stoch - 80.0) / 20.0, 0.0), 1.0)
    rsi_reversion = min(max((35.0 - rsi) / 15.0, 0.0), 1.0) if is_long else min(max((rsi - 65.0) / 15.0, 0.0), 1.0)
    vwap_gap = abs(_float(row.get("vwap_distance_pct")))
    vwap_revert = min(vwap_gap / 0.006, 1.0)
    atr = _float(row.get("atr14"))
    atr_avg = max(_float(row.get("atr_avg100")), 1e-9)
    atr_compression = min(max(1.2 - (atr / atr_avg), 0.0), 1.0)

    return {
        "ema_alignment": ema_long if is_long else ema_short,
        "adx_strength": max(adx_strength, di_bias * 0.5),
        "macd_momentum": macd_momentum,
        "rsi_bias": rsi_bias,
        "vwap_position": vwap_position,
        "obv_trend": obv_trend,
        "volume_confirmation": volume_confirmation,
        "bb_reversion": bb_reversion,
        "stochrsi_extreme": stochrsi_extreme,
        "rsi_reversion": rsi_reversion,
        "vwap_revert": vwap_revert,
        "atr_compression": atr_compression,
    }


def explain_confluence(
    frame: pd.DataFrame,
    regime: Regime,
    weights: Mapping[str, float],
    side: str = "Buy",
    threshold: float = 0.0,
) -> ConfluenceDetail:
    """Return a rich explanation for the current confluence score."""

    if frame.empty:
        return ConfluenceDetail(
            regime=regime,
            side=side,
            threshold=threshold,
            score=0.0,
            passed=False,
            weights_used=dict(weights),
            notes=["empty frame"],
        )

    conditions = _directional_conditions(frame, side)
    applicable = {name: float(weights.get(name, 0.0)) for name in conditions if name in weights}
    weighted_components = {name: applicable.get(name, 0.0) * conditions[name] for name in applicable}
    score = sum(weighted_components.values())
    notes: list[str] = []
    if regime == Regime.TRENDING:
        notes.append("using trending weight map")
    elif regime == Regime.RANGING:
        notes.append("using ranging weight map")
    else:
        notes.append("regime normally blocks entries")

    return ConfluenceDetail(
        regime=regime,
        side=side,
        threshold=threshold,
        score=score,
        passed=score >= threshold,
        weights_used=applicable,
        condition_values=conditions,
        weighted_components=weighted_components,
        notes=notes,
    )


def score_long_confluence(frame: pd.DataFrame, regime: Regime, threshold: float) -> ConfluenceResult:
    """Score long-side entry confluence for the current regime."""
    weights = _resolve_weights(regime)
    detail = explain_confluence(frame, regime, weights, side="Buy", threshold=threshold)
    return ConfluenceResult(score=detail.score, passed=detail.passed)



def score_short_confluence(frame: pd.DataFrame, regime: Regime, threshold: float) -> ConfluenceResult:
    """Score short-side entry confluence for the current regime."""
    weights = _resolve_weights(regime)
    detail = explain_confluence(frame, regime, weights, side="Sell", threshold=threshold)
    return ConfluenceResult(score=detail.score, passed=detail.passed)


def score_confluence(
    frame: pd.DataFrame,
    regime: Regime,
    weights: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> ConfluenceResult:
    """Compute a weighted score from indicator conditions.

    The original API is preserved.  It infers direction from trend bias and uses
    the appropriate regime-specific weight map.
    """

    if regime in {Regime.TRANSITIONAL, Regime.VOLATILITY_SPIKE}:
        return ConfluenceResult(score=0.0, passed=False)
    row = frame.iloc[-1]
    default_side = "Buy" if _float(row.get("close")) >= _float(row.get("ema21")) else "Sell"
    use_weights = _resolve_weights(regime, weights)
    detail = explain_confluence(frame, regime, use_weights, side=default_side, threshold=threshold)
    return ConfluenceResult(score=detail.score, passed=detail.passed)
