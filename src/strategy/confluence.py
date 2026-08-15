"""Confluence scoring for entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.strategy.regime_filter import Regime


@dataclass
class ConfluenceResult:
    """Scoring result."""

    score: float
    passed: bool


def score_confluence(
    frame: pd.DataFrame,
    regime: Regime,
    weights: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> ConfluenceResult:
    """Compute weighted score from indicator conditions."""
    row = frame.iloc[-1]
    trend_map = weights.get("trending", {})
    range_map = weights.get("ranging", {})
    if regime in {Regime.TRANSITIONAL, Regime.VOLATILITY_SPIKE}:
        return ConfluenceResult(score=0.0, passed=False)
    use_map = trend_map if regime == Regime.TRENDING else range_map

    conditions = {
        "ema_alignment": float(row["ema21"] > row["ema55"] > row["ema200"]),
        "adx_strength": float(row["adx14"] >= 25),
        "macd_momentum": float(row["macd_hist"] > 0),
        "rsi_bias": float(row["rsi14"] >= 50),
        "vwap_position": float(row["close"] >= row["vwap"]),
        "obv_trend": float(row["obv"] >= frame["obv"].iloc[-2] if len(frame) > 1 else 0),
        "volume_confirmation": float(row["volume_zscore"] > 0),
        "bb_reversion": float(row["close"] <= row["bb_lower"] or row["close"] >= row["bb_upper"]),
        "stochrsi_extreme": float(row["stochrsi_k"] <= 20 or row["stochrsi_k"] >= 80),
        "rsi_reversion": float(row["rsi14"] <= 35 or row["rsi14"] >= 65),
        "vwap_revert": float(abs(row["close"] - row["vwap"]) / max(row["close"], 1e-9) > 0.002),
        "atr_compression": float(row["atr14"] <= row["atr_avg100"]),
    }

    score = sum(float(use_map.get(name, 0.0)) * value for name, value in conditions.items())
    return ConfluenceResult(score=score, passed=score >= threshold)
