"""Market regime classification."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class Regime(StrEnum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    TRANSITIONAL = "TRANSITIONAL"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"


def classify_regime(frame: pd.DataFrame) -> Regime:
    """Classify regime using ADX and ATR expansion."""
    row = frame.iloc[-1]
    adx = float(row.get("adx14", 0.0))
    atr = float(row.get("atr14", 0.0))
    atr_avg = float(row.get("atr_avg100", atr if atr > 0 else 1e-9))

    if atr_avg > 0 and atr / atr_avg >= 1.8:
        return Regime.VOLATILITY_SPIKE
    if adx >= 25:
        return Regime.TRENDING
    if adx <= 18:
        return Regime.RANGING
    return Regime.TRANSITIONAL
