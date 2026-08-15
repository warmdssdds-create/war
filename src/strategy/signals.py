"""Signal evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.confluence import score_confluence
from src.strategy.regime_filter import Regime, classify_regime


@dataclass
class SignalResult:
    """Signal outcome for execution engine."""

    side: str | None
    regime: Regime
    score: float
    reason: str


def evaluate_signal(
    frame_15m: pd.DataFrame,
    weights: dict,
    threshold: float,
    confirmed: bool,
    frame_5m: pd.DataFrame | None = None,
    use_refinement: bool = False,
) -> SignalResult:
    """Evaluate entry only on confirmed 15m candle close."""
    if frame_15m.empty or not confirmed:
        return SignalResult(side=None, regime=Regime.TRANSITIONAL, score=0.0, reason="unconfirmed")

    regime = classify_regime(frame_15m)
    conf = score_confluence(frame_15m, regime, weights, threshold)
    if not conf.passed:
        return SignalResult(side=None, regime=regime, score=conf.score, reason="low_confluence")

    row = frame_15m.iloc[-1]
    side = "Buy" if row["close"] >= row["ema21"] else "Sell"

    if use_refinement and frame_5m is not None and not frame_5m.empty:
        r5 = frame_5m.iloc[-1]
        if side == "Buy" and r5.get("macd_hist", 0.0) <= 0:
            return SignalResult(side=None, regime=regime, score=conf.score, reason="5m_refine_reject")
        if side == "Sell" and r5.get("macd_hist", 0.0) >= 0:
            return SignalResult(side=None, regime=regime, score=conf.score, reason="5m_refine_reject")

    if regime == Regime.VOLATILITY_SPIKE:
        return SignalResult(side=None, regime=regime, score=conf.score, reason="volatility_pause")

    return SignalResult(side=side, regime=regime, score=conf.score, reason="ok")
