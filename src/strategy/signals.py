"""Signal evaluation pipeline and position-exit helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.risk.position_sizing import compute_trailing_stop
from src.strategy.confluence import score_confluence, score_long_confluence, score_short_confluence
from src.strategy.regime_filter import Regime, classify_regime, filter_regime_for_entry
from src.utils.time_utils import utc_now


@dataclass
class SignalResult:
    """Signal outcome for the execution engine."""

    side: str | None
    regime: Regime
    score: float
    reason: str


@dataclass
class ExitSignalResult:
    """Exit decision payload for active positions."""

    should_exit: bool
    reason: str
    new_stop: float | None


@dataclass
class HistoricalSignal:
    """Single historical signal event."""

    side: str | None
    score: float
    regime: Regime
    won: bool | None
    reason: str
    timestamp: datetime


class SignalHistory:
    """Track the latest signals and derived hit-rate statistics."""

    def __init__(self, maxlen: int = 20) -> None:
        self._signals: deque[HistoricalSignal] = deque(maxlen=maxlen)

    def add(
        self,
        side: str | None,
        score: float,
        regime: Regime,
        reason: str,
        won: bool | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a signal event to history."""
        self._signals.append(
            HistoricalSignal(
                side=side,
                score=score,
                regime=regime,
                won=won,
                reason=reason,
                timestamp=timestamp or utc_now(),
            )
        )

    def recent_signals(self, n: int = 5) -> list[HistoricalSignal]:
        """Return the latest ``n`` signals."""
        if n <= 0:
            return []
        return list(self._signals)[-n:]

    def win_rate(self) -> float:
        """Return historical win rate, ignoring open/unknown outcomes."""
        decided = [signal for signal in self._signals if signal.won is not None]
        if not decided:
            return 0.0
        wins = sum(1 for signal in decided if signal.won)
        return wins / len(decided)

    def __len__(self) -> int:
        return len(self._signals)


def apply_5m_refinement(frame_5m: pd.DataFrame, side: str) -> bool:
    """Use 5m momentum confirmation as an entry refinement gate."""
    if frame_5m is None or frame_5m.empty:
        return True
    row = frame_5m.iloc[-1]
    side = side.capitalize()
    macd_hist = float(row.get("macd_hist", 0.0))
    ema21 = float(row.get("ema21", row.get("close", 0.0)))
    close = float(row.get("close", 0.0))
    rsi = float(row.get("rsi14", 50.0))
    if side == "Buy":
        return macd_hist > 0 and close >= ema21 and rsi >= 45.0
    if side == "Sell":
        return macd_hist < 0 and close <= ema21 and rsi <= 55.0
    return False


def _entry_side_from_frame(frame_15m: pd.DataFrame, regime: Regime, threshold: float) -> tuple[str | None, float, str]:
    """Infer a trade direction from indicator state and regime."""
    long_score = score_long_confluence(frame_15m, regime, threshold)
    short_score = score_short_confluence(frame_15m, regime, threshold)
    if long_score.passed and short_score.passed:
        if long_score.score >= short_score.score:
            return "Buy", long_score.score, "long_edge"
        return "Sell", short_score.score, "short_edge"
    if long_score.passed:
        return "Buy", long_score.score, "long_edge"
    if short_score.passed:
        return "Sell", short_score.score, "short_edge"
    return None, max(long_score.score, short_score.score), "low_confluence"


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
    if regime == Regime.VOLATILITY_SPIKE:
        return SignalResult(side=None, regime=regime, score=0.0, reason="volatility_pause")

    compatibility = score_confluence(frame_15m, regime, weights, threshold)
    side, directional_score, reason = _entry_side_from_frame(frame_15m, regime, threshold)
    score = max(compatibility.score, directional_score)
    if not side or not compatibility.passed and directional_score < threshold:
        return SignalResult(side=None, regime=regime, score=score, reason="low_confluence")
    if not filter_regime_for_entry(regime, side):
        return SignalResult(side=None, regime=regime, score=score, reason="regime_filter")

    if use_refinement and frame_5m is not None and not frame_5m.empty and not apply_5m_refinement(frame_5m, side):
        return SignalResult(side=None, regime=regime, score=score, reason="5m_refine_reject")

    return SignalResult(side=side, regime=regime, score=score, reason=reason)


def evaluate_exit_signal(
    frame_15m: pd.DataFrame,
    position_side: str,
    entry_price: float,
    current_stop: float,
    regime: Regime,
) -> ExitSignalResult:
    """Evaluate whether an active position should exit or tighten its stop."""
    if frame_15m.empty:
        return ExitSignalResult(False, "no_data", current_stop)

    row = frame_15m.iloc[-1]
    side = position_side.capitalize()
    close = float(row.get("close", entry_price))
    high = float(row.get("high", close))
    low = float(row.get("low", close))
    atr = max(float(row.get("atr14", 0.0)), 1e-9)
    macd_hist = float(row.get("macd_hist", 0.0))
    rsi = float(row.get("rsi14", 50.0))

    if side == "Buy" and low <= current_stop:
        return ExitSignalResult(True, "stop_hit", current_stop)
    if side == "Sell" and high >= current_stop:
        return ExitSignalResult(True, "stop_hit", current_stop)

    if regime == Regime.VOLATILITY_SPIKE:
        tightened = compute_trailing_stop(side, entry_price, close, current_stop, atr, trail_mult=1.0)
        return ExitSignalResult(False, "volatility_tighten", tightened)

    if side == "Buy":
        if macd_hist < 0 and rsi < 45:
            return ExitSignalResult(True, "momentum_reversal", current_stop)
        new_stop = compute_trailing_stop(side, entry_price, max(close, high), current_stop, atr, trail_mult=1.5)
        if close < float(row.get("ema21", close)) and close > entry_price:
            new_stop = max(new_stop, entry_price)
        return ExitSignalResult(False, "hold", new_stop)

    if macd_hist > 0 and rsi > 55:
        return ExitSignalResult(True, "momentum_reversal", current_stop)
    new_stop = compute_trailing_stop(side, entry_price, min(close, low), current_stop, atr, trail_mult=1.5)
    if close > float(row.get("ema21", close)) and close < entry_price:
        new_stop = min(new_stop, entry_price)
    return ExitSignalResult(False, "hold", new_stop)
