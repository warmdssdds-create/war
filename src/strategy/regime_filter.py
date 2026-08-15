"""Market regime classification and regime-history helpers."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import pandas as pd

from src.utils.time_utils import format_ts, utc_now


class Regime(StrEnum):
    """Supported market regimes for signal selection."""

    TRENDING = "TRENDING"
    RANGING = "RANGING"
    TRANSITIONAL = "TRANSITIONAL"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"


@dataclass(frozen=True)
class RegimePoint:
    """A single observed regime and the time it was recorded."""

    regime: Regime
    timestamp: datetime


class RegimeHistory:
    """Store the latest N classified regimes with timestamps."""

    def __init__(self, maxlen: int = 100) -> None:
        self.maxlen = maxlen
        self._items: deque[RegimePoint] = deque(maxlen=maxlen)

    def add(self, regime: Regime, timestamp: datetime | None = None) -> None:
        """Append a regime observation to the rolling history."""
        self._items.append(RegimePoint(regime=Regime(regime), timestamp=timestamp or utc_now()))

    def extend(self, regimes: list[Regime]) -> None:
        """Append multiple regimes using the current timestamp."""
        for regime in regimes:
            self.add(regime)

    def last(self) -> RegimePoint | None:
        """Return the most recent regime entry if present."""
        return self._items[-1] if self._items else None

    def recent(self, n: int = 5) -> list[RegimePoint]:
        """Return the most recent ``n`` observations."""
        if n <= 0:
            return []
        return list(self._items)[-n:]

    def most_common(self, n: int) -> Regime | None:
        """Return the most common regime across the latest ``n`` values."""
        recent = [item.regime for item in self.recent(n)]
        if not recent:
            return None
        return Counter(recent).most_common(1)[0][0]

    def counts(self, n: int | None = None) -> dict[Regime, int]:
        """Return a frequency map for the history or latest ``n`` values."""
        items = [item.regime for item in (self.recent(n) if n is not None else self._items)]
        counter = Counter(items)
        return {Regime(key): int(value) for key, value in counter.items()}

    def summary(self, n: int = 10) -> str:
        """Return a small human-readable summary string."""
        recent = self.recent(n)
        if not recent:
            return "no regimes recorded"
        common = self.most_common(n)
        latest = recent[-1]
        return f"latest={latest.regime} at {format_ts(latest.timestamp)}, common={common}, count={len(recent)}"

    def __len__(self) -> int:
        return len(self._items)


def classify_regime(frame: pd.DataFrame) -> Regime:
    """Classify regime using ADX and ATR expansion.

    The original behavior is preserved while additional columns like ``plus_di``
    and ``minus_di`` may refine the transitional band.
    """

    row = frame.iloc[-1]
    adx = float(row.get("adx14", 0.0))
    atr = float(row.get("atr14", 0.0))
    atr_avg = float(row.get("atr_avg100", atr if atr > 0 else 1e-9))
    plus_di = float(row.get("plus_di", 0.0))
    minus_di = float(row.get("minus_di", 0.0))

    if atr_avg > 0 and atr / atr_avg >= 1.8:
        return Regime.VOLATILITY_SPIKE
    if adx >= 25 and abs(plus_di - minus_di) >= 3:
        return Regime.TRENDING
    if adx >= 25:
        return Regime.TRENDING
    if adx <= 18:
        return Regime.RANGING
    return Regime.TRANSITIONAL


def is_regime_stable(history: RegimeHistory, n: int = 5, required: str | Regime = "TRENDING") -> bool:
    """Check whether the latest ``n`` regime values consistently match ``required``."""
    required_regime = Regime(required)
    recent = history.recent(n)
    if len(recent) < n:
        return False
    return all(item.regime == required_regime for item in recent)


def get_regime_description(regime: str | Regime) -> str:
    """Return a human-readable explanation for a regime label."""
    regime_value = Regime(regime)
    descriptions = {
        Regime.TRENDING: "Directional market with strong momentum and better breakout follow-through.",
        Regime.RANGING: "Sideways market with mean-reversion tendencies and lower directional conviction.",
        Regime.TRANSITIONAL: "Mixed market state where signals are less reliable and risk should be reduced.",
        Regime.VOLATILITY_SPIKE: "Abnormally volatile market where the strategy pauses new entries.",
    }
    return descriptions[regime_value]


def filter_regime_for_entry(regime: str | Regime, position_side: str) -> bool:
    """Return whether a position side is allowed in the current regime.

    Trending regimes allow both long and short entries, but ranging markets are
    stricter: longs require price-strength confirmation upstream and shorts
    require weakness confirmation.  The function therefore only blocks clearly
    unsafe regimes at this stage and leaves exact scoring to confluence logic.
    """

    regime_value = Regime(regime)
    side = position_side.capitalize()
    if regime_value in {Regime.VOLATILITY_SPIKE, Regime.TRANSITIONAL}:
        return False
    if side not in {"Buy", "Sell"}:
        return False
    return True
