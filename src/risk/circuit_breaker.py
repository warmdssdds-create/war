"""Circuit-breaker controls, drawdown tracking, and trade statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.risk.position_sizing import apply_volatility_stop_tightening
from src.utils.time_utils import format_ts, utc_now


@dataclass
class TradeStats:
    """Track realized trade outcomes and derived metrics."""

    wins: int = 0
    losses: int = 0
    total: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf") if self.gross_profit > 0 else 0.0

    @property
    def expectancy(self) -> float:
        return self.net_pnl / self.total if self.total else 0.0

    def update(self, pnl: float) -> None:
        """Update statistics from a new realized trade."""
        self.total += 1
        self.net_pnl += pnl
        if pnl >= 0:
            self.wins += 1
            self.gross_profit += pnl
        else:
            self.losses += 1
            self.gross_loss += abs(pnl)


@dataclass
class DrawdownTracker:
    """Track an equity curve and drawdown statistics."""

    starting_equity: float
    equity: float | None = None
    peak: float | None = None
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.equity = self.starting_equity if self.equity is None else self.equity
        self.peak = self.starting_equity if self.peak is None else self.peak
        self.history.append(self.equity)

    def update(self, pnl: float) -> float:
        """Apply realized PnL and return the updated drawdown percentage."""
        self.equity = float(self.equity or 0.0) + pnl
        self.peak = max(float(self.peak or 0.0), self.equity)
        if self.peak > 0:
            self.current_drawdown_pct = max((self.peak - self.equity) / self.peak, 0.0)
        self.max_drawdown_pct = max(self.max_drawdown_pct, self.current_drawdown_pct)
        self.history.append(self.equity)
        return self.current_drawdown_pct

    def reset(self) -> None:
        """Reset drawdown stats while keeping the current equity as baseline."""
        self.peak = self.equity
        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.history = [float(self.equity or 0.0)]


@dataclass
class CircuitBreakerStatus:
    """Human-readable snapshot of circuit-breaker state."""

    paused: bool
    paused_until: str | None
    daily_pnl: float
    weekly_pnl: float
    consecutive_losses: int
    drawdown_pct: float
    win_rate: float
    profit_factor: float
    note: str


class VolatilityStopTightener:
    """Tighten stops when the strategy detects an active volatility spike."""

    def __init__(self) -> None:
        self.is_spike_active = False

    def set_spike(self, active: bool) -> None:
        """Update whether a volatility spike is currently active."""
        self.is_spike_active = active

    def apply(self, current_stop: float, entry: float, atr: float, side: str) -> float:
        """Apply a 0.5 ATR tightening when spike mode is active."""
        if not self.is_spike_active:
            return current_stop
        return apply_volatility_stop_tightening(
            current_stop=current_stop,
            atr=atr,
            entry=entry,
            side=side,
            vol_ratio=2.0,
            threshold=1.5,
        )


@dataclass
class CircuitBreaker:
    """Track realized PnL and enforce pause windows."""

    daily_loss_limit: float
    weekly_loss_limit: float
    max_consecutive_losses: int
    pause_minutes: int
    starting_equity: float
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    paused_until: datetime | None = field(default=None)
    drawdown_tracker: DrawdownTracker | None = None
    trade_stats: TradeStats = field(default_factory=TradeStats)

    def __post_init__(self) -> None:
        if self.drawdown_tracker is None:
            self.drawdown_tracker = DrawdownTracker(starting_equity=self.starting_equity)

    def is_paused(self) -> bool:
        """Whether trading is currently paused."""
        return bool(self.paused_until and utc_now() < self.paused_until)

    def pause(self, reason: str = "manual") -> None:
        """Trigger a pause window immediately."""
        _ = reason
        self.paused_until = utc_now() + timedelta(minutes=self.pause_minutes)

    def register_trade_result(self, pnl: float) -> None:
        """Update breaker state after each closed trade."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0
        self.trade_stats.update(pnl)
        if self.drawdown_tracker is not None:
            self.drawdown_tracker.update(pnl)

        daily_limit = -self.starting_equity * self.daily_loss_limit
        weekly_limit = -self.starting_equity * self.weekly_loss_limit
        current_dd = self.drawdown_tracker.current_drawdown_pct if self.drawdown_tracker is not None else 0.0

        tripped = (
            self.daily_pnl <= daily_limit
            or self.weekly_pnl <= weekly_limit
            or self.consecutive_losses >= self.max_consecutive_losses
            or current_dd >= max(self.daily_loss_limit * 1.5, 0.05)
        )
        if tripped:
            self.paused_until = utc_now() + timedelta(minutes=self.pause_minutes)

    def reset_daily(self) -> None:
        """Reset daily stats."""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.paused_until = None

    def reset_weekly(self) -> None:
        """Reset weekly stats."""
        self.weekly_pnl = 0.0

    def status(self) -> CircuitBreakerStatus:
        """Return a human-readable status payload."""
        paused = self.is_paused()
        paused_until = format_ts(self.paused_until) if self.paused_until else None
        drawdown_pct = self.drawdown_tracker.current_drawdown_pct if self.drawdown_tracker is not None else 0.0
        note = "paused" if paused else "active"
        return CircuitBreakerStatus(
            paused=paused,
            paused_until=paused_until,
            daily_pnl=self.daily_pnl,
            weekly_pnl=self.weekly_pnl,
            consecutive_losses=self.consecutive_losses,
            drawdown_pct=drawdown_pct,
            win_rate=self.trade_stats.win_rate,
            profit_factor=self.trade_stats.profit_factor,
            note=note,
        )
