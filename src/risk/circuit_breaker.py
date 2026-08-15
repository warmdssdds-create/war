"""Circuit breaker controls for loss limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.utils.time_utils import utc_now


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

    def is_paused(self) -> bool:
        """Whether trading is currently paused."""
        return bool(self.paused_until and utc_now() < self.paused_until)

    def register_trade_result(self, pnl: float) -> None:
        """Update breaker state after each closed trade."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.consecutive_losses = self.consecutive_losses + 1 if pnl < 0 else 0

        daily_limit = -self.starting_equity * self.daily_loss_limit
        weekly_limit = -self.starting_equity * self.weekly_loss_limit

        tripped = (
            self.daily_pnl <= daily_limit
            or self.weekly_pnl <= weekly_limit
            or self.consecutive_losses >= self.max_consecutive_losses
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
