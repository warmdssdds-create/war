"""Event-driven backtest engine for the strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators.compute import compute_indicators
from src.strategy.signals import evaluate_signal


@dataclass
class BacktestMetrics:
    """Key performance metrics."""

    win_rate: float
    profit_factor: float
    max_drawdown: float
    expectancy: float


def run_backtest(
    candles_15m: pd.DataFrame,
    weights: dict,
    threshold: float,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestMetrics:
    """Simulate one-candle hold trades from signal events."""
    df = compute_indicators(candles_15m).dropna().reset_index(drop=True)
    if len(df) < 3:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0)

    pnl_series: list[float] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for i in range(1, len(df) - 1):
        window = df.iloc[: i + 1]
        signal = evaluate_signal(window, weights=weights, threshold=threshold, confirmed=True)
        if not signal.side:
            continue

        entry = float(df.iloc[i]["close"])
        exit_price = float(df.iloc[i + 1]["close"])
        direction = 1 if signal.side == "Buy" else -1
        gross = (exit_price - entry) * direction
        costs = entry * ((fee_bps + slippage_bps) / 10_000) * 2
        net = gross - costs

        pnl_series.append(net)
        equity += net
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    if not pnl_series:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0)

    wins = [x for x in pnl_series if x > 0]
    losses = [x for x in pnl_series if x < 0]
    win_rate = len(wins) / len(pnl_series)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy = sum(pnl_series) / len(pnl_series)

    return BacktestMetrics(
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=abs(max_dd),
        expectancy=expectancy,
    )
