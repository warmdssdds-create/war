"""Comprehensive event-driven backtest engine for the ETCUSDT strategy."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.indicators.compute import compute_indicators
from src.risk.position_sizing import PositionSizingConfig, compute_trade_levels_v2, compute_trailing_stop
from src.strategy.confluence import RANGING_WEIGHTS, TRENDING_WEIGHTS
from src.strategy.regime_filter import Regime, classify_regime
from src.strategy.signals import ExitSignalResult, evaluate_exit_signal, evaluate_signal


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    symbol: str
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    fee_bps: float = 6.0
    slippage_bps: float = 1.0
    funding_rate_8h: float = 0.0001


@dataclass
class Trade:
    """Represents one simulated trade lifecycle."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    side: str
    entry_price: float
    exit_price: float | None
    qty: float
    gross_pnl: float
    fees: float
    funding_cost: float
    net_pnl: float
    exit_reason: str
    stop_loss: float
    take_profit: float
    regime: str = ""
    max_favourable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None


@dataclass
class BacktestResult:
    """Aggregate backtest metrics and artifacts."""

    symbol: str
    start: str
    end: str
    initial_capital: float
    ending_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    recovery_factor: float
    total_return_pct: float
    total_fees: float
    total_funding_cost: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class EventDrivenBacktest:
    """Event-driven backtest that consumes prepared candle data."""

    def __init__(self, config: BacktestConfig, weights: dict, threshold: float) -> None:
        self.config = config
        self.weights = weights or {"trending": TRENDING_WEIGHTS, "ranging": RANGING_WEIGHTS}
        self.threshold = threshold
        self.capital = config.initial_capital
        self.trades: list[Trade] = []
        self.current_trade: Trade | None = None
        self.equity_curve: list[float] = [config.initial_capital]

    def run(self, candles_15m: pd.DataFrame, candles_5m: pd.DataFrame | None = None) -> BacktestResult:
        """Run the event-driven simulation across all candles."""
        df = compute_indicators(candles_15m).dropna().reset_index(drop=True)
        df5 = compute_indicators(candles_5m).dropna().reset_index(drop=True) if candles_5m is not None and not candles_5m.empty else None
        if self.config.start_date:
            df = df.loc[df["start"] >= pd.Timestamp(self.config.start_date, tz="UTC")].reset_index(drop=True)
        if self.config.end_date:
            df = df.loc[df["start"] <= pd.Timestamp(self.config.end_date, tz="UTC")].reset_index(drop=True)
        for i in range(len(df)):
            self._process_candle(i, df, df5)
        if self.current_trade is not None:
            self._simulate_exit(self.current_trade, df.iloc[-1], reason="end_of_test")
            self.current_trade = None
        return compute_metrics(self.trades, self.config.initial_capital, self.config.symbol)

    def _window5m(self, candle_time: pd.Timestamp, frame_5m: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame_5m is None or frame_5m.empty:
            return None
        subset = frame_5m.loc[frame_5m["start"] <= candle_time]
        return subset.tail(50).reset_index(drop=True) if not subset.empty else None

    def _process_candle(self, i: int, df: pd.DataFrame, df5: pd.DataFrame | None = None) -> None:
        """Process one candle: exits first, then optional new entry."""
        candle = df.iloc[i]
        window = df.iloc[: i + 1]
        if self.current_trade is not None:
            self._manage_open_trade(window, candle)
        if self.current_trade is not None:
            self.equity_curve.append(self.capital + self.current_trade.net_pnl)
            return
        if len(window) < 30:
            self.equity_curve.append(self.capital)
            return
        signal = evaluate_signal(
            frame_15m=window,
            weights=self.weights,
            threshold=self.threshold,
            confirmed=True,
            frame_5m=self._window5m(candle["start"], df5),
            use_refinement=df5 is not None,
        )
        if signal.side is not None:
            self._simulate_entry(signal, candle)
        self.equity_curve.append(self.capital)

    def _simulate_entry(self, signal: object, candle: pd.Series) -> None:
        """Open a simulated trade with slippage and dynamic levels."""
        side = str(signal.side)
        entry = float(candle["close"])
        slip = entry * (self.config.slippage_bps / 10_000.0)
        entry_fill = entry + slip if side == "Buy" else entry - slip
        sizing = PositionSizingConfig(
            capital=self.capital,
            risk_per_trade=self.config.risk_per_trade,
            atr_multiple=1.5,
            reward_risk=1.8,
            qty_step=0.1,
            min_qty=0.1,
            max_notional=max(self.capital * 3, 100.0),
            fee_bps=self.config.fee_bps,
            slippage_bps=self.config.slippage_bps,
        )
        structure_stop = float(candle["low"] if side == "Buy" else candle["high"])
        levels = compute_trade_levels_v2(entry_fill, side, sizing, float(candle["atr14"]), structure_stop)
        if levels.qty <= 0:
            return
        self.current_trade = Trade(
            entry_time=pd.Timestamp(candle["start"]),
            exit_time=None,
            side=side,
            entry_price=entry_fill,
            exit_price=None,
            qty=levels.qty,
            gross_pnl=0.0,
            fees=entry_fill * levels.qty * (self.config.fee_bps / 10_000.0),
            funding_cost=0.0,
            net_pnl=0.0,
            exit_reason="open",
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            regime=str(signal.regime),
        )

    def _simulate_exit(self, trade: Trade, candle: pd.Series, reason: str) -> Trade:
        """Close an open trade and crystallize PnL."""
        exit_price = float(candle["close"])
        if reason == "stop_hit":
            exit_price = trade.stop_loss
        elif reason == "take_profit":
            exit_price = trade.take_profit
        slip = exit_price * (self.config.slippage_bps / 10_000.0)
        exit_fill = exit_price - slip if trade.side == "Buy" else exit_price + slip
        direction = 1 if trade.side == "Buy" else -1
        gross = (exit_fill - trade.entry_price) * trade.qty * direction
        exit_fee = exit_fill * trade.qty * (self.config.fee_bps / 10_000.0)
        hours_held = max((pd.Timestamp(candle["start"]) - trade.entry_time).total_seconds() / 3600.0, 0.0)
        funding_cost = self._apply_funding_cost(trade, hours_held)
        net = gross - trade.fees - exit_fee - funding_cost
        trade.exit_time = pd.Timestamp(candle["start"])
        trade.exit_price = exit_fill
        trade.gross_pnl = gross
        trade.fees += exit_fee
        trade.funding_cost += funding_cost
        trade.net_pnl = net
        trade.exit_reason = reason
        self.capital += net
        self.trades.append(trade)
        self.current_trade = None
        return trade

    def _apply_trailing_stop(self, trade: Trade, candle: pd.Series, atr: float) -> None:
        """Update an active trade's trailing stop from candle action."""
        if trade.side == "Buy":
            water_mark = max(float(candle["high"]), trade.entry_price)
            trade.max_favourable_excursion = max(trade.max_favourable_excursion, water_mark - trade.entry_price)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, float(candle["low"]) - trade.entry_price)
        else:
            water_mark = min(float(candle["low"]), trade.entry_price)
            trade.max_favourable_excursion = max(trade.max_favourable_excursion, trade.entry_price - water_mark)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, trade.entry_price - float(candle["high"]))
        trade.stop_loss = compute_trailing_stop(trade.side, trade.entry_price, water_mark, trade.stop_loss, atr, 1.5)

    def _apply_funding_cost(self, trade: Trade, hours_held: float) -> float:
        """Estimate funding cost proportional to time held."""
        if hours_held <= 0:
            return 0.0
        intervals = hours_held / 8.0
        notional = trade.entry_price * trade.qty
        return abs(notional * self.config.funding_rate_8h * intervals)

    def _manage_open_trade(self, window: pd.DataFrame, candle: pd.Series) -> None:
        """Evaluate exit logic for the currently open trade."""
        assert self.current_trade is not None
        trade = self.current_trade
        self._apply_trailing_stop(trade, candle, float(candle["atr14"]))

        if trade.side == "Buy":
            if float(candle["low"]) <= trade.stop_loss:
                self._simulate_exit(trade, candle, "stop_hit")
                return
            if float(candle["high"]) >= trade.take_profit:
                self._simulate_exit(trade, candle, "take_profit")
                return
        else:
            if float(candle["high"]) >= trade.stop_loss:
                self._simulate_exit(trade, candle, "stop_hit")
                return
            if float(candle["low"]) <= trade.take_profit:
                self._simulate_exit(trade, candle, "take_profit")
                return

        regime = classify_regime(window)
        exit_signal: ExitSignalResult = evaluate_exit_signal(window, trade.side, trade.entry_price, trade.stop_loss, regime)
        if exit_signal.should_exit:
            self._simulate_exit(trade, candle, exit_signal.reason)
        elif exit_signal.new_stop is not None:
            trade.stop_loss = exit_signal.new_stop



def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0



def compute_metrics(trades: list[Trade], initial_capital: float, symbol: str = "ETCUSDT") -> BacktestResult:
    """Compute a full metrics report from the trade list."""
    pnls = [trade.net_pnl for trade in trades]
    equity = [initial_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 0.0
    expectancy = float(np.mean(pnls)) if pnls else 0.0

    peak = equity[0]
    max_drawdown = 0.0
    drawdowns: list[float] = []
    for value in equity:
        peak = max(peak, value)
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        drawdowns.append(drawdown)
        max_drawdown = max(max_drawdown, drawdown)

    returns = np.diff(equity) / np.maximum(np.array(equity[:-1]), 1e-9)
    mean_return = float(np.mean(returns)) if len(returns) else 0.0
    std_return = _safe_std(list(returns))
    downside = [value for value in returns if value < 0]
    downside_std = _safe_std(downside)
    sharpe_ratio = (mean_return / std_return) * math.sqrt(252) if std_return > 0 else 0.0
    sortino_ratio = (mean_return / downside_std) * math.sqrt(252) if downside_std > 0 else 0.0
    ending_capital = equity[-1]
    total_return_pct = ((ending_capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
    recovery_factor = (total_return_pct / (max_drawdown * 100)) if max_drawdown > 0 else 0.0
    total_fees = sum(trade.fees for trade in trades)
    total_funding = sum(trade.funding_cost for trade in trades)
    start = str(trades[0].entry_time) if trades else ""
    end = str(trades[-1].exit_time or trades[-1].entry_time) if trades else ""
    return BacktestResult(
        symbol=symbol,
        start=start,
        end=end,
        initial_capital=initial_capital,
        ending_capital=ending_capital,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        recovery_factor=recovery_factor,
        total_return_pct=total_return_pct,
        total_fees=total_fees,
        total_funding_cost=total_funding,
        trades=trades,
        equity_curve=equity,
    )


class WalkForwardSplit:
    """Yield rolling train/test splits for walk-forward analysis."""

    def __init__(self, n_splits: int, train_pct: float = 0.7) -> None:
        self.n_splits = max(1, n_splits)
        self.train_pct = min(max(train_pct, 0.5), 0.95)

    def split(self, df: pd.DataFrame) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        if df.empty:
            return []
        window = max(len(df) // self.n_splits, 2)
        results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for split_idx in range(self.n_splits):
            start = split_idx * window
            end = min(len(df), start + window)
            subset = df.iloc[start:end]
            if len(subset) < 4:
                continue
            cut = max(int(len(subset) * self.train_pct), 2)
            train_df = subset.iloc[:cut].reset_index(drop=True)
            test_df = subset.iloc[cut:].reset_index(drop=True)
            if test_df.empty:
                continue
            results.append((train_df, test_df))
        return results



def run_walk_forward(
    candles: pd.DataFrame,
    config: BacktestConfig,
    weights: dict,
    threshold: float,
    n_splits: int = 5,
) -> list[BacktestResult]:
    """Run walk-forward backtests across sequential splits."""
    splitter = WalkForwardSplit(n_splits=n_splits, train_pct=0.7)
    results: list[BacktestResult] = []
    for _train_df, test_df in splitter.split(candles):
        engine = EventDrivenBacktest(config=config, weights=weights, threshold=threshold)
        results.append(engine.run(test_df))
    return results



def print_metrics_report(result: BacktestResult) -> None:
    """Print a formatted metrics report to stdout."""
    print(f"Backtest Report: {result.symbol}")
    print(f"Period: {result.start} -> {result.end}")
    print(f"Capital: {result.initial_capital:.2f} -> {result.ending_capital:.2f}")
    print(f"Trades: {result.total_trades} | Win rate: {result.win_rate:.2%}")
    print(f"Profit factor: {result.profit_factor:.3f} | Expectancy: {result.expectancy:.4f}")
    print(f"Avg win: {result.avg_win:.4f} | Avg loss: {result.avg_loss:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.3f} | Sortino: {result.sortino_ratio:.3f}")
    print(f"Recovery factor: {result.recovery_factor:.3f}")
    print(f"Return: {result.total_return_pct:.2f}% | Fees: {result.total_fees:.2f} | Funding: {result.total_funding_cost:.2f}")



def _default_weights() -> dict[str, dict[str, float]]:
    return {"trending": TRENDING_WEIGHTS.copy(), "ranging": RANGING_WEIGHTS.copy()}



def _load_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "start" in frame.columns:
        frame["start"] = pd.to_datetime(frame["start"], utc=True)
    return frame


def _generate_sample_candles(n: int = 500, base_price: float = 20.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 15m OHLCV candles for demo/CI use when no data file is supplied."""
    rng = np.random.default_rng(seed)
    freq = pd.tseries.frequencies.to_offset("15min")
    start_ts = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = [start_ts + i * freq for i in range(n)]  # type: ignore[operator]
    closes = [base_price]
    for _ in range(n - 1):
        closes.append(max(closes[-1] * (1 + rng.normal(0, 0.005)), 0.01))
    closes_arr = np.array(closes)
    highs = closes_arr * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = closes_arr * (1 - np.abs(rng.normal(0, 0.003, n)))
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    volumes = np.abs(rng.normal(5_000, 1_000, n))
    return pd.DataFrame({"start": timestamps, "open": opens, "high": highs, "low": lows, "close": closes_arr, "volume": volumes})


def run_backtest(
    candles_15m: pd.DataFrame,
    candles_5m: pd.DataFrame | None = None,
    symbol: str = "ETCUSDT",
    initial_capital: float = 10_000.0,
    threshold: float = 0.65,
) -> BacktestResult:
    """Run a backtest and return the result.  Convenience wrapper for external callers.

    Example::

        from src.backtest import run_backtest
        result = run_backtest(my_dataframe)
        print(result.win_rate, result.profit_factor)
    """
    config = BacktestConfig(symbol=symbol, initial_capital=initial_capital)
    engine = EventDrivenBacktest(config=config, weights=_default_weights(), threshold=threshold)
    return engine.run(candles_15m, candles_5m)


def main() -> None:
    """CLI entry point for standalone backtests.

    When called with no arguments a synthetic candle dataset is generated
    automatically so that ``run_backtest.bat`` works out of the box.
    """
    parser = argparse.ArgumentParser(description="Run ETCUSDT strategy backtest")
    parser.add_argument("candles", nargs="?", default=None, help="CSV containing 15m candles (omit to use synthetic sample data)")
    parser.add_argument("--candles-5m", default=None, help="Optional CSV containing 5m candles")
    parser.add_argument("--symbol", default="ETCUSDT")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    if args.candles is None:
        print("No candle file supplied — using synthetic sample data (500 × 15m candles).")
        candles_15m = _generate_sample_candles()
    else:
        candles_15m = _load_csv(args.candles)
    candles_5m = _load_csv(args.candles_5m) if args.candles_5m else None
    result = run_backtest(candles_15m, candles_5m, symbol=args.symbol, initial_capital=args.capital, threshold=args.threshold)
    print_metrics_report(result)


if __name__ == "__main__":
    main()
