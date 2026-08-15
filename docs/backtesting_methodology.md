# Backtesting Methodology

This document explains the design of the event-driven backtester (`src/backtest.py`), its assumptions, limitations, and how to interpret results.

---

## Overview

The backtester replays historical OHLCV candles through the same indicator and signal pipeline used in live trading, simulating order fills and tracking portfolio equity.

---

## Event-Driven Design

```
historical candles (sorted by time)
         │
         ▼ for each bar
compute_indicators(window_df)
         │
         ▼
evaluate_signal()
         │
         ▼  signal.side not None
compute_trade_levels()     → qty, stop_loss, take_profit
         │
         ▼
simulate fill at next bar's open
         │
         ▼
manage open position:
  ├── hit stop_loss?   → close (loss)
  ├── hit take_profit? → close (win)
  └── trailing stop update (optional)
         │
         ▼
record trade in journal + update equity curve
```

The key property of event-driven replay is that **each signal can only see data available at that bar's close**, preventing look-ahead bias.

---

## Assumptions and Limitations

| Assumption | Detail |
|------------|--------|
| Fill model | Orders fill at the **next bar's open** price.  This is optimistic for fast-moving markets. |
| Slippage | A configurable flat slippage (default 0.05%) is applied to each fill. |
| Fees | Taker fee (default 0.055%) deducted on entry and exit. |
| Partial fills | Not modelled; full fill assumed. |
| Funding rate | Not deducted.  For long holding periods, add expected funding cost manually. |
| Liquidation | Not simulated; position sizing limits should prevent margin calls in practice. |
| Market impact | Not modelled.  Results may not scale to large position sizes. |
| Spread | Not applied during backtesting (only enforced in live mode). |

> ⚠️  **Backtest results are not indicative of future performance.**  Historical patterns may not repeat.  Always forward-test on a paper/testnet account before committing real capital.

---

## CLI Usage

```bash
# Run backtest with default config
python -m src.backtest \
  --symbol ETCUSDT \
  --interval 15 \
  --start 2024-01-01 \
  --end 2024-06-01

# Override risk settings inline
python -m src.backtest \
  --symbol ETCUSDT \
  --interval 15 \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --risk-per-trade 0.01 \
  --atr-multiple 1.5 \
  --reward-risk 2.0

# Save output report
python -m src.backtest ... --report backtest_results.json
```

---

## Metrics Reported

| Metric | Description |
|--------|-------------|
| `total_trades` | Total number of closed trades |
| `win_rate` | Fraction of trades with positive PnL |
| `profit_factor` | Gross profit / gross loss |
| `total_return_pct` | Final equity vs starting equity |
| `max_drawdown_pct` | Peak-to-trough equity drawdown |
| `sharpe_ratio` | Annualised Sharpe ratio (daily returns, risk-free = 0) |
| `avg_trade_pnl` | Mean PnL per trade in USDT |
| `avg_bars_held` | Mean number of bars a position was held |

---

## Walk-Forward Validation

To reduce overfitting, use multiple non-overlapping test windows:

```bash
# In-sample: 2023-01-01 – 2023-09-30
python -m src.backtest --start 2023-01-01 --end 2023-09-30 --report is.json

# Out-of-sample: 2023-10-01 – 2024-03-31
python -m src.backtest --start 2023-10-01 --end 2024-03-31 --report oos.json
```

A strategy whose out-of-sample metrics are substantially worse than in-sample is likely overfitted.

---

## Optimisation Warning

The backtester intentionally does **not** include a parameter optimiser.  Running grid searches over indicator parameters on the same data used to evaluate them inflates apparent performance (data-snooping bias).  If you optimise parameters, validate on a separate held-out period.

---

## Data Source

Historical candles are fetched from the Bybit REST API (`/v5/market/kline`) and cached locally as Parquet files.  The backtester reads from the cache when available, avoiding redundant API calls.

---

## Reproducibility

All backtests are deterministic given the same candle data and configuration.  To reproduce a run:

1. Keep the Parquet cache from the original run.
2. Use the same `config/strategy.yaml` and `config/risk.yaml`.
3. Re-run the same CLI command.

Results are written to a timestamped JSON file so multiple runs can be compared.
