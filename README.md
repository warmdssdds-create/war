# Bybit ETCUSDT Algorithmic Trading Bot (Python 3.11+)

Production-oriented, modular trading bot codebase for Bybit Unified Trading with **testnet-first** defaults, risk controls, backtesting, and unit tests.

> ⚠️ **Risk warning:** Trading carries substantial risk. This software is provided for research/engineering purposes only and does **not** guarantee profitability or future returns.

## Architecture

- `src/main.py` live runtime loop
- `src/backtest.py` event-driven candle backtest
- `src/exchange/` Bybit REST/WS wrappers and idempotent order IDs
- `src/data/` candle fetching and parquet cache
- `src/indicators/` EMA/ADX/RSI/MACD/StochRSI/ATR/Bollinger/VWAP/OBV/volume z-score
- `src/strategy/` regime filter, confluence scoring, signal evaluation
- `src/risk/` fixed-fractional sizing + circuit breaker controls
- `src/execution/` spread guard, market execution, reconciliation
- `src/monitoring/` YAML logging and optional Telegram alerts

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with Bybit API credentials (testnet keys recommended).

## Testnet live run

Default config uses testnet mode (`config/risk.yaml -> mode: testnet`).

```bash
python -m src.main
```

## Backtest run

Import `run_backtest` and provide 15m candles as a DataFrame:

```python
from src.backtest import run_backtest
```

Outputs: win rate, profit factor, max drawdown, expectancy.

## Tests

```bash
pytest -q
```

## Key runtime commands

- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest -q`
- Run bot (testnet): `python -m src.main`
