# Architecture Overview

This document describes the high-level architecture of the Bybit ETCUSDT perpetuals trading bot.

## Design Principles

- **Testnet-first**: Live trading is disabled unless both `--env mainnet` and `--confirm-live` flags are explicitly supplied.
- **Fail-safe defaults**: Missing config keys fall back to conservative defaults; the bot halts cleanly on unexpected errors.
- **Modular layout**: Each concern (exchange, data, indicators, strategy, risk, execution, monitoring) lives in its own package.
- **Typed code**: All public functions are fully type-annotated; `dataclasses` are used for structured data transfer.

---

## Package Map

```
src/
├── main.py               # Live runner CLI entry point
├── backtest.py           # Event-driven backtester CLI
├── data/
│   ├── fetcher.py        # REST candle backfill + live subscription shim
│   └── storage.py        # Parquet-backed local cache
├── exchange/
│   ├── bybit_rest.py     # Signed REST client with retry/back-off
│   ├── bybit_ws.py       # WebSocket ticker + kline subscription
│   └── order_manager.py  # Idempotent order lifecycle management
├── indicators/
│   └── compute.py        # Vectorised indicator computation (pandas)
├── strategy/
│   ├── signals.py        # Confluence-based signal generation
│   ├── regime_filter.py  # Trending vs. ranging regime detection
│   └── confluence.py     # Weighted indicator scoring
├── risk/
│   ├── position_sizing.py # ATR-based sizing + notional cap
│   └── circuit_breaker.py # Daily/weekly loss limits + consecutive-loss pause
├── execution/
│   └── engine.py         # Order placement, spread check, trade journal
├── monitoring/
│   ├── logger.py         # Structured rotating-file logger setup
│   ├── alerts.py         # Telegram alert dispatcher
│   └── health.py         # Background HTTP /health endpoint
└── utils/
    ├── time_utils.py     # UTC helpers, millisecond conversion
    └── validators.py     # Input validation primitives
```

---

## Data Flow

```
CandleFetcher (REST backfill / WS live)
        │
        ▼
ParquetCache (optional persistence)
        │
        ▼
compute_indicators()          ← ema21/55/200, adx14, rsi14, macd,
        │                          stochrsi_k, atr14, vwap, obv, vol_zscore
        ▼
evaluate_signal()
  ├── RegimeFilter.classify()   → trending | ranging
  ├── ConfluenceScorer.score()  → weighted float in [0, 1]
  └── SignalResult(side, score, regime)
        │
        ▼ (signal.side is not None)
CircuitBreaker.is_paused()?
        │  no
        ▼
ExecutionEngine.place_entry()
  ├── Spread check (max_spread_bps)
  ├── compute_trade_levels()    → qty, stop_loss, take_profit
  ├── OrderManager.submit()     → idempotent orderLinkId
  └── TradeJournal CSV write
        │
        ▼
HealthState.update()  ← last_order_id, last_signal, daily_pnl …
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/strategy.yaml` | Symbol, timeframes, confluence thresholds, indicator weights |
| `config/risk.yaml` | Risk per trade, ATR multiples, circuit-breaker limits |
| `config/logging.yaml` | Log levels, file rotation, structured JSON option |

All values in YAML files can be overridden at runtime via environment variables where noted in `.env.example`.

---

## Safety Controls

1. **Testnet guard** – `BybitRestClient(testnet=True)` targets `api-testnet.bybit.com` by default.  Setting `mode: mainnet` in `config/risk.yaml` is necessary but **not sufficient**; the runner also requires `--env mainnet --confirm-live` on the command line.

2. **Circuit breakers** – Three independent triggers, any one of which pauses trading for a configurable window:
   - Daily PnL loss exceeds `daily_loss_limit` fraction of starting equity.
   - Weekly PnL loss exceeds `weekly_loss_limit` fraction of starting equity.
   - `max_consecutive_losses` losing trades in a row.

3. **Max notional cap** – Position size is always capped at `max_notional_usdt` regardless of ATR-computed size.

4. **Spread gate** – Orders are skipped when the current bid-ask spread exceeds `max_spread_bps`.

5. **Idempotent order IDs** – `orderLinkId` is derived from `{symbol}_{side}_{signal_ts}`, preventing duplicate fills on retried requests.

---

## Threading Model

- **Main thread**: bot loop (~15-second sleep cadence).
- **Health server thread** (daemon): `HealthServer` runs `HTTPServer.serve_forever()`.
- **WebSocket thread** (daemon): `BybitWebSocket` manages its own reconnect loop.

All shared state updates go through `threading.Lock` inside `health.py`.
