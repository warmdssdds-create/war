# Operations Guide

This guide covers day-to-day operation of the bot: startup, monitoring, graceful shutdown, and alert configuration.

---

## Prerequisites

- Python 3.11+
- A Bybit account with API keys (testnet recommended first)
- Dependencies installed: `pip install -r requirements.txt`
- `.env` file populated from `.env.example`

---

## Environment Setup

```bash
cp .env.example .env
# Edit .env and fill in your API credentials
nano .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `BYBIT_API_KEY` | REST / WS API key |
| `BYBIT_API_SECRET` | Corresponding secret |
| `BYBIT_TESTNET` | `true` (default) or `false` |
| `TELEGRAM_BOT_TOKEN` | Optional: Telegram alert bot token |
| `TELEGRAM_CHAT_ID` | Optional: Telegram chat/channel ID |

---

## Starting the Bot

### Testnet (safe default)

```bash
python -m src.main
```

No extra flags required.  The bot will connect to `api-testnet.bybit.com` and refuse to place real orders.

### Mainnet (live trading)

```bash
python -m src.main --env mainnet --confirm-live
```

Both flags are **required**.  Without them the runner raises `SystemExit` before connecting to mainnet.

> ⚠️  **Risk disclaimer**: Live trading involves real financial risk.  Past performance of any strategy does not guarantee future results.  Size positions conservatively and never risk capital you cannot afford to lose.

---

## Health Check

The bot exposes a JSON health endpoint at `http://127.0.0.1:8080/health` (port configurable in `config/strategy.yaml`).

```bash
curl -s http://127.0.0.1:8080/health | python -m json.tool
```

Example response:

```json
{
  "status": "running",
  "last_candle_ts": "2024-06-01T12:00:00+00:00",
  "last_signal": "Buy",
  "last_order_id": "ETCUSDT_Buy_1717243200",
  "circuit_breaker_active": false,
  "daily_pnl": -2.50,
  "weekly_pnl": 12.30,
  "consecutive_losses": 0,
  "uptime_start": "2024-06-01T08:00:00+00:00",
  "checked_at": "2024-06-01T12:05:00+00:00"
}
```

---

## Logs

Log files are written to `logs/` with daily rotation (14-day retention by default).

```
logs/
├── bot.log          # Main structured log
└── trades.csv       # Trade journal (append-only CSV)
```

Tail live:

```bash
tail -f logs/bot.log
```

---

## Monitoring via Telegram

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, the bot sends alerts on:

- Bot start / stop
- Order placement
- Circuit breaker trigger or release
- Any unhandled exception (with traceback summary)

---

## Circuit Breakers

See `config/risk.yaml` for thresholds.  When a breaker trips:

1. The bot logs a warning and updates health state to `"paused"`.
2. All new signal evaluations are skipped.
3. After `pause_minutes` the breaker resets automatically.

To force-resume early, restart the process.

---

## Graceful Shutdown

Send `SIGINT` (Ctrl-C) or `SIGTERM`:

```bash
kill -SIGTERM <pid>
```

The bot catches the signal, cancels any open working orders, flushes the trade journal, and exits with code 0.

---

## Running as a systemd Service

```ini
[Unit]
Description=Bybit ETCUSDT Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/etcusdt-bot
EnvironmentFile=/opt/etcusdt-bot/.env
ExecStart=/opt/etcusdt-bot/.venv/bin/python -m src.main
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable etcusdt-bot
sudo systemctl start etcusdt-bot
sudo journalctl -u etcusdt-bot -f
```

---

## Updating the Bot

```bash
git pull origin main
pip install -r requirements.txt
sudo systemctl restart etcusdt-bot
```
