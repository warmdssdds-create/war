# Troubleshooting Guide

Common issues and their resolutions.

---

## Bot refuses to start

### "mainnet mode requires --env mainnet --confirm-live"

**Cause**: `config/risk.yaml` sets `mode: mainnet` but the live-trading flags were not supplied on the command line.

**Fix**: Either:
- Add `--env mainnet --confirm-live` to your start command, **or**
- Set `mode: testnet` in `config/risk.yaml` to use testnet.

---

## API authentication errors (HTTP 401 / 10003)

**Symptoms**: Log line `BybitRestError: retCode=10003` or HTTP 401 responses.

**Checklist**:
1. Verify `BYBIT_API_KEY` and `BYBIT_API_SECRET` in `.env` match the Bybit dashboard.
2. Ensure the API key has **Derivatives** read + trade permissions enabled.
3. Check system time is synchronised (`timedatectl status`).  Bybit rejects requests with >5 s clock skew.
4. Confirm `BYBIT_TESTNET=true/false` matches the key's environment (testnet keys don't work on mainnet and vice-versa).

---

## Orders not placing

### "spread too wide, skipping"

**Cause**: The current bid-ask spread exceeds `max_spread_bps` in `config/strategy.yaml`.

**Fix**: Increase `max_spread_bps` or wait for tighter market conditions.  The default of 12 bps is conservative.

### "circuit breaker active"

**Cause**: A loss limit was hit.

**Fix**: Wait for the `pause_minutes` window to expire, or restart the bot to force-resume.

### qty rounds to zero

**Cause**: `capital_base_usdt` or `risk_per_trade` is too small relative to the current price and ATR.

**Fix**: Increase `capital_base_usdt`, or raise `risk_per_trade` (use caution — higher risk per trade increases drawdown risk).

---

## WebSocket disconnections

**Symptoms**: Log warnings `WebSocket disconnected, reconnecting…` every few seconds.

**Cause**: Network instability or Bybit server-side maintenance.

**Fix**: The WebSocket client has automatic exponential-backoff reconnection.  If disconnections persist:
- Check network connectivity and firewall rules.
- Verify the Bybit status page at [https://status.bybit.com](https://status.bybit.com).

---

## Parquet cache errors

**Symptoms**: `pyarrow.lib.ArrowInvalid: …` or `FileNotFoundError` for cache files.

**Fix**:
```bash
rm -rf data/cache/*.parquet
```
The bot will re-fetch candle history on the next start.

---

## Indicators return NaN

**Cause**: Insufficient history for the longest indicator period (EMA-200 needs ≥200 bars).

**Fix**: The default backfill fetches 500+ bars.  If you see NaN in live signals, check:
- The cache is not corrupted (delete and re-fetch as above).
- The exchange returned enough history for the requested interval.

---

## Health endpoint not responding

**Symptoms**: `curl http://127.0.0.1:8080/health` times out.

**Checklist**:
1. Confirm the bot is running (`ps aux | grep main.py`).
2. Check the port is not blocked by a local firewall.
3. Confirm the correct host/port in `config/strategy.yaml` (`health.host` / `health.port`).

---

## Memory / CPU growth over time

**Cause**: The in-memory candle DataFrames grow if the trim logic is not triggering.

**Fix**: The fetcher trims to the most recent `max_rows` (default 1000) on every cycle.  If memory still grows, reduce `max_rows` in `config/strategy.yaml`.

---

## Logs not rotating

**Cause**: The log directory does not exist or has wrong permissions.

**Fix**:
```bash
mkdir -p logs
chmod 755 logs
```

---

## Trade journal CSV has duplicate rows

**Cause**: The bot was hard-killed and restarted mid-write.

**Fix**: The CSV is append-only.  Deduplicate with:
```python
import pandas as pd
df = pd.read_csv("logs/trades.csv").drop_duplicates()
df.to_csv("logs/trades.csv", index=False)
```
