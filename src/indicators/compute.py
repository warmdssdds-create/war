"""Indicator computation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception:  # pragma: no cover - fallback for limited environments
    ta = None


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strategy indicators on OHLCV dataframe."""
    out = df.copy()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema55"] = out["close"].ewm(span=55, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()

    if ta is not None:
        out["rsi14"] = ta.rsi(out["close"], length=14)
    else:
        delta = out["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi14"] = 100 - (100 / (1 + rs))

    fast = out["close"].ewm(span=12, adjust=False).mean()
    slow = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    if ta is not None:
        stoch = ta.stochrsi(out["close"], length=14, rsi_length=14, k=3, d=3)
        out["stochrsi_k"] = stoch.iloc[:, 0]
        out["stochrsi_d"] = stoch.iloc[:, 1]
    else:
        rsi_min = out["rsi14"].rolling(14).min()
        rsi_max = out["rsi14"].rolling(14).max()
        stoch_rsi = (out["rsi14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        out["stochrsi_k"] = stoch_rsi.rolling(3).mean() * 100
        out["stochrsi_d"] = out["stochrsi_k"].rolling(3).mean()

    tr = _true_range(out)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_avg100"] = out["atr14"].rolling(100).mean()

    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + (2 * std)
    out["bb_lower"] = mid - (2 * std)

    typical = (out["high"] + out["low"] + out["close"]) / 3
    cum_vol = out["volume"].cumsum().replace(0, np.nan)
    out["vwap"] = (typical * out["volume"]).cumsum() / cum_vol

    close_diff = out["close"].diff().fillna(0)
    out["obv"] = (np.sign(close_diff) * out["volume"]).cumsum()
    vol_std = out["volume"].rolling(50).std().replace(0, np.nan)
    out["volume_zscore"] = (out["volume"] - out["volume"].rolling(50).mean()) / vol_std

    if ta is not None:
        adx = ta.adx(out["high"], out["low"], out["close"], length=14)
        out["adx14"] = adx.iloc[:, 0]
    else:
        plus_dm = (out["high"].diff()).clip(lower=0)
        minus_dm = (-out["low"].diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr14"])
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr14"])
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        out["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return out
