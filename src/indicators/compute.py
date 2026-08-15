"""Indicator computation helpers for the ETCUSDT strategy.

The functions in this module are intentionally self-contained.  They avoid a
hard dependency on third-party technical-analysis libraries while still using
``pandas_ta`` when it is available in the runtime.  Each derived column is
named predictably so the live engine, optimizer, and backtester can share the
same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.time_utils import session_start_utc

try:
    import pandas_ta as ta
except Exception:  # pragma: no cover - optional dependency
    ta = None


_REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DivergenceWindow:
    """Container describing the price and OBV swings used for divergence."""

    price_now: float
    price_then: float
    obv_now: float
    obv_then: float


def _ensure_columns(df: pd.DataFrame, required: Iterable[str] = _REQUIRED_COLUMNS) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Compute classic true range."""
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


def _wilder_average(series: pd.Series, length: int) -> pd.Series:
    """Compute Wilder's moving average using EMA semantics."""
    return series.ewm(alpha=1 / max(length, 1), adjust=False).mean()


def _compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute RSI with a pandas-only fallback."""
    delta = close.diff()
    gain = _wilder_average(delta.clip(lower=0), length)
    loss = _wilder_average((-delta).clip(lower=0), length)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_adx(df: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, +DI, and -DI."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index,
    )

    atr = _wilder_average(_true_range(df), length)
    plus_di = 100 * (_wilder_average(plus_dm, length) / atr.replace(0, np.nan))
    minus_di = 100 * (_wilder_average(minus_dm, length) / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = _wilder_average(dx.fillna(0.0), length)
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def _compute_stoch_rsi(rsi: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Compute Stochastic RSI using only pandas operations."""
    rsi_min = rsi.rolling(length, min_periods=1).min()
    rsi_max = rsi.rolling(length, min_periods=1).max()
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)).clip(0, 1)
    k = (stoch_rsi.rolling(smooth_k, min_periods=1).mean() * 100).fillna(50.0)
    d = k.rolling(smooth_d, min_periods=1).mean().fillna(50.0)
    return k, d


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute a VWAP that resets at midnight UTC.

    The input frame may span multiple sessions.  A per-session cumulative value
    better reflects intraday execution anchors than a multi-day cumulative VWAP.
    """

    starts = pd.to_datetime(df["start"], utc=True)
    session_keys = starts.dt.floor("D")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    weighted = typical * df["volume"]
    cum_weighted = weighted.groupby(session_keys).cumsum()
    cum_volume = df["volume"].groupby(session_keys).cumsum().replace(0, np.nan)
    return (cum_weighted / cum_volume).ffill()


def detect_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Return a boolean series identifying fractal swing highs.

    A swing high is a candle whose high equals the local maximum within a
    symmetric window ``2 * lookback + 1`` and is strictly greater than the
    highs immediately before and after when possible.
    """

    _ensure_columns(df, ["high"])
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    highs = df["high"]
    window = 2 * lookback + 1
    rolling_max = highs.rolling(window, center=True, min_periods=window).max()
    left = highs > highs.shift(1)
    right = highs > highs.shift(-1)
    result = (highs == rolling_max) & left.fillna(False) & right.fillna(False)
    return result.fillna(False)


def detect_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Return a boolean series identifying fractal swing lows."""

    _ensure_columns(df, ["low"])
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    lows = df["low"]
    window = 2 * lookback + 1
    rolling_min = lows.rolling(window, center=True, min_periods=window).min()
    left = lows < lows.shift(1)
    right = lows < lows.shift(-1)
    result = (lows == rolling_min) & left.fillna(False) & right.fillna(False)
    return result.fillna(False)


def _latest_divergence_window(df: pd.DataFrame, lookback: int = 20) -> DivergenceWindow | None:
    """Collect the comparison points used to identify OBV divergence."""

    if len(df) < 4:
        return None
    swing_highs = detect_swing_highs(df, lookback=max(2, min(lookback // 2, 5)))
    swing_lows = detect_swing_lows(df, lookback=max(2, min(lookback // 2, 5)))
    recent = df.iloc[-lookback:].copy()
    local_highs = recent.loc[swing_highs.iloc[-lookback:]]
    local_lows = recent.loc[swing_lows.iloc[-lookback:]]
    if len(local_highs) >= 2:
        prev, curr = local_highs.iloc[-2], local_highs.iloc[-1]
        return DivergenceWindow(float(curr["high"]), float(prev["high"]), float(curr["obv"]), float(prev["obv"]))
    if len(local_lows) >= 2:
        prev, curr = local_lows.iloc[-2], local_lows.iloc[-1]
        return DivergenceWindow(float(curr["low"]), float(prev["low"]), float(curr["obv"]), float(prev["obv"]))
    return None


def detect_obv_divergence(df: pd.DataFrame, lookback: int = 20) -> int:
    """Detect bullish or bearish divergence between price and OBV.

    Returns
    -------
    int
        ``+1`` for bullish divergence, ``-1`` for bearish divergence, and
        ``0`` when no divergence is present.
    """

    _ensure_columns(df, ["close", "volume"])
    work = df.copy()
    if "obv" not in work.columns:
        close_diff = work["close"].diff().fillna(0.0)
        work["obv"] = (np.sign(close_diff) * work["volume"]).cumsum()
    if len(work) < max(lookback, 5):
        return 0

    recent = work.iloc[-lookback:]
    prior = recent.iloc[:-1]
    highs = recent["high"] if "high" in recent.columns else recent["close"]
    lows = recent["low"] if "low" in recent.columns else recent["close"]
    price_now = float(recent["close"].iloc[-1])
    obv_now = float(recent["obv"].iloc[-1])
    price_high_then = float(highs.iloc[:-1].max())
    price_low_then = float(lows.iloc[:-1].min())
    obv_high_then = float(recent.loc[highs.iloc[:-1].idxmax(), "obv"])
    obv_low_then = float(recent.loc[lows.iloc[:-1].idxmin(), "obv"])
    prior_obv_min = float(prior["obv"].min())
    prior_obv_max = float(prior["obv"].max())

    bearish = (price_now > price_high_then and obv_now < obv_high_then) or (price_now > float(prior["close"].max()) and obv_now < prior_obv_max)
    bullish = (price_now < price_low_then and obv_now > obv_low_then) or (price_now < float(prior["close"].min()) and obv_now > prior_obv_min)

    window = _latest_divergence_window(work, lookback=lookback)
    if window is not None:
        bearish = bearish or (window.price_now > window.price_then and window.obv_now < window.obv_then)
        bullish = bullish or (window.price_now < window.price_then and window.obv_now > window.obv_then)

    if bullish and not bearish:
        return 1
    if bearish and not bullish:
        return -1
    return 0


def detect_volatility_spike(df: pd.DataFrame, threshold: float = 1.8) -> bool:
    """Return ``True`` when ATR is above ``threshold * ATR average``."""

    _ensure_columns(df, ["high", "low", "close"])
    work = df if {"atr14", "atr_avg100"}.issubset(df.columns) else compute_indicators(df)
    row = work.iloc[-1]
    atr = float(row.get("atr14", 0.0))
    atr_avg = float(row.get("atr_avg100", 0.0))
    if atr_avg <= 0:
        return False
    return atr > threshold * atr_avg


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strategy indicators on an OHLCV dataframe.

    Existing column names are preserved and new columns are added in-place on a
    copied frame.  The function is tolerant of already-computed columns and will
    simply overwrite them with a fresh calculation.
    """

    _ensure_columns(df)
    out = df.copy().reset_index(drop=True)
    if "start" not in out.columns:
        out["start"] = pd.date_range(session_start_utc(), periods=len(out), freq="15min", tz="UTC")
    out["start"] = pd.to_datetime(out["start"], utc=True)

    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["turnover"] = out.get("turnover", out["close"] * out["volume"]).fillna(out["close"] * out["volume"])

    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema55"] = out["close"].ewm(span=55, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["sma20"] = out["close"].rolling(20, min_periods=1).mean()
    out["sma50"] = out["close"].rolling(50, min_periods=1).mean()

    if ta is not None:
        out["rsi14"] = ta.rsi(out["close"], length=14).fillna(50.0)
    else:
        out["rsi14"] = _compute_rsi(out["close"], 14)

    fast = out["close"].ewm(span=12, adjust=False).mean()
    slow = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    if ta is not None:
        stoch = ta.stochrsi(out["close"], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and not stoch.empty:
            out["stochrsi_k"] = stoch.iloc[:, 0].fillna(50.0)
            out["stochrsi_d"] = stoch.iloc[:, 1].fillna(50.0)
        else:
            out["stochrsi_k"], out["stochrsi_d"] = _compute_stoch_rsi(out["rsi14"])
    else:
        out["stochrsi_k"], out["stochrsi_d"] = _compute_stoch_rsi(out["rsi14"])

    tr = _true_range(out)
    out["tr"] = tr
    out["atr14"] = _wilder_average(tr, 14).fillna(0.0)
    out["atr_avg100"] = out["atr14"].rolling(100, min_periods=1).mean().bfill()
    out["atr_ratio"] = out["atr14"] / out["atr_avg100"].replace(0, np.nan)

    mid = out["close"].rolling(20, min_periods=1).mean()
    std = out["close"].rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + (2.0 * std)
    out["bb_lower"] = mid - (2.0 * std)
    out["bb_width"] = ((out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan)).fillna(0.0)
    denom = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_pct"] = ((out["close"] - out["bb_lower"]) / denom).clip(0.0, 1.0).fillna(0.5)

    out["vwap"] = _session_vwap(out)
    out["vwap_distance_pct"] = ((out["close"] - out["vwap"]) / out["vwap"].replace(0, np.nan)).fillna(0.0)

    close_diff = out["close"].diff().fillna(0.0)
    out["obv"] = (np.sign(close_diff) * out["volume"]).cumsum()
    out["obv_ema21"] = out["obv"].ewm(span=21, adjust=False).mean()
    vol_mean = out["volume"].rolling(50, min_periods=1).mean()
    vol_std = out["volume"].rolling(50, min_periods=1).std(ddof=0).replace(0, np.nan)
    out["volume_ma20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["volume_zscore"] = ((out["volume"] - vol_mean) / vol_std).fillna(0.0)

    if ta is not None:
        adx = ta.adx(out["high"], out["low"], out["close"], length=14)
        if adx is not None and not adx.empty:
            out["adx14"] = adx.iloc[:, 0].fillna(0.0)
            out["plus_di"] = adx.iloc[:, 1].fillna(0.0)
            out["minus_di"] = adx.iloc[:, 2].fillna(0.0)
        else:
            out["adx14"], out["plus_di"], out["minus_di"] = _compute_adx(out, 14)
    else:
        out["adx14"], out["plus_di"], out["minus_di"] = _compute_adx(out, 14)

    out["returns"] = out["close"].pct_change().fillna(0.0)
    out["rolling_volatility"] = out["returns"].rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    out["swing_high"] = detect_swing_highs(out, lookback=3)
    out["swing_low"] = detect_swing_lows(out, lookback=3)
    out["obv_divergence"] = 0
    if len(out) >= 25:
        divergences: list[int] = [0] * len(out)
        for idx in range(20, len(out)):
            divergences[idx] = detect_obv_divergence(out.iloc[: idx + 1], lookback=min(20, idx + 1))
        out["obv_divergence"] = divergences
    out["volatility_spike"] = out["atr_ratio"].fillna(0.0) > 1.8
    out["trend_bias"] = np.where(out["ema21"] >= out["ema55"], 1, -1)

    fill_columns = [
        "ema21",
        "ema55",
        "ema200",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "stochrsi_k",
        "stochrsi_d",
        "atr14",
        "atr_avg100",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "vwap",
        "obv",
        "adx14",
        "plus_di",
        "minus_di",
    ]
    out[fill_columns] = out[fill_columns].ffill().bfill()
    return out
