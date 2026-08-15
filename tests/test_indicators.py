from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.compute import (
    compute_indicators,
    detect_obv_divergence,
    detect_swing_highs,
    detect_swing_lows,
    detect_volatility_spike,
)



def _base_frame(rows: int = 260) -> pd.DataFrame:
    start = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    close = np.linspace(20, 30, rows)
    return pd.DataFrame(
        {
            "start": start,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000, 1200, rows),
            "turnover": close * np.linspace(1000, 1200, rows),
        }
    )



def test_compute_indicators_adds_required_columns() -> None:
    out = compute_indicators(_base_frame())
    for col in [
        "ema21",
        "ema55",
        "ema200",
        "adx14",
        "rsi14",
        "macd",
        "stochrsi_k",
        "atr14",
        "vwap",
        "obv",
        "volume_zscore",
    ]:
        assert col in out.columns
    assert out["ema21"].iloc[-1] > 0



def test_swing_high_detection() -> None:
    frame = _base_frame(15)
    frame["high"] = [1, 2, 3, 5, 3, 2, 1, 1, 2, 4, 2, 1, 1, 2, 1]
    swings = detect_swing_highs(frame, lookback=1)
    assert bool(swings.iloc[3]) is True
    assert bool(swings.iloc[9]) is True
    assert swings.sum() >= 2



def test_swing_low_detection() -> None:
    frame = _base_frame(15)
    frame["low"] = [4, 3, 2, 1, 2, 3, 4, 4, 3, 2, 0.5, 2, 3, 4, 5]
    swings = detect_swing_lows(frame, lookback=1)
    assert bool(swings.iloc[3]) is True
    assert bool(swings.iloc[10]) is True
    assert swings.sum() >= 2



def test_obv_divergence_bullish() -> None:
    frame = _base_frame(25)
    close = np.array([20.0, 19.8, 19.6, 19.4, 19.2, 19.0, 18.8, 18.6, 18.4, 18.2, 18.0, 18.3, 18.6, 18.9, 19.1, 18.9, 18.7, 18.5, 18.3, 18.1, 17.9, 17.8, 17.7, 17.6, 17.5])
    frame["close"] = close
    frame["high"] = close + 0.2
    frame["low"] = close - 0.2
    frame["volume"] = np.linspace(100, 180, len(frame))
    frame["obv"] = np.array(
        [-100, -180, -240, -290, -330, -360, -380, -390, -395, -398, -400, -320, -240, -160, -80, -40, 0, 40, 80, 120, 160, 200, 240, 280, 320],
        dtype=float,
    )
    assert detect_obv_divergence(frame, lookback=20) == 1



def test_obv_divergence_bearish() -> None:
    frame = _base_frame(25)
    close = np.array([20.0, 20.2, 20.4, 20.6, 20.8, 21.0, 21.2, 21.4, 21.6, 21.8, 22.0, 21.9, 21.8, 21.7, 21.6, 21.8, 22.0, 22.2, 22.4, 22.6, 22.8, 23.0, 23.1, 23.2, 23.3])
    frame["close"] = close
    frame["high"] = close + 0.2
    frame["low"] = close - 0.2
    frame["volume"] = np.linspace(100, 180, len(frame))
    frame["obv"] = np.array([100, 180, 240, 290, 330, 360, 380, 390, 395, 398, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120], dtype=float)
    assert detect_obv_divergence(frame, lookback=20) == -1



def test_volatility_spike_detection() -> None:
    frame = _base_frame(220)
    frame.loc[219, "high"] = frame.loc[219, "close"] + 7.0
    frame.loc[219, "low"] = frame.loc[219, "close"] - 7.0
    enriched = compute_indicators(frame)
    assert detect_volatility_spike(enriched, threshold=1.2) is True



def test_vwap_session_reset() -> None:
    frame = _base_frame(200)
    frame["start"] = pd.date_range("2024-01-01 23:00:00", periods=200, freq="15min", tz="UTC")
    out = compute_indicators(frame)
    jan1_last = out.loc[out["start"].dt.date == pd.Timestamp("2024-01-01").date(), "vwap"].iloc[-1]
    jan2_first = out.loc[out["start"].dt.date == pd.Timestamp("2024-01-02").date(), "vwap"].iloc[0]
    typical = ((out.loc[out["start"].dt.date == pd.Timestamp("2024-01-02").date()].iloc[0][["high", "low", "close"]].sum()) / 3)
    assert jan2_first != jan1_last
    assert abs(jan2_first - typical) < 1e-9



def test_bb_width_computed() -> None:
    out = compute_indicators(_base_frame())
    assert "bb_width" in out.columns
    assert "bb_pct" in out.columns
    assert out["bb_width"].iloc[-1] >= 0



def test_all_columns_present() -> None:
    out = compute_indicators(_base_frame())
    required = {
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
        "bb_width",
        "bb_pct",
        "vwap",
        "obv",
        "adx14",
        "plus_di",
        "minus_di",
        "obv_divergence",
        "volatility_spike",
    }
    assert required.issubset(set(out.columns))



def test_indicator_output_preserves_row_count() -> None:
    frame = _base_frame(128)
    out = compute_indicators(frame)
    assert len(out) == len(frame)
    assert out["start"].iloc[0] == frame["start"].iloc[0]



def test_indicator_output_is_numeric_for_key_columns() -> None:
    out = compute_indicators(_base_frame())
    for col in ["ema21", "rsi14", "macd_hist", "atr14", "bb_pct", "plus_di", "minus_di"]:
        assert pd.api.types.is_numeric_dtype(out[col])
