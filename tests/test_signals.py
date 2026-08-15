import numpy as np
import pandas as pd

from src.indicators.compute import compute_indicators
from src.strategy.signals import evaluate_signal


WEIGHTS = {
    "trending": {
        "ema_alignment": 0.25,
        "adx_strength": 0.20,
        "macd_momentum": 0.15,
        "rsi_bias": 0.10,
        "vwap_position": 0.10,
        "obv_trend": 0.10,
        "volume_confirmation": 0.10,
    },
    "ranging": {
        "bb_reversion": 0.25,
        "stochrsi_extreme": 0.20,
        "rsi_reversion": 0.15,
        "vwap_revert": 0.15,
        "atr_compression": 0.15,
        "volume_confirmation": 0.10,
    },
}


def _frame() -> pd.DataFrame:
    rows = 260
    close = np.linspace(20, 35, rows)
    df = pd.DataFrame(
        {
            "start": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000, 1800, rows),
            "turnover": close * np.linspace(1000, 1800, rows),
        }
    )
    out = compute_indicators(df)
    out.loc[out.index[-1], "volume"] *= 3
    out.loc[out.index[-1], "volume_zscore"] = 2.0
    out.loc[out.index[-1], "adx14"] = 30
    out.loc[out.index[-1], "macd_hist"] = 1
    out.loc[out.index[-1], "rsi14"] = 60
    return out


def test_signal_requires_confirmed_candle() -> None:
    signal = evaluate_signal(_frame(), weights=WEIGHTS, threshold=0.6, confirmed=False)
    assert signal.side is None
    assert signal.reason == "unconfirmed"


def test_signal_generates_side_when_confluence_passes() -> None:
    signal = evaluate_signal(_frame(), weights=WEIGHTS, threshold=0.6, confirmed=True)
    assert signal.side in {"Buy", "Sell"}
    assert signal.reason == "ok"
