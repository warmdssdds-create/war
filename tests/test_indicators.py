import numpy as np
import pandas as pd

from src.indicators.compute import compute_indicators


def test_compute_indicators_adds_required_columns() -> None:
    rows = 260
    base = pd.DataFrame(
        {
            "start": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": np.linspace(20, 30, rows),
            "high": np.linspace(20.2, 30.2, rows),
            "low": np.linspace(19.8, 29.8, rows),
            "close": np.linspace(20, 30, rows),
            "volume": np.linspace(1000, 1200, rows),
            "turnover": np.linspace(20000, 36000, rows),
        }
    )

    out = compute_indicators(base)

    for col in ["ema21", "ema55", "ema200", "adx14", "rsi14", "macd", "stochrsi_k", "atr14", "vwap", "obv", "volume_zscore"]:
        assert col in out.columns

    assert out["ema21"].iloc[-1] > 0
