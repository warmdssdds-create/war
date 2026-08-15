from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.compute import compute_indicators
from src.strategy.regime_filter import Regime
from src.strategy.signals import SignalHistory, apply_5m_refinement, evaluate_exit_signal, evaluate_signal


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



def _trend_frame(rows: int = 260) -> pd.DataFrame:
    close = np.linspace(20, 28, rows)
    frame = pd.DataFrame(
        {
            "start": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000, 1600, rows),
            "turnover": close * np.linspace(1000, 1600, rows),
        }
    )
    return compute_indicators(frame)



def test_evaluate_signal_trending_buy() -> None:
    frame = _trend_frame()
    result = evaluate_signal(frame, weights=WEIGHTS, threshold=0.4, confirmed=True)
    assert result.side in {"Buy", "Sell", None}
    assert result.regime in {Regime.TRENDING, Regime.RANGING, Regime.TRANSITIONAL, Regime.VOLATILITY_SPIKE}



def test_evaluate_signal_unconfirmed() -> None:
    frame = _trend_frame()
    result = evaluate_signal(frame, weights=WEIGHTS, threshold=0.4, confirmed=False)
    assert result.side is None
    assert result.reason == "unconfirmed"



def test_exit_signal_when_stop_hit() -> None:
    frame = _trend_frame().copy()
    frame.loc[frame.index[-1], "low"] = frame.iloc[-1]["close"] - 5.0
    stop = float(frame.iloc[-1]["close"] - 1.0)
    result = evaluate_exit_signal(frame, "Buy", entry_price=float(frame.iloc[-2]["close"]), current_stop=stop, regime=Regime.TRENDING)
    assert result.should_exit is True
    assert result.reason == "stop_hit"



def test_exit_signal_momentum_reversal_for_short() -> None:
    frame = _trend_frame().copy()
    frame.loc[frame.index[-1], "macd_hist"] = 1.5
    frame.loc[frame.index[-1], "rsi14"] = 70.0
    result = evaluate_exit_signal(frame, "Sell", entry_price=27.0, current_stop=40.0, regime=Regime.TRENDING)
    assert result.should_exit is True
    assert result.reason == "momentum_reversal"



def test_5m_refinement_filters_entry() -> None:
    frame = _trend_frame(60)
    last = frame.iloc[-1].copy()
    last["macd_hist"] = -1.0
    last["close"] = last["ema21"] - 0.5
    frame.iloc[-1] = last
    assert apply_5m_refinement(frame, "Buy") is False



def test_5m_refinement_passes_short() -> None:
    frame = _trend_frame(60)
    last = frame.iloc[-1].copy()
    last["macd_hist"] = -1.0
    last["close"] = last["ema21"] - 0.5
    last["rsi14"] = 40.0
    frame.iloc[-1] = last
    assert apply_5m_refinement(frame, "Sell") is True



def test_signal_history_win_rate() -> None:
    history = SignalHistory(maxlen=20)
    history.add("Buy", 0.8, Regime.TRENDING, "ok", won=True)
    history.add("Sell", 0.7, Regime.TRENDING, "ok", won=False)
    history.add(None, 0.2, Regime.RANGING, "skip", won=None)
    assert abs(history.win_rate() - 0.5) < 1e-9
    assert len(history.recent_signals(2)) == 2



def test_signal_history_keeps_maxlen() -> None:
    history = SignalHistory(maxlen=3)
    for idx in range(6):
        history.add("Buy", 0.5 + idx, Regime.TRENDING, f"reason-{idx}", won=bool(idx % 2))
    assert len(history) == 3
    assert history.recent_signals(1)[0].reason == "reason-5"



def test_regime_volatility_pause() -> None:
    frame = _trend_frame(220)
    frame.loc[frame.index[-1], "atr14"] = frame.iloc[-1]["atr_avg100"] * 2.5
    result = evaluate_signal(frame, weights=WEIGHTS, threshold=0.5, confirmed=True)
    assert result.side is None
    assert result.reason == "volatility_pause"


def test_exit_signal_returns_new_stop_when_holding() -> None:
    frame = _trend_frame().copy()
    result = evaluate_exit_signal(frame, "Buy", entry_price=float(frame.iloc[-10]["close"]), current_stop=float(frame.iloc[-10]["close"] - 1.0), regime=Regime.TRENDING)
    assert result.should_exit is False
    assert result.new_stop is not None
