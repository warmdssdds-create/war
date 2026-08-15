from __future__ import annotations

from src.risk.circuit_breaker import DrawdownTracker, TradeStats, VolatilityStopTightener
from src.risk.position_sizing import (
    PositionSizingConfig,
    RiskLimits,
    apply_volatility_stop_tightening,
    compute_trade_levels_v2,
    compute_trailing_stop,
    kelly_fraction,
    validate_risk_limits,
)



def test_kelly_fraction() -> None:
    value = kelly_fraction(win_rate=0.55, avg_win=2.0, avg_loss=1.0)
    assert 0 < value < 1



def test_kelly_fraction_invalid_inputs() -> None:
    assert kelly_fraction(win_rate=0.5, avg_win=0.0, avg_loss=1.0) == 0.0
    assert kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=0.0) == 0.0



def test_trailing_stop_buy() -> None:
    stop = compute_trailing_stop("Buy", entry=100.0, high_water_mark=110.0, current_stop=95.0, atr=2.0, trail_mult=1.5)
    assert stop > 95.0
    assert stop < 110.0



def test_trailing_stop_sell() -> None:
    stop = compute_trailing_stop("Sell", entry=100.0, high_water_mark=90.0, current_stop=105.0, atr=2.0, trail_mult=1.5)
    assert stop < 105.0
    assert stop > 90.0



def test_volatility_stop_tightening() -> None:
    tightened = apply_volatility_stop_tightening(current_stop=95.0, atr=4.0, entry=100.0, side="Buy", vol_ratio=2.0, threshold=1.5)
    assert tightened >= 98.0



def test_drawdown_tracker() -> None:
    tracker = DrawdownTracker(starting_equity=1000.0)
    tracker.update(-100.0)
    tracker.update(20.0)
    assert tracker.current_drawdown_pct > 0
    assert tracker.max_drawdown_pct >= tracker.current_drawdown_pct
    tracker.reset()
    assert tracker.current_drawdown_pct == 0.0



def test_trade_stats_update() -> None:
    stats = TradeStats()
    stats.update(10.0)
    stats.update(-5.0)
    assert stats.total == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate == 0.5



def test_compute_trade_levels_v2_and_validate() -> None:
    config = PositionSizingConfig(
        capital=1000.0,
        risk_per_trade=0.01,
        atr_multiple=1.5,
        reward_risk=2.0,
        qty_step=0.1,
        min_qty=0.1,
        max_notional=500.0,
    )
    levels = compute_trade_levels_v2(entry=100.0, side="Buy", config=config, atr=2.0, structure_stop=96.0)
    validation = validate_risk_limits(levels, RiskLimits(max_position_notional=600.0, max_trade_risk=20.0))
    assert levels.qty > 0
    assert validation.valid is True



def test_volatility_stop_tightener_class() -> None:
    tightener = VolatilityStopTightener()
    tightener.set_spike(True)
    tightened = tightener.apply(current_stop=95.0, entry=100.0, atr=4.0, side="Buy")
    assert tightened >= 98.0

def test_trade_stats_profit_factor_nonzero() -> None:
    stats = TradeStats()
    stats.update(20.0)
    stats.update(-10.0)
    assert stats.profit_factor == 2.0
