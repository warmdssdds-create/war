from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizing import compute_trade_levels


def test_position_sizing_respects_notional_cap() -> None:
    levels = compute_trade_levels(
        entry=30,
        side="Buy",
        capital=1000,
        risk_per_trade=0.01,
        atr=0.8,
        atr_multiple=1.5,
        structure_stop=28.8,
        reward_risk=2.0,
        qty_step=0.1,
        min_qty=0.1,
        max_notional=100,
    )
    assert levels.qty * 30 <= 100.5
    assert levels.stop_loss < 30
    assert levels.take_profit > 30


def test_circuit_breaker_pauses_after_consecutive_losses() -> None:
    breaker = CircuitBreaker(
        daily_loss_limit=0.03,
        weekly_loss_limit=0.06,
        max_consecutive_losses=2,
        pause_minutes=30,
        starting_equity=1000,
    )
    breaker.register_trade_result(-5)
    assert not breaker.is_paused()
    breaker.register_trade_result(-5)
    assert breaker.is_paused()
