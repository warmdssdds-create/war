"""Position sizing and stop/target calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeLevels:
    """Derived execution levels."""

    qty: float
    stop_loss: float
    take_profit: float


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def compute_trade_levels(
    entry: float,
    side: str,
    capital: float,
    risk_per_trade: float,
    atr: float,
    atr_multiple: float,
    structure_stop: float,
    reward_risk: float,
    qty_step: float,
    min_qty: float,
    max_notional: float,
) -> TradeLevels:
    """Compute conservative qty and SL/TP with fixed-fractional risk."""
    atr_stop_distance = atr * atr_multiple
    structure_distance = abs(entry - structure_stop)
    stop_distance = min(atr_stop_distance, structure_distance) if structure_distance > 0 else atr_stop_distance
    stop_distance = max(stop_distance, entry * 0.001)

    if side == "Buy":
        stop_loss = entry - stop_distance
        take_profit = entry + (stop_distance * reward_risk)
    else:
        stop_loss = entry + stop_distance
        take_profit = entry - (stop_distance * reward_risk)

    risk_budget = capital * risk_per_trade
    raw_qty = risk_budget / max(stop_distance, 1e-9)
    max_qty_by_notional = max_notional / max(entry, 1e-9)
    qty = min(raw_qty, max_qty_by_notional)
    qty = max(_round_step(qty, qty_step), min_qty)

    return TradeLevels(qty=qty, stop_loss=stop_loss, take_profit=take_profit)


def update_protective_stop(
    side: str,
    entry: float,
    current_price: float,
    current_stop: float,
    atr: float,
    trigger_rr: float,
    risk_distance: float,
    trail_atr_mult: float,
) -> float:
    """Apply breakeven then trailing stop logic."""
    if risk_distance <= 0:
        return current_stop

    if side == "Buy":
        rr_now = (current_price - entry) / risk_distance
        breakeven_stop = entry if rr_now >= trigger_rr else current_stop
        trail = current_price - (atr * trail_atr_mult)
        return max(current_stop, breakeven_stop, trail)

    rr_now = (entry - current_price) / risk_distance
    breakeven_stop = entry if rr_now >= trigger_rr else current_stop
    trail = current_price + (atr * trail_atr_mult)
    return min(current_stop, breakeven_stop, trail)
