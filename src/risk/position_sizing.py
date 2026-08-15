"""Position sizing, stop logic, and portfolio-level risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass
class TradeLevels:
    """Derived execution levels."""

    qty: float
    stop_loss: float
    take_profit: float


@dataclass
class PositionSizingConfig:
    """Configuration bundle for position sizing and trade-level generation."""

    capital: float
    risk_per_trade: float
    atr_multiple: float
    reward_risk: float
    qty_step: float
    min_qty: float
    max_notional: float
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    max_risk_pct: float = 0.02
    min_stop_distance_pct: float = 0.001


@dataclass
class RiskLimits:
    """Hard risk limits checked before order placement."""

    max_position_notional: float
    max_trade_risk: float
    max_leverage: float = 5.0
    allow_zero_sl: bool = False


@dataclass
class PositionValidation:
    """Validation result for a prospective position."""

    valid: bool
    reason: str


@dataclass
class RichTradeLevels(TradeLevels):
    """Extended trade levels used by later validation stages."""

    entry: float = 0.0
    side: str = "Buy"
    stop_distance: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0
    reward_distance: float = 0.0
    expected_fees: float = 0.0



def _round_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest exchange lot step."""
    if step <= 0:
        return value
    return round(value / step) * step



def _floor_step(value: float, step: float) -> float:
    """Floor ``value`` to the exchange lot step."""
    if step <= 0:
        return value
    return floor(value / step) * step



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



def compute_trade_levels_v2(
    entry: float,
    side: str,
    config: PositionSizingConfig,
    atr: float,
    structure_stop: float,
) -> RichTradeLevels:
    """Compute trade levels using a dataclass configuration object."""
    side = side.capitalize()
    atr_stop_distance = max(atr * config.atr_multiple, entry * config.min_stop_distance_pct)
    structure_distance = abs(entry - structure_stop) if structure_stop > 0 else 0.0
    if structure_distance > 0:
        stop_distance = min(atr_stop_distance, structure_distance)
    else:
        stop_distance = atr_stop_distance
    stop_distance = max(stop_distance, entry * config.min_stop_distance_pct)

    if side == "Buy":
        stop_loss = entry - stop_distance
        take_profit = entry + (stop_distance * config.reward_risk)
    else:
        stop_loss = entry + stop_distance
        take_profit = entry - (stop_distance * config.reward_risk)

    risk_budget = config.capital * min(config.risk_per_trade, config.max_risk_pct)
    friction_per_unit = entry * ((config.fee_bps + config.slippage_bps) / 10_000.0)
    effective_distance = stop_distance + friction_per_unit
    raw_qty = risk_budget / max(effective_distance, 1e-9)
    max_qty_by_notional = config.max_notional / max(entry, 1e-9)
    qty = max(_floor_step(min(raw_qty, max_qty_by_notional), config.qty_step), 0.0)
    if 0 < qty < config.min_qty:
        qty = config.min_qty
    notional = qty * entry
    reward_distance = abs(take_profit - entry)
    expected_fees = notional * ((config.fee_bps + config.slippage_bps) / 10_000.0)
    return RichTradeLevels(
        qty=qty,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry=entry,
        side=side,
        stop_distance=stop_distance,
        risk_amount=qty * stop_distance,
        notional=notional,
        reward_distance=reward_distance,
        expected_fees=expected_fees,
    )



def check_position_valid(levels: TradeLevels, min_qty: float, max_qty: float) -> tuple[bool, str]:
    """Validate basic quantity and stop/target geometry."""
    if levels.qty < min_qty:
        return False, "qty_below_min"
    if levels.qty > max_qty:
        return False, "qty_above_max"
    if levels.stop_loss <= 0 or levels.take_profit <= 0:
        return False, "non_positive_level"
    return True, "ok"



def validate_risk_limits(levels: RichTradeLevels, limits: RiskLimits) -> PositionValidation:
    """Validate a rich trade level set against hard risk limits."""
    if levels.qty <= 0:
        return PositionValidation(False, "zero_qty")
    if levels.notional > limits.max_position_notional:
        return PositionValidation(False, "notional_limit")
    if levels.risk_amount > limits.max_trade_risk:
        return PositionValidation(False, "risk_limit")
    if not limits.allow_zero_sl and levels.stop_distance <= 0:
        return PositionValidation(False, "missing_stop")
    leverage = levels.notional / max(limits.max_trade_risk, 1e-9)
    if leverage > limits.max_leverage * 100:
        return PositionValidation(False, "implied_leverage_limit")
    return PositionValidation(True, "ok")



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



def compute_trailing_stop(
    side: str,
    entry: float,
    high_water_mark: float,
    current_stop: float,
    atr: float,
    trail_mult: float,
) -> float:
    """Return an updated trailing stop for long or short positions."""
    distance = max(atr * trail_mult, entry * 0.001)
    side = side.capitalize()
    if side == "Buy":
        proposed = high_water_mark - distance
        return max(current_stop, min(proposed, high_water_mark))
    proposed = high_water_mark + distance
    return min(current_stop, max(proposed, high_water_mark))



def apply_volatility_stop_tightening(
    current_stop: float,
    atr: float,
    entry: float,
    side: str,
    vol_ratio: float,
    threshold: float = 1.5,
) -> float:
    """Tighten a stop when volatility expands sharply."""
    if vol_ratio < threshold:
        return current_stop
    side = side.capitalize()
    half_atr = max(atr * 0.5, entry * 0.0005)
    if side == "Buy":
        tightened = entry - half_atr
        return max(current_stop, tightened)
    tightened = entry + half_atr
    return min(current_stop, tightened)



def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute the Kelly criterion fraction, clipped to ``[0, 1]``."""
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    fraction = win_rate - ((1.0 - win_rate) / b)
    return max(min(fraction, 1.0), 0.0)
