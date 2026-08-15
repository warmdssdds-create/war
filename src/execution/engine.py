"""Execution engine for live trading and ongoing position management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.exchange.bybit_rest import BybitRestClient
from src.exchange.order_manager import OrderIntent, OrderManager, generate_order_link_id
from src.risk.position_sizing import (
    PositionSizingConfig,
    RichTradeLevels,
    compute_trade_levels,
    compute_trade_levels_v2,
    compute_trailing_stop,
)


@dataclass
class ExecutionConfig:
    """Runtime execution controls."""

    category: str
    symbol: str
    max_spread_bps: float
    fee_bps: float = 6.0
    trailing_atr_mult: float = 1.5
    default_reward_risk: float = 1.8


@dataclass
class ExecutionState:
    """Current execution state for a symbol."""

    has_position: bool
    position_side: str | None
    position_qty: float
    entry_price: float
    current_sl: float | None
    current_tp: float | None



def fee_adjusted_tp(entry: float, side: str, stop_loss: float, reward_risk: float, fee_bps: float) -> float:
    """Calculate a TP preserving net reward:risk after fees."""
    risk_distance = abs(entry - stop_loss)
    gross_target = risk_distance * reward_risk
    fee_distance = entry * (fee_bps / 10_000.0) * 2
    if side.capitalize() == "Buy":
        return entry + gross_target + fee_distance
    return entry - gross_target - fee_distance



def check_instrument_limits(rest: BybitRestClient, category: str, symbol: str) -> tuple[float, float, float]:
    """Fetch min qty, qty step, and tick size from the exchange."""
    return rest.extract_instrument_limits(category, symbol)



def get_execution_state(rest: BybitRestClient, category: str, symbol: str) -> ExecutionState:
    """Build execution state from live REST positions."""
    positions = rest.get_positions(category=category, symbol=symbol)
    items = positions.get("result", {}).get("list", [])
    for item in items:
        size = float(item.get("size", 0.0))
        if size <= 0:
            continue
        side = str(item.get("side", ""))
        return ExecutionState(
            has_position=True,
            position_side=side,
            position_qty=size,
            entry_price=float(item.get("avgPrice", item.get("entryPrice", 0.0))),
            current_sl=float(item.get("stopLoss", 0.0)) or None,
            current_tp=float(item.get("takeProfit", 0.0)) or None,
        )
    return ExecutionState(False, None, 0.0, 0.0, None, None)



def manage_exit(rest: BybitRestClient, state: ExecutionState, atr: float, regime: str) -> tuple[float | None, float | None]:
    """Compute updated protective exits for an open position."""
    if not state.has_position or state.position_side is None or state.current_sl is None:
        return state.current_sl, state.current_tp
    side = state.position_side.capitalize()
    trail_mult = 1.0 if regime == "VOLATILITY_SPIKE" else 1.5
    water_mark = state.entry_price + atr * 2 if side == "Buy" else state.entry_price - atr * 2
    new_sl = compute_trailing_stop(side, state.entry_price, water_mark, state.current_sl, atr, trail_mult)
    return new_sl, state.current_tp



def emergency_close(rest: BybitRestClient, category: str, symbol: str) -> dict[str, Any]:
    """Cancel all orders and close any open position with a market order."""
    cancel_resp = rest.cancel_all_orders(category=category, symbol=symbol)
    state = get_execution_state(rest, category, symbol)
    close_resp: dict[str, Any] | None = None
    if state.has_position and state.position_side is not None and state.position_qty > 0:
        close_resp = rest.close_position(category=category, symbol=symbol, side=state.position_side, qty=state.position_qty)
    return {"cancel": cancel_resp, "close": close_resp}


class ExecutionEngine:
    """Perform guarded order placement, tracking, and exit adjustments."""

    def __init__(
        self,
        rest: BybitRestClient,
        cfg: ExecutionConfig,
        logger: logging.Logger,
        order_manager: OrderManager | None = None,
    ) -> None:
        self.rest = rest
        self.cfg = cfg
        self.logger = logger
        self.order_manager = order_manager or OrderManager()
        self.state = ExecutionState(False, None, 0.0, 0.0, None, None)

    def _spread_bps(self, orderbook: dict[str, Any]) -> float:
        bids = orderbook.get("result", {}).get("b", [])
        asks = orderbook.get("result", {}).get("a", [])
        if not bids or not asks:
            return 10_000
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        return ((best_ask - best_bid) / max(mid, 1e-9)) * 10_000

    def can_enter(self) -> bool:
        """Check spread guard before any entry."""
        orderbook = self.rest.get_orderbook(self.cfg.category, self.cfg.symbol)
        spread_bps = self._spread_bps(orderbook)
        if spread_bps > self.cfg.max_spread_bps:
            self.logger.warning("Spread too wide: %.2f bps", spread_bps)
            return False
        if self.state.has_position:
            self.logger.info("Skipping entry because position already exists")
            return False
        return True

    def _build_sizing_config(self, risk_cfg: dict[str, Any], capital: float) -> PositionSizingConfig:
        ps = risk_cfg["position_sizing"]
        stops = risk_cfg["stops"]
        return PositionSizingConfig(
            capital=capital,
            risk_per_trade=float(ps["risk_per_trade"]),
            atr_multiple=float(stops["atr_multiple"]),
            reward_risk=float(stops["reward_risk"]),
            qty_step=float(ps["qty_step"]),
            min_qty=float(ps["min_qty"]),
            max_notional=float(ps["max_notional_usdt"]),
            fee_bps=self.cfg.fee_bps,
        )

    def place_entry(
        self,
        side: str,
        signal_ts: int,
        entry: float,
        capital: float,
        atr: float,
        structure_stop: float,
        risk_cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Place market order with attached SL/TP using idempotent order id."""
        if not self.can_enter():
            return None

        min_qty, qty_step, _ = check_instrument_limits(self.rest, self.cfg.category, self.cfg.symbol)
        config = self._build_sizing_config(risk_cfg, capital)
        if min_qty > 0:
            config.min_qty = min_qty
        if qty_step > 0:
            config.qty_step = qty_step

        levels_v2: RichTradeLevels = compute_trade_levels_v2(
            entry=entry,
            side=side,
            config=config,
            atr=atr,
            structure_stop=structure_stop,
        )
        take_profit = fee_adjusted_tp(entry, side, levels_v2.stop_loss, config.reward_risk, self.cfg.fee_bps)
        levels = compute_trade_levels(
            entry=entry,
            side=side,
            capital=capital,
            risk_per_trade=config.risk_per_trade,
            atr=atr,
            atr_multiple=config.atr_multiple,
            structure_stop=structure_stop,
            reward_risk=config.reward_risk,
            qty_step=config.qty_step,
            min_qty=config.min_qty,
            max_notional=config.max_notional,
        )
        levels.take_profit = take_profit

        link_id = generate_order_link_id(OrderIntent(symbol=self.cfg.symbol, side=side, qty=levels.qty, signal_ts=signal_ts))
        self.logger.info("Placing %s qty=%.4f link=%s", side, levels.qty, link_id)
        response = self.order_manager.place_entry(
            rest=self.rest,
            category=self.cfg.category,
            symbol=self.cfg.symbol,
            side=side,
            qty=levels.qty,
            sl=levels.stop_loss,
            tp=levels.take_profit,
            signal_ts=signal_ts,
        )
        self.state = ExecutionState(True, side, levels.qty, entry, levels.stop_loss, levels.take_profit)
        return response

    def place_exit_adjustment(self, position_idx: int, new_sl: float | None, new_tp: float | None) -> dict[str, Any]:
        """Update stops or targets for the active position."""
        response = self.order_manager.update_exit_order(
            rest=self.rest,
            category=self.cfg.category,
            symbol=self.cfg.symbol,
            position_idx=position_idx,
            new_sl=new_sl,
            new_tp=new_tp,
        )
        self.state.current_sl = new_sl
        self.state.current_tp = new_tp
        return response

    def reconcile(self) -> dict[str, Any]:
        """Reconcile open orders and positions from REST."""
        self.state = get_execution_state(self.rest, self.cfg.category, self.cfg.symbol)
        positions = self.rest.get_positions(self.cfg.category, self.cfg.symbol)
        orders = self.rest.get_open_orders(self.cfg.category, self.cfg.symbol)
        snapshot = {"positions": positions, "orders": orders, "state": self.state}
        self.logger.info("Reconciled exchange state")
        return snapshot

    def manage_open_position(self, atr: float, regime: str, position_idx: int = 0) -> dict[str, Any] | None:
        """Update trailing stop for an open position when warranted."""
        self.state = get_execution_state(self.rest, self.cfg.category, self.cfg.symbol)
        if not self.state.has_position:
            return None
        new_sl, new_tp = manage_exit(self.rest, self.state, atr=atr, regime=regime)
        if new_sl == self.state.current_sl and new_tp == self.state.current_tp:
            return None
        return self.place_exit_adjustment(position_idx=position_idx, new_sl=new_sl, new_tp=new_tp)
