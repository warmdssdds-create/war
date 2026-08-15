"""Execution engine for live trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.exchange.bybit_rest import BybitRestClient
from src.exchange.order_manager import OrderIntent, generate_order_link_id
from src.risk.position_sizing import TradeLevels, compute_trade_levels


@dataclass
class ExecutionConfig:
    """Runtime execution controls."""

    category: str
    symbol: str
    max_spread_bps: float


class ExecutionEngine:
    """Performs guarded order placement and reconciliation."""

    def __init__(self, rest: BybitRestClient, cfg: ExecutionConfig, logger: logging.Logger) -> None:
        self.rest = rest
        self.cfg = cfg
        self.logger = logger

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
        return True

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

        levels: TradeLevels = compute_trade_levels(
            entry=entry,
            side=side,
            capital=capital,
            risk_per_trade=float(risk_cfg["position_sizing"]["risk_per_trade"]),
            atr=atr,
            atr_multiple=float(risk_cfg["stops"]["atr_multiple"]),
            structure_stop=structure_stop,
            reward_risk=float(risk_cfg["stops"]["reward_risk"]),
            qty_step=float(risk_cfg["position_sizing"]["qty_step"]),
            min_qty=float(risk_cfg["position_sizing"]["min_qty"]),
            max_notional=float(risk_cfg["position_sizing"]["max_notional_usdt"]),
        )

        link_id = generate_order_link_id(
            OrderIntent(symbol=self.cfg.symbol, side=side, qty=levels.qty, signal_ts=signal_ts)
        )
        self.logger.info("Placing %s qty=%.4f link=%s", side, levels.qty, link_id)
        return self.rest.place_market_order(
            category=self.cfg.category,
            symbol=self.cfg.symbol,
            side=side,
            qty=levels.qty,
            order_link_id=link_id,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
        )

    def reconcile(self) -> dict[str, Any]:
        """Reconcile open orders and positions from REST."""
        positions = self.rest.get_positions(self.cfg.category, self.cfg.symbol)
        orders = self.rest.get_open_orders(self.cfg.category, self.cfg.symbol)
        snapshot = {"positions": positions, "orders": orders}
        self.logger.info("Reconciled exchange state")
        return snapshot
