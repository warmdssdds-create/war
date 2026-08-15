"""Order helpers, idempotent link-id generation, and active-order tracking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.utils.time_utils import utc_now


@dataclass(frozen=True)
class OrderIntent:
    """Serializable shape for stable order id generation."""

    symbol: str
    side: str
    qty: float
    signal_ts: int


@dataclass
class OrderRecord:
    """Tracked order state used by the live execution engine."""

    link_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    status: str
    sl: float
    tp: float
    created_at: datetime
    updated_at: datetime
    exchange_order_id: str | None = None
    signal_ts: int | None = None

    @property
    def remaining_qty(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    @property
    def is_open(self) -> bool:
        return self.status in {"Created", "New", "Open", "PartiallyFilled"}



def generate_order_link_id(intent: OrderIntent, namespace: str = "bot") -> str:
    """Create deterministic 32-char idempotent orderLinkId."""
    raw = f"{namespace}|{intent.symbol}|{intent.side}|{intent.qty:.8f}|{intent.signal_ts}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:32]


class OrderManager:
    """Manage active entry orders and synchronize them with websocket state."""

    def __init__(self) -> None:
        self.active_orders: dict[str, OrderRecord] = {}

    def place_entry(
        self,
        rest: Any,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        sl: float,
        tp: float,
        signal_ts: int,
    ) -> dict[str, Any]:
        """Place an entry order, record it locally, and return the exchange response."""
        intent = OrderIntent(symbol=symbol, side=side, qty=qty, signal_ts=signal_ts)
        link_id = generate_order_link_id(intent)
        response = rest.place_market_order(
            category=category,
            symbol=symbol,
            side=side,
            qty=qty,
            order_link_id=link_id,
            stop_loss=sl,
            take_profit=tp,
        )
        now = utc_now()
        result = response.get("result", {}) if isinstance(response, dict) else {}
        self.active_orders[link_id] = OrderRecord(
            link_id=link_id,
            symbol=symbol,
            side=side,
            qty=float(qty),
            filled_qty=float(result.get("cumExecQty", 0.0)),
            status=str(result.get("orderStatus", "Created")),
            sl=float(sl),
            tp=float(tp),
            created_at=now,
            updated_at=now,
            exchange_order_id=result.get("orderId"),
            signal_ts=signal_ts,
        )
        return response

    def get_order(self, link_id: str) -> OrderRecord | None:
        """Return a tracked order by link ID."""
        return self.active_orders.get(link_id)

    def update_from_ws_message(self, message: dict[str, Any]) -> None:
        """Parse a websocket order update and merge it into local state."""
        data = message.get("data", [])
        if isinstance(data, dict):
            data = [data]
        for item in data:
            link_id = str(item.get("orderLinkId", ""))
            if not link_id:
                continue
            record = self.active_orders.get(link_id)
            if record is None:
                qty = float(item.get("qty", item.get("cumExecQty", 0.0)))
                record = OrderRecord(
                    link_id=link_id,
                    symbol=str(item.get("symbol", "")),
                    side=str(item.get("side", "")),
                    qty=qty,
                    filled_qty=0.0,
                    status=str(item.get("orderStatus", item.get("status", "Created"))),
                    sl=0.0,
                    tp=0.0,
                    created_at=utc_now(),
                    updated_at=utc_now(),
                    exchange_order_id=str(item.get("orderId", "")) or None,
                )
                self.active_orders[link_id] = record
            record.status = str(item.get("orderStatus", item.get("status", record.status)))
            record.filled_qty = float(item.get("cumExecQty", item.get("filledQty", record.filled_qty)))
            if "stopLoss" in item and item.get("stopLoss") not in (None, ""):
                record.sl = float(item["stopLoss"])
            if "takeProfit" in item and item.get("takeProfit") not in (None, ""):
                record.tp = float(item["takeProfit"])
            record.updated_at = utc_now()
            record.exchange_order_id = str(item.get("orderId", record.exchange_order_id or "")) or record.exchange_order_id
            if record.status in {"Filled", "Cancelled", "Rejected", "Deactivated"} and record.remaining_qty <= 0:
                self.active_orders.pop(link_id, None)

    def handle_partial_fill(self, link_id: str, filled_qty: float) -> bool:
        """Update partial-fill quantity and return whether the order is complete."""
        record = self.active_orders[link_id]
        record.filled_qty = min(max(float(filled_qty), 0.0), record.qty)
        record.updated_at = utc_now()
        if record.filled_qty >= record.qty:
            record.status = "Filled"
            return True
        record.status = "PartiallyFilled"
        return False

    def cancel_all(self, rest: Any, category: str, symbol: str) -> dict[str, Any]:
        """Cancel all open orders for a symbol and update local records."""
        response = rest.cancel_all_orders(category=category, symbol=symbol)
        for record in self.active_orders.values():
            if record.symbol == symbol and record.is_open:
                record.status = "Cancelled"
                record.updated_at = utc_now()
        return response

    def get_active_entries(self) -> list[OrderRecord]:
        """Return tracked entry orders that are still open."""
        return [record for record in self.active_orders.values() if record.is_open]

    def update_exit_order(
        self,
        rest: Any,
        category: str,
        symbol: str,
        position_idx: int,
        new_sl: float | None,
        new_tp: float | None,
    ) -> dict[str, Any]:
        """Update exchange stops for an open position."""
        return rest.set_trading_stop(
            category=category,
            symbol=symbol,
            position_idx=position_idx,
            stop_loss=new_sl,
            take_profit=new_tp,
            trailing_stop=None,
        )
