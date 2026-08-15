"""Bybit websocket helpers with in-memory state reconciliation."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - environment dependent
    from pybit.unified_trading import WebSocket
except Exception:  # pragma: no cover - optional dependency
    WebSocket = None  # type: ignore[assignment]

from src.utils.time_utils import ensure_utc, utc_now


class _NullWebSocket:
    """Offline stand-in used when pybit is not available."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.callbacks: list[Callable[[dict[str, Any]], None]] = []

    def kline_stream(self, symbol: str, interval: int, callback: Callable[[dict[str, Any]], None]) -> None:
        _ = (symbol, interval)
        self.callbacks.append(callback)

    def order_stream(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)

    def position_stream(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callbacks.append(callback)

    def exit(self) -> None:
        return None


@dataclass
class PositionState:
    """Latest known position snapshot from websocket updates."""

    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    updated_at: float


@dataclass
class OrderState:
    """Latest known order snapshot from websocket updates."""

    order_id: str
    link_id: str
    symbol: str
    side: str
    status: str
    filled_qty: float
    remaining_qty: float
    price: float


class KlineBuffer:
    """Store confirmed and in-progress candles by symbol and interval."""

    def __init__(self) -> None:
        self._confirmed: dict[tuple[str, str], dict[str, Any]] = {}
        self._partial: dict[tuple[str, str], dict[str, Any]] = {}

    def update(self, message: dict[str, Any]) -> None:
        """Update buffers from a Bybit kline message."""
        topic = str(message.get("topic", ""))
        data = message.get("data", [])
        parts = topic.split(".")
        interval = str(parts[1]) if len(parts) >= 3 else str(message.get("interval", ""))
        symbol = str(parts[2]) if len(parts) >= 3 else str(message.get("symbol", ""))
        if isinstance(data, dict):
            data = [data]
        for candle in data:
            payload = {
                "start": int(candle.get("start", candle.get("startTime", 0))),
                "end": int(candle.get("end", candle.get("endTime", 0))),
                "open": float(candle.get("open", 0.0)),
                "high": float(candle.get("high", 0.0)),
                "low": float(candle.get("low", 0.0)),
                "close": float(candle.get("close", 0.0)),
                "volume": float(candle.get("volume", 0.0)),
                "turnover": float(candle.get("turnover", 0.0)),
                "confirm": bool(candle.get("confirm", False)),
            }
            key = (symbol, interval)
            if payload["confirm"]:
                self._confirmed[key] = payload
                self._partial.pop(key, None)
            else:
                self._partial[key] = payload

    def get_confirmed(self, symbol: str, interval: str | int) -> dict[str, Any] | None:
        """Return the latest confirmed candle."""
        return self._confirmed.get((symbol, str(interval)))

    def get_partial(self, symbol: str, interval: str | int) -> dict[str, Any] | None:
        """Return the latest in-progress candle."""
        return self._partial.get((symbol, str(interval)))


class StateReconciler:
    """Track the latest position and order state from websocket messages."""

    def __init__(self) -> None:
        self.positions: dict[str, PositionState] = {}
        self.orders: dict[str, OrderState] = {}

    def update_from_position_message(self, msg: dict[str, Any]) -> None:
        """Update positions from a Bybit private-stream payload."""
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        for item in data:
            symbol = str(item.get("symbol", ""))
            self.positions[symbol] = PositionState(
                symbol=symbol,
                side=str(item.get("side", "")),
                size=float(item.get("size", 0.0)),
                entry_price=float(item.get("avgPrice", item.get("entryPrice", 0.0))),
                unrealized_pnl=float(item.get("unrealisedPnl", item.get("unrealizedPnl", 0.0))),
                updated_at=utc_now().timestamp(),
            )

    def update_from_order_message(self, msg: dict[str, Any]) -> None:
        """Update orders from a Bybit private-stream payload."""
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        for item in data:
            order_id = str(item.get("orderId", ""))
            self.orders[order_id] = OrderState(
                order_id=order_id,
                link_id=str(item.get("orderLinkId", "")),
                symbol=str(item.get("symbol", "")),
                side=str(item.get("side", "")),
                status=str(item.get("orderStatus", item.get("status", ""))),
                filled_qty=float(item.get("cumExecQty", item.get("filledQty", 0.0))),
                remaining_qty=max(float(item.get("qty", 0.0)) - float(item.get("cumExecQty", item.get("filledQty", 0.0))), 0.0),
                price=float(item.get("price", 0.0)),
            )

    def get_position(self, symbol: str) -> PositionState | None:
        """Return the latest position for a symbol."""
        return self.positions.get(symbol)

    def get_open_orders(self, symbol: str) -> list[OrderState]:
        """Return open orders for a symbol."""
        open_statuses = {"New", "PartiallyFilled", "Created", "Untriggered"}
        return [order for order in self.orders.values() if order.symbol == symbol and order.status in open_statuses]


class ReconnectManager:
    """Encapsulate websocket staleness checks and reconnect actions."""

    def should_reconnect(self, last_ts: float, stale_thresh: int = 60) -> bool:
        """Return ``True`` if the websocket is considered stale."""
        return (time.time() - last_ts) > stale_thresh

    def reconnect(self, ws_client: "BybitWsClient") -> None:
        """Reconnect a websocket client in-place."""
        ws_client.reconnect()


class BybitWsClient:
    """Wrapper for public/private Bybit websockets."""

    def __init__(self, testnet: bool = True) -> None:
        self.testnet = testnet
        socket_cls = WebSocket or _NullWebSocket
        self.public = socket_cls(testnet=testnet, channel_type="linear")
        self.private = socket_cls(
            testnet=testnet,
            channel_type="private",
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        self._last_message_ts = time.time()
        self.kline_buffer = KlineBuffer()
        self.state = StateReconciler()

    @property
    def last_message_ts(self) -> float:
        return self._last_message_ts

    def _touch(self, _message: dict[str, Any]) -> None:
        self._last_message_ts = time.time()

    def subscribe_klines(self, symbol: str, interval: int, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to public klines."""

        def wrapped(message: dict[str, Any]) -> None:
            self._touch(message)
            self.kline_buffer.update(message)
            callback(message)

        self.public.kline_stream(symbol=symbol, interval=interval, callback=wrapped)

    def subscribe_private(self, order_cb: Callable[[dict[str, Any]], None], position_cb: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to private order and position streams."""

        def wrap(cb: Callable[[dict[str, Any]], None], updater: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
            def wrapped(message: dict[str, Any]) -> None:
                self._touch(message)
                updater(message)
                cb(message)

            return wrapped

        self.private.order_stream(callback=wrap(order_cb, self.state.update_from_order_message))
        self.private.position_stream(callback=wrap(position_cb, self.state.update_from_position_message))

    def is_stale(self, stale_after_sec: int = 60) -> bool:
        """Detect stale stream."""
        return (time.time() - self._last_message_ts) > stale_after_sec

    def reconnect(self) -> None:
        """Rebuild websocket clients for reconnect flows."""
        self.public.exit()
        self.private.exit()
        socket_cls = WebSocket or _NullWebSocket
        self.public = socket_cls(testnet=self.testnet, channel_type="linear")
        self.private = socket_cls(
            testnet=self.testnet,
            channel_type="private",
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        self._last_message_ts = time.time()
