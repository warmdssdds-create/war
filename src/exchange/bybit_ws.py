"""Bybit websocket helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from pybit.unified_trading import WebSocket


class BybitWsClient:
    """Wrapper for public/private Bybit websockets."""

    def __init__(self, testnet: bool = True) -> None:
        self.testnet = testnet
        self.public = WebSocket(testnet=testnet, channel_type="linear")
        self.private = WebSocket(
            testnet=testnet,
            channel_type="private",
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        self._last_message_ts = time.time()

    def _touch(self, _message: dict[str, Any]) -> None:
        self._last_message_ts = time.time()

    def subscribe_klines(self, symbol: str, interval: int, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to public klines."""

        def wrapped(message: dict[str, Any]) -> None:
            self._touch(message)
            callback(message)

        self.public.kline_stream(symbol=symbol, interval=interval, callback=wrapped)

    def subscribe_private(self, order_cb: Callable[[dict[str, Any]], None], position_cb: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to private order and position streams."""

        def wrap(cb: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
            def wrapped(message: dict[str, Any]) -> None:
                self._touch(message)
                cb(message)

            return wrapped

        self.private.order_stream(callback=wrap(order_cb))
        self.private.position_stream(callback=wrap(position_cb))

    def is_stale(self, stale_after_sec: int = 60) -> bool:
        """Detect stale stream."""
        return (time.time() - self._last_message_ts) > stale_after_sec

    def reconnect(self) -> None:
        """Rebuild websocket clients for reconnect flows."""
        self.public.exit()
        self.private.exit()
        self.public = WebSocket(testnet=self.testnet, channel_type="linear")
        self.private = WebSocket(
            testnet=self.testnet,
            channel_type="private",
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        self._last_message_ts = time.time()
