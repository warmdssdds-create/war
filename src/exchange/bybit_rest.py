"""Bybit REST client wrapper with retries."""

from __future__ import annotations

import os
from typing import Any

from pybit.unified_trading import HTTP
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class BybitRestClient:
    """Thin resilient wrapper around pybit HTTP client."""

    def __init__(self, testnet: bool = True) -> None:
        self.client = HTTP(
            testnet=testnet,
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            recv_window=int(os.getenv("BYBIT_RECV_WINDOW", "5000")),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def place_market_order(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: str,
        stop_loss: float,
        take_profit: float,
    ) -> dict[str, Any]:
        """Place market order with attached SL/TP."""
        return self.client.place_order(
            category=category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            orderLinkId=order_link_id,
            takeProfit=str(take_profit),
            stopLoss=str(stop_loss),
            tpslMode="Full",
        )

    def get_orderbook(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch top-of-book data."""
        return self.client.get_orderbook(category=category, symbol=symbol)

    def get_positions(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch open positions."""
        return self.client.get_positions(category=category, symbol=symbol)

    def get_open_orders(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch open orders."""
        return self.client.get_open_orders(category=category, symbol=symbol)

    def get_kline(self, category: str, symbol: str, interval: str, limit: int = 200) -> dict[str, Any]:
        """Fetch candles."""
        return self.client.get_kline(category=category, symbol=symbol, interval=interval, limit=limit)
