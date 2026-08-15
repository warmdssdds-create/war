"""Resilient Bybit REST client wrappers.

The live bot prefers ``pybit`` for convenience, but the repository should also
work in offline and test environments where pybit is unavailable.  For that
reason this module provides a small HMAC fallback using ``requests`` directly.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:  # pragma: no cover - depends on environment
    from pybit.unified_trading import HTTP
except Exception:  # pragma: no cover - optional dependency
    HTTP = None  # type: ignore[assignment]


class _FallbackHTTP:
    """A small pybit-like wrapper backed by the HMAC helper."""

    def __init__(self, testnet: bool, api_key: str, api_secret: str, recv_window: int) -> None:
        self._fallback = HmacFallback(testnet=testnet, api_key=api_key, api_secret=api_secret, recv_window=recv_window)

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_post("/v5/order/create", kwargs)

    def get_orderbook(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/market/orderbook", kwargs)

    def get_positions(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/position/list", kwargs)

    def get_open_orders(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/order/realtime", kwargs)

    def get_kline(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/market/kline", kwargs)

    def get_wallet_balance(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/account/wallet-balance", kwargs)

    def cancel_all_orders(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_post("/v5/order/cancel-all", kwargs)

    def get_instruments_info(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/market/instruments-info", kwargs)

    def get_funding_rate_history(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/market/funding/history", kwargs)

    def set_trading_stop(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_post("/v5/position/trading-stop", kwargs)

    def get_order_history(self, **kwargs: Any) -> dict[str, Any]:
        return self._fallback.signed_get("/v5/order/history", kwargs)


@dataclass
class HmacFallback:
    """Fallback client using direct HMAC-SHA256 signing via ``requests``."""

    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    recv_window: int = 5000
    timeout: int = 15

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = self.api_secret or os.getenv("BYBIT_API_SECRET", "")
        host = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        self.base_url = host.rstrip("/")
        self.session = requests.Session()

    def _timestamp(self) -> str:
        return str(int(__import__("time").time() * 1000))

    def _normalise_params(self, params: dict[str, Any] | None) -> dict[str, str]:
        return {key: str(value) for key, value in (params or {}).items() if value is not None}

    def _headers(self, query_string: str) -> dict[str, str]:
        ts = self._timestamp()
        payload = f"{ts}{self.api_key}{self.recv_window}{query_string}"
        signature = hmac.new(self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json",
        }

    def signed_get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a signed GET request."""
        normalised = self._normalise_params(params)
        query_string = urlencode(sorted(normalised.items()))
        resp = self.session.get(
            f"{self.base_url}{endpoint}",
            params=normalised,
            headers=self._headers(query_string),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def signed_post(self, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a signed POST request."""
        normalised = self._normalise_params(body)
        query_string = __import__("json").dumps(normalised, separators=(",", ":"), sort_keys=True)
        resp = self.session.post(
            f"{self.base_url}{endpoint}",
            json=normalised,
            headers=self._headers(query_string),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


class BybitRestClient:
    """Thin resilient wrapper around pybit HTTP client."""

    def __init__(self, testnet: bool = True) -> None:
        self.testnet = testnet
        self.api_key = os.getenv("BYBIT_API_KEY", "")
        self.api_secret = os.getenv("BYBIT_API_SECRET", "")
        self.recv_window = int(os.getenv("BYBIT_RECV_WINDOW", "5000"))
        self.hmac = HmacFallback(testnet=testnet, api_key=self.api_key, api_secret=self.api_secret, recv_window=self.recv_window)
        if HTTP is not None:
            self.client = HTTP(
                testnet=testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=self.recv_window,
            )
        else:
            self.client = _FallbackHTTP(testnet, self.api_key, self.api_secret, self.recv_window)

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call a pybit method and fall back to direct requests on failure."""
        method = getattr(self.client, method_name)
        try:
            return method(*args, **kwargs)
        except Exception:
            fallback_method = getattr(self.hmac, "signed_post" if method_name in {"place_order", "cancel_all_orders", "set_trading_stop"} else "signed_get")
            endpoint_map = {
                "place_order": "/v5/order/create",
                "get_orderbook": "/v5/market/orderbook",
                "get_positions": "/v5/position/list",
                "get_open_orders": "/v5/order/realtime",
                "get_kline": "/v5/market/kline",
                "get_wallet_balance": "/v5/account/wallet-balance",
                "cancel_all_orders": "/v5/order/cancel-all",
                "get_instruments_info": "/v5/market/instruments-info",
                "get_funding_rate_history": "/v5/market/funding/history",
                "set_trading_stop": "/v5/position/trading-stop",
                "get_order_history": "/v5/order/history",
            }
            payload = kwargs or (args[0] if args else {})
            return fallback_method(endpoint_map[method_name], payload)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
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
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place market order with attached SL/TP."""
        return self._call(
            "place_order",
            category=category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            orderLinkId=order_link_id,
            takeProfit=str(take_profit) if take_profit else None,
            stopLoss=str(stop_loss) if stop_loss else None,
            reduceOnly=reduce_only,
            tpslMode="Full",
        )

    def place_limit_order(
        self,
        category: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_link_id: str,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a limit order."""
        return self._call(
            "place_order",
            category=category,
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty),
            price=str(price),
            orderLinkId=order_link_id,
            reduceOnly=reduce_only,
            timeInForce="GTC",
        )

    def get_orderbook(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch top-of-book data."""
        return self._call("get_orderbook", category=category, symbol=symbol)

    def get_positions(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch open positions."""
        return self._call("get_positions", category=category, symbol=symbol)

    def get_open_orders(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch open orders."""
        return self._call("get_open_orders", category=category, symbol=symbol)

    def get_kline(self, category: str, symbol: str, interval: str, limit: int = 200, start: int | None = None) -> dict[str, Any]:
        """Fetch candles."""
        payload: dict[str, Any] = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            payload["start"] = start
        return self._call("get_kline", **payload)

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict[str, Any]:
        """Fetch account wallet balance and equity."""
        return self._call("get_wallet_balance", accountType=account_type)

    def cancel_all_orders(self, category: str, symbol: str) -> dict[str, Any]:
        """Cancel all open orders for a symbol."""
        return self._call("cancel_all_orders", category=category, symbol=symbol)

    def close_position(self, category: str, symbol: str, side: str, qty: float) -> dict[str, Any]:
        """Close a position with a reduce-only market order."""
        close_side = "Sell" if side.capitalize() == "Buy" else "Buy"
        link_id = f"close-{symbol.lower()}-{int(float(qty) * 1000)}"
        return self.place_market_order(
            category=category,
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_link_id=link_id,
            stop_loss=0.0,
            take_profit=0.0,
            reduce_only=True,
        )

    def get_instruments_info(self, category: str, symbol: str) -> dict[str, Any]:
        """Fetch instrument specifications such as tick size and lot step."""
        return self._call("get_instruments_info", category=category, symbol=symbol)

    def get_funding_rate_history(self, category: str, symbol: str, limit: int = 200) -> dict[str, Any]:
        """Fetch historical funding rates."""
        return self._call("get_funding_rate_history", category=category, symbol=symbol, limit=limit)

    def set_trading_stop(
        self,
        category: str,
        symbol: str,
        position_idx: int,
        stop_loss: float | None,
        take_profit: float | None,
        trailing_stop: float | None,
    ) -> dict[str, Any]:
        """Update an existing position's stop-loss/take-profit values."""
        return self._call(
            "set_trading_stop",
            category=category,
            symbol=symbol,
            positionIdx=position_idx,
            stopLoss=str(stop_loss) if stop_loss is not None else None,
            takeProfit=str(take_profit) if take_profit is not None else None,
            trailingStop=str(trailing_stop) if trailing_stop is not None else None,
            tpslMode="Full",
        )

    def get_order_history(self, category: str, symbol: str, limit: int = 50) -> dict[str, Any]:
        """Fetch historical orders."""
        return self._call("get_order_history", category=category, symbol=symbol, limit=limit)

    def get_equity(self, account_type: str = "UNIFIED", coin: str = "USDT") -> float:
        """Extract wallet equity from the balance response."""
        balance = self.get_wallet_balance(account_type=account_type)
        accounts = balance.get("result", {}).get("list", [])
        for account in accounts:
            for coin_info in account.get("coin", []):
                if str(coin_info.get("coin", "")).upper() == coin.upper():
                    return float(coin_info.get("equity", coin_info.get("walletBalance", 0.0)))
        return 0.0

    def extract_instrument_limits(self, category: str, symbol: str) -> tuple[float, float, float]:
        """Return ``(min_qty, qty_step, tick_size)`` for an instrument."""
        info = self.get_instruments_info(category=category, symbol=symbol)
        instruments = info.get("result", {}).get("list", [])
        if not instruments:
            return 0.0, 0.0, 0.0
        instrument = instruments[0]
        lot_filter = instrument.get("lotSizeFilter", {})
        price_filter = instrument.get("priceFilter", {})
        return (
            float(lot_filter.get("minOrderQty", 0.0)),
            float(lot_filter.get("qtyStep", 0.0)),
            float(price_filter.get("tickSize", 0.0)),
        )
