"""Historical candle fetch and backfill."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.storage import ParquetCache
from src.exchange.bybit_rest import BybitRestClient


@dataclass
class CandleFetcher:
    """Fetches and caches candles for configured intervals."""

    rest: BybitRestClient
    cache: ParquetCache

    @staticmethod
    def _normalize(raw: dict) -> pd.DataFrame:
        result = raw.get("result", {}).get("list", [])
        frame = pd.DataFrame(
            result,
            columns=["start", "open", "high", "low", "close", "volume", "turnover"],
        )
        if frame.empty:
            return frame
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        frame["start"] = pd.to_datetime(frame["start"].astype(int), unit="ms", utc=True)
        frame = frame.sort_values("start").reset_index(drop=True)
        return frame

    def get_candles(
        self,
        category: str,
        symbol: str,
        interval: str,
        limit: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return candles with parquet caching."""
        if not force_refresh:
            cached = self.cache.load(symbol, interval)
            if cached is not None and len(cached) >= min(50, limit):
                return cached.tail(limit).reset_index(drop=True)

        raw = self.rest.get_kline(category=category, symbol=symbol, interval=interval, limit=limit)
        frame = self._normalize(raw)
        if not frame.empty:
            self.cache.save(symbol, interval, frame)
        return frame

    def backfill_required(self, category: str, symbol: str) -> dict[str, pd.DataFrame]:
        """Backfill all required strategy horizons."""
        return {
            "60": self.get_candles(category, symbol, "60", limit=500),
            "15": self.get_candles(category, symbol, "15", limit=1000),
            "5": self.get_candles(category, symbol, "5", limit=1200),
        }
