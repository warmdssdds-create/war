"""Historical candle fetching, pagination, live-candle assembly, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.storage import ParquetCache
from src.exchange.bybit_rest import BybitRestClient
from src.utils.time_utils import candle_open_time, ensure_utc, interval_to_minutes, is_candle_confirmed, to_millis


CANDLE_COLUMNS = ["start", "open", "high", "low", "close", "volume", "turnover"]


@dataclass
class LiveCandleAssembler:
    """Track partial and confirmed websocket candles by interval."""

    symbol: str = ""
    confirmed: dict[str, dict[str, Any]] = field(default_factory=dict)
    partial: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_seen_start: dict[str, int] = field(default_factory=dict)

    def update(self, ws_message: dict[str, Any]) -> None:
        """Parse a Bybit kline message and update internal candle state."""
        topic = str(ws_message.get("topic", ""))
        data = ws_message.get("data", [])
        if isinstance(data, dict):
            data = [data]
        interval = topic.split(".")[1] if "." in topic else str(ws_message.get("interval", ""))
        for item in data:
            start = int(item.get("start", item.get("startTime", 0)))
            candle = {
                "start": start,
                "open": float(item.get("open", 0.0)),
                "high": float(item.get("high", 0.0)),
                "low": float(item.get("low", 0.0)),
                "close": float(item.get("close", 0.0)),
                "volume": float(item.get("volume", 0.0)),
                "turnover": float(item.get("turnover", 0.0)),
                "confirm": bool(item.get("confirm", False)),
            }
            self.symbol = self.symbol or topic.split(".")[-1]
            self.last_seen_start[interval] = start
            if candle["confirm"]:
                self.confirmed[interval] = candle
                self.partial.pop(interval, None)
            else:
                self.partial[interval] = candle

    def get_confirmed(self, interval: str | int) -> dict[str, Any] | None:
        """Return the latest confirmed candle for an interval."""
        return self.confirmed.get(str(interval))

    def get_partial(self, interval: str | int) -> dict[str, Any] | None:
        """Return the latest partial candle for an interval."""
        return self.partial.get(str(interval))

    def is_new_candle(self, interval: str | int, timestamp: int) -> bool:
        """Detect the start of a new candle period."""
        key = str(interval)
        previous = self.last_seen_start.get(key)
        if previous is None:
            return True
        return int(timestamp) > int(previous)


@dataclass
class CandleFetcher:
    """Fetches and caches candles for configured intervals."""

    rest: BybitRestClient
    cache: ParquetCache

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> pd.DataFrame:
        """Normalize Bybit candle payloads to a DataFrame."""
        result = raw.get("result", {}).get("list", [])
        frame = pd.DataFrame(result, columns=CANDLE_COLUMNS)
        if frame.empty:
            return frame
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        frame["start"] = pd.to_datetime(frame["start"].astype(int), unit="ms", utc=True)
        frame = frame.sort_values("start").drop_duplicates(subset=["start"], keep="last").reset_index(drop=True)
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


def backfill_with_pagination(
    rest: BybitRestClient,
    category: str,
    symbol: str,
    interval: str,
    total_candles: int = 1000,
) -> pd.DataFrame:
    """Fetch candles in pages from most recent backwards."""
    page_size = min(1000, max(50, total_candles))
    frames: list[pd.DataFrame] = []
    start: int | None = None
    remaining = total_candles
    while remaining > 0:
        raw = rest.get_kline(category=category, symbol=symbol, interval=interval, limit=min(page_size, remaining), start=start)
        frame = CandleFetcher._normalize(raw)
        if frame.empty:
            break
        frames.append(frame)
        remaining -= len(frame)
        earliest = frame["start"].min()
        step_ms = interval_to_minutes(interval) * 60 * 1000
        start = max(to_millis(earliest) - (len(frame) * step_ms), 0)
        if len(frame) < min(page_size, remaining + len(frame)):
            break
    if not frames:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("start").drop_duplicates(subset=["start"], keep="last").tail(total_candles)
    return combined.reset_index(drop=True)



def validate_candles(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a candle dataframe for schema and continuity issues."""
    issues: list[str] = []
    if df.empty:
        return False, ["empty_frame"]
    for column in CANDLE_COLUMNS:
        if column not in df.columns:
            issues.append(f"missing_{column}")
    if issues:
        return False, issues
    if df["start"].duplicated().any():
        issues.append("duplicate_timestamps")
    if df[["open", "high", "low", "close"]].isna().any().any():
        issues.append("nan_ohlc")
    starts = pd.to_datetime(df["start"], utc=True).sort_values().reset_index(drop=True)
    inferred = pd.Series(starts.diff().dropna().dt.total_seconds() / 60)
    if not inferred.empty:
        mode = inferred.mode()
        if not mode.empty:
            expected = float(mode.iloc[0])
            gaps = inferred[inferred != expected]
            if not gaps.empty:
                issues.append("time_gaps")
    return len(issues) == 0, issues



def merge_candles(historical: pd.DataFrame, live: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Merge historical cache with live candles and deduplicate by timestamp."""
    live_frame = pd.DataFrame(live) if not isinstance(live, pd.DataFrame) else live.copy()
    combined = pd.concat([historical, live_frame], ignore_index=True)
    if "start" in combined.columns:
        combined["start"] = pd.to_datetime(combined["start"], utc=True, unit="ms", errors="ignore")
    return combined.sort_values("start").drop_duplicates(subset=["start"], keep="last").reset_index(drop=True)
