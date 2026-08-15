"""Parquet storage for market data caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CANDLE_SCHEMA: dict[str, str] = {
    "start": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "turnover": "float64",
}


@dataclass
class CacheStats:
    """Simple metadata summary for a cached candle file."""

    symbol: str
    interval: str
    rows: int
    start: str | None
    end: str | None
    path: str
    size_bytes: int


class ParquetCache:
    """Simple parquet cache reader/writer with schema enforcement."""

    def __init__(self, base_dir: str = "data/cache") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, interval: str) -> Path:
        """Compute cache file path."""
        safe = symbol.replace("/", "_")
        return self.base_dir / f"{safe}_{interval}.parquet"

    def _ensure_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce a candle frame to the expected schema."""
        out = frame.copy()
        for column, dtype in CANDLE_SCHEMA.items():
            if column not in out.columns:
                out[column] = pd.NA if column == "start" else 0.0
            if column == "start":
                out[column] = pd.to_datetime(out[column], utc=True)
            else:
                out[column] = pd.to_numeric(out[column], errors="coerce").astype(dtype)
        ordered = out[list(CANDLE_SCHEMA.keys())].sort_values("start").drop_duplicates(subset=["start"], keep="last")
        return ordered.reset_index(drop=True)

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Load parquet if available."""
        path = self.path_for(symbol, interval)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return self._ensure_schema(frame)

    def save(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        """Save candles as parquet with enforced schema."""
        path = self.path_for(symbol, interval)
        clean = self._ensure_schema(frame)
        clean.to_parquet(path, index=False)

    def append_candles(self, symbol: str, interval: str, new_candles: pd.DataFrame) -> pd.DataFrame:
        """Append candles to an existing cache, deduplicating by timestamp."""
        existing = self.load(symbol, interval)
        if existing is None or existing.empty:
            merged = self._ensure_schema(new_candles)
        else:
            merged = self._ensure_schema(pd.concat([existing, new_candles], ignore_index=True))
        self.save(symbol, interval, merged)
        return merged

    def get_latest_timestamp(self, symbol: str, interval: str) -> pd.Timestamp | None:
        """Return the latest cached candle timestamp."""
        frame = self.load(symbol, interval)
        if frame is None or frame.empty:
            return None
        return pd.Timestamp(frame["start"].max())

    def prune_old_candles(self, symbol: str, interval: str, keep_days: int = 30) -> pd.DataFrame | None:
        """Remove candles older than ``keep_days`` from the cache."""
        frame = self.load(symbol, interval)
        if frame is None or frame.empty:
            return frame
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=keep_days)
        pruned = frame.loc[frame["start"] >= cutoff].reset_index(drop=True)
        self.save(symbol, interval, pruned)
        return pruned

    def get_stats(self, symbol: str, interval: str) -> CacheStats:
        """Return cache metadata for a symbol and interval."""
        path = self.path_for(symbol, interval)
        frame = self.load(symbol, interval)
        rows = 0 if frame is None else len(frame)
        start = None if frame is None or frame.empty else str(frame["start"].min())
        end = None if frame is None or frame.empty else str(frame["start"].max())
        size_bytes = path.stat().st_size if path.exists() else 0
        return CacheStats(symbol=symbol, interval=interval, rows=rows, start=start, end=end, path=str(path), size_bytes=size_bytes)

    def list_cached_symbols(self) -> list[tuple[str, str]]:
        """Return available cached ``(symbol, interval)`` pairs."""
        pairs: list[tuple[str, str]] = []
        for path in sorted(self.base_dir.glob("*.parquet")):
            stem = path.stem
            if "_" not in stem:
                continue
            symbol, interval = stem.rsplit("_", 1)
            pairs.append((symbol.replace("_", "/"), interval))
        return pairs
