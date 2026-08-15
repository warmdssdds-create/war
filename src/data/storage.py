"""Parquet storage for market data caches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    """Simple parquet cache reader/writer."""

    def __init__(self, base_dir: str = "data/cache") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, interval: str) -> Path:
        """Compute cache file path."""
        safe = symbol.replace("/", "_")
        return self.base_dir / f"{safe}_{interval}.parquet"

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Load parquet if available."""
        path = self.path_for(symbol, interval)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        """Save candles as parquet."""
        path = self.path_for(symbol, interval)
        frame.to_parquet(path, index=False)
