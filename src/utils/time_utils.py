"""Time utility helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(tz=UTC)


def to_millis(ts: datetime) -> int:
    """Convert datetime to unix milliseconds."""
    return int(ts.timestamp() * 1000)
