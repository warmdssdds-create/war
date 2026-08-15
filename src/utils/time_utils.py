"""Utility helpers for UTC timestamps, candle alignment, and session boundaries.

The trading bot works with a mixture of Python ``datetime`` objects, pandas
``Timestamp`` values, integer milliseconds from Bybit, and ISO strings from
configuration or logs.  Keeping the conversions in a single module reduces
subtle off-by-one-candle errors and makes tests deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Iterable

import pandas as pd


DatetimeLike = datetime | pd.Timestamp | int | float | str


_INTERVAL_ALIASES: dict[str, int] = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 24 * 60,
    "1D": 24 * 60,
}


def utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(tz=UTC)


def ensure_utc(ts: DatetimeLike | None) -> datetime:
    """Normalise many timestamp representations to an aware UTC datetime.

    Parameters
    ----------
    ts:
        May be ``None`` (uses current UTC time), a datetime-like object,
        a unix timestamp in seconds or milliseconds, or an ISO string.
    """

    if ts is None:
        return utc_now()
    if isinstance(ts, pd.Timestamp):
        stamp = ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)
        return stamp.to_pydatetime()
    if isinstance(ts, datetime):
        return ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
    if isinstance(ts, (int, float)):
        value = float(ts)
        if value > 10_000_000_000:
            value /= 1000.0
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(ts, str):
        raw = ts.strip()
        if raw.isdigit():
            return ensure_utc(int(raw))
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")


def interval_to_minutes(interval: int | str) -> int:
    """Convert Bybit style interval labels to integer minutes."""
    if isinstance(interval, int):
        return interval
    text = str(interval).strip().upper()
    if text.isdigit():
        return int(text)
    if text not in _INTERVAL_ALIASES:
        raise ValueError(f"Unsupported interval: {interval}")
    return _INTERVAL_ALIASES[text]


def to_millis(ts: datetime) -> int:
    """Convert datetime to unix milliseconds."""
    return int(ensure_utc(ts).timestamp() * 1000)


def to_seconds(ts: DatetimeLike | None = None) -> int:
    """Convert a timestamp-like value to unix seconds."""
    return int(ensure_utc(ts).timestamp())


def parse_bybit_ts(ts_str_or_int: DatetimeLike) -> datetime:
    """Parse a Bybit timestamp or ISO string into a UTC datetime.

    Bybit commonly uses integer milliseconds, but several endpoints return text.
    This helper accepts both formats.
    """

    return ensure_utc(ts_str_or_int)


def format_ts(ts: DatetimeLike | None) -> str:
    """Format a timestamp as ``YYYY-mm-dd HH:MM:SS UTC``."""
    value = ensure_utc(ts)
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def session_start_utc(ts: DatetimeLike | None = None) -> datetime:
    """Return midnight UTC for the supplied day."""
    value = ensure_utc(ts)
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def candle_open_time(ts: DatetimeLike, interval_min: int | str) -> datetime:
    """Snap a timestamp down to the candle open time in UTC."""
    value = ensure_utc(ts)
    minutes = interval_to_minutes(interval_min)
    start = session_start_utc(value)
    elapsed = int((value - start).total_seconds() // 60)
    snapped = elapsed - (elapsed % minutes)
    return start + timedelta(minutes=snapped)


def next_candle_time(ts: DatetimeLike, interval_min: int | str) -> datetime:
    """Return the next candle open after ``ts``."""
    current_open = candle_open_time(ts, interval_min)
    return current_open + timedelta(minutes=interval_to_minutes(interval_min))


def previous_candle_time(ts: DatetimeLike, interval_min: int | str) -> datetime:
    """Return the previous candle open before ``ts``."""
    return candle_open_time(ts, interval_min) - timedelta(minutes=interval_to_minutes(interval_min))


def is_candle_confirmed(
    candle_start_ts: DatetimeLike,
    interval_min: int | str,
    current_ts: DatetimeLike | None = None,
) -> bool:
    """Check whether the full candle interval has elapsed."""
    start = ensure_utc(candle_start_ts)
    now = ensure_utc(current_ts)
    return now >= start + timedelta(minutes=interval_to_minutes(interval_min))


def candle_close_time(ts: DatetimeLike, interval_min: int | str) -> datetime:
    """Return the close time for the candle containing ``ts``."""
    return next_candle_time(ts, interval_min)


def candle_range(
    start_ts: DatetimeLike,
    end_ts: DatetimeLike,
    interval_min: int | str,
) -> list[datetime]:
    """Generate candle open timestamps between two endpoints, inclusive.

    The returned timestamps are aligned to the interval boundary.  If
    ``end_ts`` falls inside a candle, the open for that candle is included.
    """

    start = candle_open_time(start_ts, interval_min)
    end = candle_open_time(end_ts, interval_min)
    step = timedelta(minutes=interval_to_minutes(interval_min))
    points: list[datetime] = []
    current = start
    while current <= end:
        points.append(current)
        current += step
    return points


def iter_candle_range(
    start_ts: DatetimeLike,
    end_ts: DatetimeLike,
    interval_min: int | str,
) -> Iterable[datetime]:
    """Yield aligned candle opens across a range."""
    for point in candle_range(start_ts, end_ts, interval_min):
        yield point


def seconds_since(ts: DatetimeLike) -> float:
    """Return elapsed seconds from ``ts`` until now."""
    return (utc_now() - ensure_utc(ts)).total_seconds()


def minutes_since(ts: DatetimeLike) -> float:
    """Return elapsed minutes from ``ts`` until now."""
    return seconds_since(ts) / 60.0


def time_to_next_candle(interval_min: int | str) -> float:
    """Return seconds until the next candle close/open boundary."""
    now = utc_now()
    nxt = next_candle_time(now, interval_min)
    return max((nxt - now).total_seconds(), 0.0)


def pandas_timestamp(ts: DatetimeLike | None = None) -> pd.Timestamp:
    """Return a pandas ``Timestamp`` normalised to UTC."""
    return pd.Timestamp(ensure_utc(ts))


def normalize_timestamp_series(values: Iterable[DatetimeLike]) -> pd.Series:
    """Convert a sequence of timestamps into an aware UTC pandas Series."""
    return pd.Series([pandas_timestamp(value) for value in values], dtype="datetime64[ns, UTC]")
