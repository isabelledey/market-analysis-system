"""Centralized timezone-aware datetime conversion and formatting helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# yfinance-style interval tokens that pandas' Timedelta parser rejects outright
# (it only understands "W"/"D", not "wk", and has no fixed-duration unit for
# calendar months at all). 1mo/3mo are approximated as 30/90 days since a
# Timedelta cannot represent a true calendar month.
_INTERVAL_TIMEDELTA_ALIASES = {
    "1wk": "7D",
    "1mo": "30D",
    "3mo": "90D",
}


# While an analysis of daily/weekly/monthly candles runs, user-facing candle timestamps are shown
# as dates ("2026-09-18", "2026-09", ...) in the exchange timezone. A date-only candle has no
# meaningful time of day, and converting its exchange-local midnight into the display timezone
# would print an artificial hour such as "07:00 Asia/Jerusalem". The scope is set by
# analysis.analyze_dataframe; ``exact=True`` opts a value (e.g. the analysis time) out.
_CANDLE_DISPLAY_SCOPE: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "candle_display_scope",
    default=None,
)


@contextmanager
def candle_display_scope(span: str | None, exchange_timezone: str | None) -> Iterator[None]:
    """Render candle timestamps as exchange-local dates for 'session'/'week'/'month' candles."""
    token = _CANDLE_DISPLAY_SCOPE.set((span, exchange_timezone) if span else None)
    try:
        yield
    finally:
        _CANDLE_DISPLAY_SCOPE.reset(token)


def format_candle_date(value: Any, span: str, exchange_timezone: str | ZoneInfo | None = None) -> str:
    """Format a candle timestamp as its exchange-local date ('YYYY-MM' for monthly candles)."""
    timestamp = ensure_timezone_aware(value)
    if exchange_timezone is not None:
        timestamp = convert_to_timezone(timestamp, exchange_timezone)
    return timestamp.strftime("%Y-%m" if span == "month" else "%Y-%m-%d")


def _scoped_candle_date(value: Any) -> str | None:
    scope = _CANDLE_DISPLAY_SCOPE.get()
    if scope is None:
        return None
    span, exchange_timezone = scope
    return format_candle_date(value, span, exchange_timezone)


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Convert a yfinance-style interval string into a pandas Timedelta."""
    return pd.to_timedelta(_INTERVAL_TIMEDELTA_ALIASES.get(interval, interval))


def to_zoneinfo(timezone: str | ZoneInfo) -> ZoneInfo:
    """Return a ZoneInfo instance for a timezone name or ZoneInfo object."""
    if isinstance(timezone, ZoneInfo):
        return timezone
    return ZoneInfo(str(timezone))


def ensure_timezone_aware(value: Any, *, field_name: str = "datetime") -> pd.Timestamp:
    """Normalize a value into a timezone-aware pandas Timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return timestamp


def convert_to_timezone(value: Any, timezone: str | ZoneInfo) -> pd.Timestamp:
    """Convert one authoritative instant into another timezone without changing the instant."""
    timestamp = ensure_timezone_aware(value)
    zone = to_zoneinfo(timezone)
    return pd.Timestamp(timestamp.to_pydatetime().astimezone(zone))


def format_iso_timestamp(
    value: Any,
    *,
    timezone: str | ZoneInfo | None = None,
    timespec: str = "minutes",
) -> str:
    """Format a timezone-aware datetime as ISO-8601."""
    timestamp = ensure_timezone_aware(value)
    if timezone is not None:
        timestamp = convert_to_timezone(timestamp, timezone)
    return timestamp.isoformat(timespec=timespec)


def format_display_datetime(
    value: Any,
    timezone: str | ZoneInfo,
    *,
    exact: bool = False,
) -> str:
    """Format a user-facing timestamp with offset and timezone name.

    Inside a daily/weekly/monthly ``candle_display_scope`` candle timestamps are rendered as dates
    instead, unless ``exact`` is set.
    """
    if not exact:
        scoped = _scoped_candle_date(value)
        if scoped is not None:
            return scoped
    converted = convert_to_timezone(value, timezone)
    zone = to_zoneinfo(timezone)
    zone_name = getattr(zone, "key", str(zone))
    return f"{converted.strftime('%Y-%m-%d %H:%M:%S%z')} {zone_name}"


def format_compact_display_datetime(
    value: Any,
    timezone: str | ZoneInfo,
    *,
    exact: bool = False,
) -> str:
    """Format a compact user-facing timestamp for explanations (date-only inside a candle scope)."""
    if not exact:
        scoped = _scoped_candle_date(value)
        if scoped is not None:
            return scoped
    converted = convert_to_timezone(value, timezone)
    zone = to_zoneinfo(timezone)
    zone_name = getattr(zone, "key", str(zone))
    return f"{converted.strftime('%Y-%m-%d %H:%M')} {zone_name}"
