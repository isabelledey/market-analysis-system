"""Centralized timezone-aware datetime conversion and formatting helpers."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


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
) -> str:
    """Format a user-facing timestamp with offset and timezone name."""
    converted = convert_to_timezone(value, timezone)
    zone = to_zoneinfo(timezone)
    zone_name = getattr(zone, "key", str(zone))
    return f"{converted.strftime('%Y-%m-%d %H:%M:%S%z')} {zone_name}"


def format_compact_display_datetime(
    value: Any,
    timezone: str | ZoneInfo,
) -> str:
    """Format a compact user-facing timestamp for explanations."""
    converted = convert_to_timezone(value, timezone)
    zone = to_zoneinfo(timezone)
    zone_name = getattr(zone, "key", str(zone))
    return f"{converted.strftime('%Y-%m-%d %H:%M')} {zone_name}"
