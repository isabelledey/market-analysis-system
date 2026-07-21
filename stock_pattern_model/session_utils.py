"""Helpers for exchange-session-aware intraday grouping and filtering."""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.datetime_utils import to_zoneinfo


DEFAULT_REGULAR_SESSION_START = "09:30"
DEFAULT_REGULAR_SESSION_END = "16:00"
DEFAULT_SESSION_MODE = "regular"
SUPPORTED_SESSION_MODES = (
    "regular",
    "extended",
    "premarket",
    "regular-and-afterhours",
)

SESSION_MODE_SEGMENTS: dict[str, tuple[str, ...]] = {
    "regular": ("regular",),
    "extended": ("premarket", "regular", "afterhours"),
    "premarket": ("premarket",),
    "regular-and-afterhours": ("regular", "afterhours"),
}


def parse_session_clock(value: str) -> time:
    return time.fromisoformat(value)


def normalize_session_mode(value: str | None) -> str:
    if value is None:
        return DEFAULT_SESSION_MODE
    normalized = str(value).strip().lower()
    if normalized not in SESSION_MODE_SEGMENTS:
        raise ValueError(
            f"Unsupported session mode '{value}'. Supported values: {', '.join(SUPPORTED_SESSION_MODES)}"
        )
    return normalized


def session_segments_for_mode(session_mode: str | None) -> tuple[str, ...]:
    return SESSION_MODE_SEGMENTS[normalize_session_mode(session_mode)]


def session_mode_requires_extended_hours(session_mode: str | None) -> bool:
    return any(segment != "regular" for segment in session_segments_for_mode(session_mode))


def exchange_datetime_series(
    datetimes: pd.Series,
    exchange_timezone: str | ZoneInfo | None = None,
) -> pd.Series:
    parsed = pd.to_datetime(datetimes)
    if exchange_timezone is None:
        return parsed
    zone = to_zoneinfo(exchange_timezone)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    return parsed.dt.tz_convert(zone)


def session_date_series(
    datetimes: pd.Series,
    exchange_timezone: str | ZoneInfo | None = None,
) -> pd.Series:
    exchange_datetimes = exchange_datetime_series(datetimes, exchange_timezone)
    return exchange_datetimes.dt.strftime("%Y-%m-%d")


def session_segment_series(
    datetimes: pd.Series,
    *,
    exchange_timezone: str | ZoneInfo | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> pd.Series:
    exchange_datetimes = exchange_datetime_series(datetimes, exchange_timezone)
    start_clock = parse_session_clock(regular_session_start)
    end_clock = parse_session_clock(regular_session_end)
    local_times = exchange_datetimes.dt.time
    return pd.Series(
        [
            "premarket"
            if local_time < start_clock
            else "regular"
            if local_time < end_clock
            else "afterhours"
            for local_time in local_times
        ],
        index=exchange_datetimes.index,
        dtype="object",
    )


def regular_session_mask(
    datetimes: pd.Series,
    *,
    exchange_timezone: str | ZoneInfo | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> pd.Series:
    return session_segment_series(
        datetimes,
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    ) == "regular"


def allowed_session_mask(
    datetimes: pd.Series,
    *,
    session_mode: str | None = None,
    exchange_timezone: str | ZoneInfo | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> pd.Series:
    segments = session_segment_series(
        datetimes,
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    allowed_segments = set(session_segments_for_mode(session_mode))
    return segments.isin(allowed_segments)


def pattern_session_key_series(
    datetimes: pd.Series,
    *,
    exchange_timezone: str | ZoneInfo | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> pd.Series:
    session_dates = session_date_series(datetimes, exchange_timezone)
    segments = session_segment_series(
        datetimes,
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    return session_dates + ":" + segments
