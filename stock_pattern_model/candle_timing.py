"""Exchange-calendar-aware timing of daily, weekly, and monthly candles.

A candle is *completed* only once the final trading session it covers has closed:

* Daily   - the session of that day has closed (early closes included).
* Weekly  - the last trading session of the trading week has closed.
* Monthly - the last trading session of the trading month has closed.

Plain date arithmetic (``start + 1 day``, ``start + 7 days``, ``start + 30 days``) is wrong for
all three: a daily candle stamped 00:00 only "ends" at the next midnight although the session
closed at 16:00, a week ending on a holiday-shortened Thursday looks incomplete on Friday, and a
31-day month looks complete a day too early. This module derives the completion time from the
exchange calendar instead (holidays, weekends, early closes, and the exchange timezone including
daylight-saving transitions). Intraday intervals keep the simple ``start + interval`` rule.

If no calendar is available (unknown exchange, or ``exchange_calendars`` not installed), a
weekday-based approximation is used and a warning is reported; it never treats a candle as
complete earlier than the calendar would, apart from holidays it cannot know about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.datetime_utils import interval_to_timedelta
from stock_pattern_model.exceptions import DataValidationError

try:  # The calendar package is a declared dependency, but degrade gracefully if it is missing.
    import exchange_calendars as _xcals
except ImportError:  # pragma: no cover - exercised only when the dependency is absent
    _xcals = None

LOGGER = logging.getLogger(__name__)

BAR_END_COLUMN = "Bar_End"
DEFAULT_REGULAR_SESSION_END = "16:00"

# Candle interval -> what one candle spans.
CALENDAR_INTERVAL_SPANS: dict[str, str] = {"1d": "session", "1wk": "week", "1mo": "month"}

# Package exchange-calendar names that exchange_calendars does not know under the same name.
_CALENDAR_NAME_ALIASES = {"AMEX": "XASE"}

SOURCE_NOMINAL = "nominal"
SOURCE_EXCHANGE_CALENDAR = "exchange_calendar"
SOURCE_WEEKDAY_FALLBACK = "weekday_fallback"


def is_calendar_interval(interval: str) -> bool:
    """True for the candle sizes whose completion follows the exchange calendar."""
    return interval in CALENDAR_INTERVAL_SPANS


def candle_span(interval: str) -> str | None:
    """'session', 'week', or 'month' for calendar candles, None for other intervals."""
    return CALENDAR_INTERVAL_SPANS.get(interval)


@lru_cache(maxsize=16)
def load_exchange_calendar(name: str | None) -> Any | None:
    """Return the exchange_calendars calendar for a package calendar name, or None."""
    if _xcals is None or not name:
        return None
    key = _CALENDAR_NAME_ALIASES.get(str(name).strip().upper(), str(name).strip())
    try:
        return _xcals.get_calendar(key)
    except Exception:  # noqa: BLE001 - unknown calendar name
        LOGGER.debug("Exchange calendar %r is not available.", name)
        return None


def _span_key(timestamps: pd.Series, span: str) -> pd.Series:
    """Map tz-naive local dates to the key of the period (day / Monday / first of month)."""
    dates = pd.to_datetime(timestamps).dt.normalize()
    if span == "session":
        return dates
    if span == "week":
        return dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    return dates.dt.to_period("M").dt.to_timestamp()


@lru_cache(maxsize=32)
def _calendar_period_closes(name: str, span: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """period key -> close (UTC) of the last trading session of that period, from the calendar."""
    calendar = load_exchange_calendar(name)
    closes = calendar.closes
    keys = _span_key(pd.Series(closes.index, index=closes.index), span)
    last_close = pd.Series(closes.to_numpy(), index=keys.to_numpy()).groupby(level=0).max()
    return {pd.Timestamp(key): pd.Timestamp(value) for key, value in last_close.items()}


def _parse_clock(value: str) -> time:
    return time.fromisoformat(value)


@dataclass(frozen=True)
class CandleClock:
    """Computes when candles of one interval complete for one exchange."""

    interval: str
    exchange_timezone: str | None = None
    exchange_calendar: str | None = None
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END

    @property
    def span(self) -> str | None:
        return candle_span(self.interval)

    @property
    def calendar_based(self) -> bool:
        return self.span is not None

    def _calendar(self) -> Any | None:
        return load_exchange_calendar(self.exchange_calendar) if self.calendar_based else None

    @property
    def source(self) -> str:
        if not self.calendar_based:
            return SOURCE_NOMINAL
        return SOURCE_EXCHANGE_CALENDAR if self._calendar() is not None else SOURCE_WEEKDAY_FALLBACK

    @property
    def warnings(self) -> list[str]:
        if self.source != SOURCE_WEEKDAY_FALLBACK:
            return []
        return [
            f"No exchange calendar was available for {self.interval} candles "
            f"({self.exchange_calendar or 'unknown exchange'}); candle completion uses a weekday-based "
            f"{self.regular_session_end} close, so exchange holidays and early closes are not applied."
        ]

    def _zone(self, timestamps: pd.Series) -> ZoneInfo:
        if self.exchange_timezone:
            return ZoneInfo(str(self.exchange_timezone))
        tz = timestamps.dt.tz
        if tz is None:
            raise DataValidationError("Candle timestamps must be timezone-aware.")
        return ZoneInfo(str(tz))

    def _fallback_close(self, key: pd.Timestamp, zone: ZoneInfo) -> pd.Timestamp:
        """Weekday-based completion of the period starting at ``key`` (a naive local date)."""
        if self.span == "session":
            last_day = key
        elif self.span == "week":
            last_day = key + pd.Timedelta(days=4)
        else:
            last_day = key + pd.offsets.MonthEnd(0)
            while last_day.weekday() >= 5:
                last_day -= pd.Timedelta(days=1)
        clock = _parse_clock(self.regular_session_end)
        local_close = pd.Timestamp.combine(last_day.date(), clock)
        return local_close.tz_localize(zone, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")

    def bar_ends(self, datetimes: pd.Series) -> pd.Series:
        """Completion time of every candle, aligned with ``datetimes``."""
        parsed = pd.to_datetime(datetimes)
        if not self.calendar_based:
            return parsed + interval_to_timedelta(self.interval)
        zone = self._zone(parsed)
        if parsed.dt.tz is None:
            raise DataValidationError("Candle timestamps must be timezone-aware.")
        local_naive = parsed.dt.tz_convert(zone).dt.tz_localize(None)
        keys = _span_key(local_naive, self.span)
        calendar_closes = (
            _calendar_period_closes(self.exchange_calendar, self.span)
            if self._calendar() is not None
            else {}
        )
        resolved: dict[pd.Timestamp, pd.Timestamp] = {}
        for key in keys.drop_duplicates():
            close = calendar_closes.get(pd.Timestamp(key))
            resolved[key] = close if close is not None else self._fallback_close(pd.Timestamp(key), zone)
        ends = pd.Series(
            [resolved[key] for key in keys],
            index=parsed.index,
            dtype="datetime64[ns, UTC]",
        )
        return ends.dt.tz_convert(zone)

    def bar_end(self, start: pd.Timestamp) -> pd.Timestamp:
        """Completion time of a single candle."""
        series = pd.Series([pd.Timestamp(start)])
        return pd.Timestamp(self.bar_ends(series).iloc[0])

    def trading_dates(self, start: pd.Timestamp) -> tuple[date, date]:
        """First and last trading dates covered by the candle that starts at ``start``."""
        if not self.calendar_based:
            local = pd.Timestamp(start)
            return local.date(), local.date()
        zone = self._zone(pd.Series([pd.Timestamp(start)]))
        local_start = pd.Timestamp(start).tz_convert(zone).tz_localize(None).normalize()
        key = _span_key(pd.Series([local_start]), self.span).iloc[0]
        if self.span == "session":
            return key.date(), key.date()
        period_end = key + (pd.Timedelta(days=6) if self.span == "week" else pd.offsets.MonthEnd(0))
        calendar = self._calendar()
        if calendar is not None:
            try:
                sessions = calendar.sessions_in_range(key, period_end)
            except Exception:  # noqa: BLE001 - range outside the calendar bounds
                sessions = []
            if len(sessions):
                return sessions[0].date(), sessions[-1].date()
        weekdays = pd.bdate_range(key, period_end)
        return weekdays[0].date(), weekdays[-1].date()


def bar_end_from_frame(df: pd.DataFrame, index: int, interval: str) -> pd.Timestamp:
    """Completion time of the candle at positional ``index``.

    Uses the pre-computed ``Bar_End`` column when the frame carries one (frames produced by the
    analysis pipeline do); otherwise falls back to ``start + interval`` (hand-built frames).
    """
    if BAR_END_COLUMN in df.columns:
        value = df[BAR_END_COLUMN].iloc[index]
        if not pd.isna(value):
            return pd.Timestamp(value)
    return pd.Timestamp(df.iloc[index]["Datetime"]) + interval_to_timedelta(interval)


def start_from_bar_end(df: pd.DataFrame, interval: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """Reverse lookup ``bar end -> candle start`` matching :func:`bar_end_from_frame`."""
    if BAR_END_COLUMN in df.columns:
        return {
            pd.Timestamp(end): pd.Timestamp(start)
            for end, start in zip(df[BAR_END_COLUMN], df["Datetime"])
            if not pd.isna(end)
        }
    delta = interval_to_timedelta(interval)
    return {pd.Timestamp(start) + delta: pd.Timestamp(start) for start in df["Datetime"]}
