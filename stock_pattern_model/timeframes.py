"""Single source of truth for the user-selectable analysis timeframes.

A *timeframe* is the duration of ONE candle (a trading day, a trading week, or a trading month).
It is not the amount of history that gets downloaded: the historical lookback is stored next to
it because a candle size needs enough completed candles to be analysed reliably.

Every value that depends on the selected timeframe (Yahoo Finance interval, lookback, trend
horizons, local-trend window, output labels) lives in ``TIMEFRAME_SPECS``. Nothing else in the
package should hard-code them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stock_pattern_model.exceptions import ConfigurationError


class Timeframe(str, Enum):
    """Supported candle sizes."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


@dataclass(frozen=True)
class TrendHorizons:
    """Number of completed candles used by the short, medium, and long trend windows."""

    short: int
    medium: int
    long: int

    def __post_init__(self) -> None:
        if not 1 <= self.short < self.medium < self.long:
            raise ConfigurationError(
                "Trend horizons must satisfy 1 <= short < medium < long, "
                f"got ({self.short}, {self.medium}, {self.long})."
            )

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.short, self.medium, self.long)


@dataclass(frozen=True)
class TimeframeSpec:
    """Everything that depends on the selected candle size."""

    timeframe: Timeframe
    label: str
    interval: str
    lookback_period: str
    trend_horizons: TrendHorizons
    local_trend_lookback_bars: int
    # "session" | "week" | "month": what one candle spans, used for labels and exchange-calendar logic.
    candle_span: str
    candle_description: str
    completed_candle_label: str
    # Sanity range for the number of completed candles a normal lookback should return.
    target_completed_candles: tuple[int, int] | None = None
    # Yahoo Finance period to retry with when the primary lookback returned too little history.
    fallback_lookback_period: str | None = None

    @property
    def minimum_completed_candles(self) -> int:
        """Fewest completed candles for which every trend horizon can be evaluated."""
        return self.trend_horizons.long

    @property
    def period_interval(self) -> tuple[str, str]:
        return (self.lookback_period, self.interval)


# Trend-horizon choices (all values are numbers of completed candles of the selected size).
#
# The previous implementation used 12/48/120 bars for every interval. The ranges below were
# requested per timeframe; within each range the value closest to the existing behaviour was kept,
# subject to two constraints of the existing trend code:
#   * the short window must hold enough candles for swing/pivot structure (pivots need 2 bars on
#     each side, and the snapshot needs at least min(8, horizon) candles), so the short windows sit
#     at the top of their ranges (10, 8, 6);
#   * the long window must be reachable inside the default lookback with margin, otherwise the long
#     trend would almost always report "Insufficient Data". Daily uses 100 (not 120) because six
#     months of trading days only contain about 125 completed candles.
# The local-trend window is the low end of each medium range (20, 13, 12): a recent, multi-candle
# read that stays clearly shorter than the medium trend.
TIMEFRAME_SPECS: dict[Timeframe, TimeframeSpec] = {
    Timeframe.DAILY: TimeframeSpec(
        timeframe=Timeframe.DAILY,
        label="Daily",
        interval="1d",
        lookback_period="6mo",
        trend_horizons=TrendHorizons(short=10, medium=50, long=100),
        local_trend_lookback_bars=20,
        candle_span="session",
        candle_description="one trading day",
        completed_candle_label="Latest Completed Trading Session",
        target_completed_candles=(120, 130),
    ),
    Timeframe.WEEKLY: TimeframeSpec(
        timeframe=Timeframe.WEEKLY,
        label="Weekly",
        interval="1wk",
        lookback_period="5y",
        trend_horizons=TrendHorizons(short=8, medium=26, long=104),
        local_trend_lookback_bars=13,
        candle_span="week",
        candle_description="one trading week",
        completed_candle_label="Latest Completed Trading Week",
        target_completed_candles=(230, 270),
    ),
    Timeframe.MONTHLY: TimeframeSpec(
        timeframe=Timeframe.MONTHLY,
        label="Monthly",
        interval="1mo",
        lookback_period="10y",
        trend_horizons=TrendHorizons(short=6, medium=24, long=60),
        local_trend_lookback_bars=12,
        candle_span="month",
        candle_description="one trading month",
        completed_candle_label="Latest Completed Trading Month",
        target_completed_candles=(110, 125),
        fallback_lookback_period="max",
    ),
}

# Names used before the timeframes were renamed; still accepted on the command line.
LEGACY_TIMEFRAME_ALIASES: dict[str, Timeframe] = {
    "1_DAY": Timeframe.DAILY,
    "1_WEEK": Timeframe.WEEKLY,
    "1_MONTH": Timeframe.MONTHLY,
}

SUPPORTED_TIMEFRAMES: tuple[str, ...] = tuple(timeframe.value for timeframe in Timeframe)

_SPEC_BY_INTERVAL: dict[str, TimeframeSpec] = {
    spec.interval: spec for spec in TIMEFRAME_SPECS.values()
}


def parse_timeframe(value: str | Timeframe) -> Timeframe:
    """Normalise a user-supplied timeframe (case-insensitive, legacy names accepted)."""
    if isinstance(value, Timeframe):
        return value
    key = str(value).strip().upper()
    if key in LEGACY_TIMEFRAME_ALIASES:
        return LEGACY_TIMEFRAME_ALIASES[key]
    try:
        return Timeframe(key)
    except ValueError as error:
        raise ConfigurationError(
            f"Unsupported timeframe '{value}'. Supported values: {', '.join(SUPPORTED_TIMEFRAMES)}."
        ) from error


def get_timeframe_spec(timeframe: str | Timeframe) -> TimeframeSpec:
    return TIMEFRAME_SPECS[parse_timeframe(timeframe)]


def spec_for_interval(interval: str | None) -> TimeframeSpec | None:
    """Timeframe spec for a candle interval, or None for intervals without one (e.g. intraday)."""
    if interval is None:
        return None
    return _SPEC_BY_INTERVAL.get(interval)


TIMEFRAME_TO_PERIOD_INTERVAL: dict[str, tuple[str, str]] = {
    timeframe.value: TIMEFRAME_SPECS[timeframe].period_interval for timeframe in Timeframe
}


_PERIOD_UNITS = {"d": "day", "wk": "week", "mo": "month", "y": "year"}


def format_lookback_period(period: str | None) -> str:
    """Human-readable form of a Yahoo Finance period string ('6mo' -> '6 months (6mo)')."""
    if not period:
        return "Not specified"
    text = str(period).strip().lower()
    if text == "max":
        return "Maximum available history (max)"
    digits = "".join(character for character in text if character.isdigit())
    unit = text[len(digits):]
    if not digits or unit not in _PERIOD_UNITS:
        return str(period)
    count = int(digits)
    noun = _PERIOD_UNITS[unit]
    return f"{count} {noun}{'' if count == 1 else 's'} ({period})"
