"""Daily / Weekly / Monthly timeframe behaviour.

A timeframe is the duration of ONE candle (a trading session, a trading week, a trading month).
These tests cover the central configuration, exchange-calendar aware completed-candle rules
(weekends, holidays, early closes, DST in the US and Israel), timeframe-specific trend horizons,
completed-candle-only pattern detection, candle-count based lifecycle, and symbol resolution.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

import stock_pattern_model.market_data as market_data_module
from stock_pattern_model.analysis import analyze_dataframe, analyze_stock
from stock_pattern_model.candle_timing import BAR_END_COLUMN, CandleClock
from stock_pattern_model.cli import ExitCode, main
from stock_pattern_model.context import build_analysis_context
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import ConfigurationError
from stock_pattern_model.formatters import format_analysis_json, format_analysis_text
from stock_pattern_model.timeframes import (
    LEGACY_TIMEFRAME_ALIASES,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_SPECS,
    Timeframe,
    format_lookback_period,
    get_timeframe_spec,
    parse_timeframe,
    spec_for_interval,
)

NY = "America/New_York"
JERUSALEM = "Asia/Jerusalem"

PYPL = ResolvedInstrument(
    input_identifier="PYPL",
    symbol="PYPL",
    name="PayPal Holdings",
    exchange="NASDAQ",
    currency="USD",
    exchange_timezone=NY,
)
TEVA = ResolvedInstrument(
    input_identifier="TEVA.TA",
    symbol="TEVA.TA",
    name="Teva Pharmaceutical",
    exchange="TASE",
    currency="ILS",
    exchange_timezone=JERUSALEM,
)


# ---------------------------------------------------------------------------
# Synthetic yfinance-shaped data: daily bars are stamped 00:00 exchange-local, weekly bars on the
# Monday of the week, monthly bars on the 1st. The newest bar is the still-forming period.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _cached_frames(calendar: str, tz: str, through: str, start: str) -> dict[str, pd.DataFrame]:
    sessions = xcals.get_calendar(calendar).sessions_in_range(start, through)
    rng = np.random.default_rng(7)
    count = len(sessions)
    close = 60 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, count)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, count))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, count)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, count)))
    volume = rng.integers(6_000_000, 20_000_000, count).astype(float)
    daily = pd.DataFrame(
        {
            "Datetime": pd.DatetimeIndex(sessions.tz_localize(None)),
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )

    def aggregate(key: pd.Series) -> pd.DataFrame:
        grouped = daily.groupby(key, sort=True)
        out = pd.DataFrame(
            {
                "Open": grouped["Open"].first(),
                "High": grouped["High"].max(),
                "Low": grouped["Low"].min(),
                "Close": grouped["Close"].last(),
                "Volume": grouped["Volume"].sum(),
            }
        )
        out.index.name = "Datetime"
        return out.reset_index()

    week_key = daily["Datetime"] - pd.to_timedelta(daily["Datetime"].dt.weekday, unit="D")
    month_key = daily["Datetime"].dt.to_period("M").dt.to_timestamp()
    frames = {"1d": daily.copy(), "1wk": aggregate(week_key), "1mo": aggregate(month_key)}
    # Keep each series long enough for its own trend horizons while keeping the tests fast.
    first_kept = {"1d": "2025-06-02", "1wk": "2021-01-04", "1mo": "2016-01-01"}
    for interval, frame in frames.items():
        frames[interval] = frame.loc[frame["Datetime"] >= first_kept[interval]].reset_index(drop=True)
        frames[interval]["Datetime"] = pd.to_datetime(frames[interval]["Datetime"]).dt.tz_localize(tz)
    return frames


def frames(
    through: str,
    *,
    calendar: str = "XNAS",
    tz: str = NY,
    start: str = "2014-01-02",
) -> dict[str, pd.DataFrame]:
    return {key: value.copy() for key, value in _cached_frames(calendar, tz, through, start).items()}


def run(
    frame: pd.DataFrame,
    interval: str,
    as_of: str,
    *,
    instrument: ResolvedInstrument = PYPL,
    tz: str = NY,
    **kwargs,
) -> dict:
    return analyze_dataframe(
        df=frame.copy(),
        symbol=instrument.symbol,
        interval=interval,
        as_of=pd.Timestamp(as_of, tz=tz),
        display_timezone=JERUSALEM,
        instrument=instrument,
        **kwargs,
    )


def latest_value(result: dict) -> str:
    return result["latest_completed_candle"]["value"]


# ---------------------------------------------------------------------------
# Central configuration and naming
# ---------------------------------------------------------------------------


def test_timeframes_are_named_daily_weekly_monthly() -> None:
    assert SUPPORTED_TIMEFRAMES == ("DAILY", "WEEKLY", "MONTHLY")
    assert [TIMEFRAME_SPECS[timeframe].label for timeframe in Timeframe] == ["Daily", "Weekly", "Monthly"]


def test_each_timeframe_maps_to_one_candle_interval_and_a_lookback() -> None:
    expected = {
        Timeframe.DAILY: ("1d", "6mo"),
        Timeframe.WEEKLY: ("1wk", "5y"),
        Timeframe.MONTHLY: ("1mo", "10y"),
    }
    for timeframe, (interval, lookback) in expected.items():
        spec = get_timeframe_spec(timeframe)
        assert (spec.interval, spec.lookback_period) == (interval, lookback)
        assert spec_for_interval(interval) is spec
    assert TIMEFRAME_SPECS[Timeframe.MONTHLY].fallback_lookback_period == "max"
    assert spec_for_interval("15m") is None


def test_trend_horizons_stay_inside_the_requested_ranges() -> None:
    ranges = {
        Timeframe.DAILY: ((5, 10), (20, 50), (100, 120)),
        Timeframe.WEEKLY: ((4, 8), (13, 26), (52, 104)),
        Timeframe.MONTHLY: ((3, 6), (12, 24), (36, 60)),
    }
    for timeframe, (short, medium, long) in ranges.items():
        horizons = TIMEFRAME_SPECS[timeframe].trend_horizons
        assert short[0] <= horizons.short <= short[1]
        assert medium[0] <= horizons.medium <= medium[1]
        assert long[0] <= horizons.long <= long[1]
        assert horizons.short < horizons.medium < horizons.long


@pytest.mark.parametrize(
    "value,expected",
    [
        ("DAILY", Timeframe.DAILY),
        ("daily", Timeframe.DAILY),
        ("Weekly", Timeframe.WEEKLY),
        ("MONTHLY", Timeframe.MONTHLY),
        ("1_DAY", Timeframe.DAILY),
        ("1_week", Timeframe.WEEKLY),
        ("1_MONTH", Timeframe.MONTHLY),
        (Timeframe.WEEKLY, Timeframe.WEEKLY),
    ],
)
def test_parse_timeframe_accepts_new_and_legacy_names(value, expected) -> None:
    assert parse_timeframe(value) is expected
    assert set(LEGACY_TIMEFRAME_ALIASES) == {"1_DAY", "1_WEEK", "1_MONTH"}


def test_parse_timeframe_rejects_unknown_names() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported timeframe"):
        parse_timeframe("2_WEEKS")


def test_lookback_display_is_human_readable() -> None:
    assert format_lookback_period("6mo") == "6 months (6mo)"
    assert format_lookback_period("5y") == "5 years (5y)"
    assert format_lookback_period("1y") == "1 year (1y)"
    assert format_lookback_period("max") == "Maximum available history (max)"


def test_default_lookbacks_hold_enough_completed_candles() -> None:
    calendar = xcals.get_calendar("XNAS")
    end = pd.Timestamp("2026-09-18")
    daily = len(calendar.sessions_in_range(end - pd.DateOffset(months=6), end))
    assert 118 <= daily <= 132  # roughly 120-130 completed daily candles in six months
    weekly = len(calendar.sessions_in_range(end - pd.DateOffset(years=5), end)) / 5
    assert weekly >= 250
    monthly = len(calendar.sessions_in_range(end - pd.DateOffset(years=10), end)) / 21
    assert monthly >= 118
    assert TIMEFRAME_SPECS[Timeframe.DAILY].minimum_completed_candles <= daily


def test_timeframe_constants_are_defined_only_in_the_central_module() -> None:
    package = Path(__file__).resolve().parents[1] / "stock_pattern_model"
    offenders = []
    for path in package.glob("*.py"):
        if path.name in {"timeframes.py", "candle_timing.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if '"6mo"' in text and '"1d"' in text and "TIMEFRAME" not in text:
            offenders.append(path.name)
        assert "One day" not in text and "One week" not in text and "One month" not in text, path.name
    assert offenders == []


# ---------------------------------------------------------------------------
# 1-2. Daily: today's candle is excluded until the NASDAQ session ends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of,expected_session",
    [
        ("2026-09-18 09:30:00", "2026-09-17"),  # market opens: today's candle is still forming
        ("2026-09-18 10:00:00", "2026-09-17"),
        ("2026-09-18 15:59:59", "2026-09-17"),
        ("2026-09-18 16:00:00", "2026-09-18"),  # the closing bell completes the session
        ("2026-09-18 16:00:01", "2026-09-18"),
        ("2026-09-18 21:00:00", "2026-09-18"),
    ],
)
def test_daily_candle_is_complete_only_after_the_session_closes(as_of: str, expected_session: str) -> None:
    data = frames("2026-09-18")["1d"]
    result = run(data, "1d", as_of)

    assert latest_value(result) == expected_session
    assert result["latest_completed_candle"]["label"] == "Latest Completed Trading Session"
    assert result["completed_candles_used"] == (len(data) - 1 if expected_session == "2026-09-17" else len(data))


def test_daily_latest_price_is_the_last_completed_close() -> None:
    data = frames("2026-09-18")["1d"]
    data.loc[data.index[-1], "Close"] = 999.0  # the still-forming candle must never leak in
    data.loc[data.index[-1], "High"] = 1000.0

    forming = run(data, "1d", "2026-09-18 10:00:00")
    closed = run(data, "1d", "2026-09-18 16:00:01")

    assert forming["latest_close"] == round(float(data["Close"].iloc[-2]), 2)
    assert closed["latest_close"] == 999.0


# ---------------------------------------------------------------------------
# 3-4. Weekly: the current week is excluded until its final trading session has ended
# ---------------------------------------------------------------------------


def test_weekly_mid_week_excludes_the_current_week() -> None:
    data = frames("2026-09-16")["1wk"]  # Wednesday: the newest weekly bar is still forming
    result = run(data, "1wk", "2026-09-16 12:00:00")

    # Monday 2026-09-07 was Labor Day, so that trading week starts on Tuesday.
    assert latest_value(result) == "2026-09-08 to 2026-09-11"
    assert result["latest_completed_candle"]["label"] == "Latest Completed Trading Week"
    assert result["completed_candles_used"] == len(data) - 1


@pytest.mark.parametrize(
    "as_of,expected_week",
    [
        ("2026-09-18 15:59:59", "2026-09-08 to 2026-09-11"),  # Friday, one second before the close
        ("2026-09-18 16:00:00", "2026-09-14 to 2026-09-18"),
        ("2026-09-19 12:00:00", "2026-09-14 to 2026-09-18"),  # Saturday
        ("2026-09-20 12:00:00", "2026-09-14 to 2026-09-18"),  # Sunday
    ],
)
def test_weekly_candle_completes_after_the_final_session_of_the_week(as_of: str, expected_week: str) -> None:
    data = frames("2026-09-18")["1wk"]
    result = run(data, "1wk", as_of)

    assert latest_value(result) == expected_week


# ---------------------------------------------------------------------------
# 5-6. Monthly: the current month is excluded until its final trading session has ended
# ---------------------------------------------------------------------------


def test_monthly_before_month_end_excludes_the_current_month() -> None:
    data = frames("2026-09-29")["1mo"]
    result = run(data, "1mo", "2026-09-29 12:00:00")

    assert latest_value(result) == "2026-08"
    assert result["latest_completed_candle"]["label"] == "Latest Completed Trading Month"
    assert result["completed_candles_used"] == len(data) - 1


@pytest.mark.parametrize(
    "as_of,expected_month",
    [
        ("2026-09-30 15:59:59", "2026-08"),
        ("2026-09-30 16:00:00", "2026-09"),
        ("2026-10-01 09:00:00", "2026-09"),
    ],
)
def test_monthly_candle_completes_after_the_final_session_of_the_month(as_of: str, expected_month: str) -> None:
    data = frames("2026-09-30")["1mo"]
    if as_of.startswith("2026-10"):
        data = data  # October's candle does not exist yet; September is the newest bar
    result = run(data, "1mo", as_of)

    assert latest_value(result) == expected_month


def test_thirty_one_day_month_is_not_completed_a_day_early() -> None:
    # The former nominal 30-day arithmetic completed August (31 days) at 00:00 on August 31.
    data = frames("2026-08-31")["1mo"]
    result = run(data, "1mo", "2026-08-31 12:00:00")

    assert latest_value(result) == "2026-07"


# ---------------------------------------------------------------------------
# 7. Weekends, exchange holidays and early-close sessions
# ---------------------------------------------------------------------------


def test_weekend_does_not_create_a_candle_and_keeps_the_friday_session() -> None:
    data = frames("2026-09-18")["1d"]

    assert latest_value(run(data, "1d", "2026-09-19 12:00:00")) == "2026-09-18"  # Saturday
    assert latest_value(run(data, "1d", "2026-09-20 12:00:00")) == "2026-09-18"  # Sunday


def test_exchange_holiday_keeps_the_previous_session_as_latest() -> None:
    data = frames("2026-09-04")["1d"]  # Labor Day Monday 2026-09-07 has no candle

    result = run(data, "1d", "2026-09-07 12:00:00")

    assert latest_value(result) == "2026-09-04"


def test_early_close_session_completes_at_the_early_close() -> None:
    # Day after Thanksgiving 2025-11-28: NASDAQ closes at 13:00 ET.
    data = frames("2025-11-28")["1d"]

    assert latest_value(run(data, "1d", "2025-11-28 12:59:00")) == "2025-11-26"  # Thanksgiving was closed
    assert latest_value(run(data, "1d", "2025-11-28 13:00:00")) == "2025-11-28"
    assert latest_value(run(data, "1d", "2025-11-28 13:01:00")) == "2025-11-28"


def test_early_close_completes_the_week_at_the_early_close() -> None:
    data = frames("2025-11-28")["1wk"]

    before = run(data, "1wk", "2025-11-28 12:59:00")
    after = run(data, "1wk", "2025-11-28 13:01:00")

    assert latest_value(before) == "2025-11-17 to 2025-11-21"
    assert latest_value(after) == "2025-11-24 to 2025-11-28"


def test_holiday_friday_completes_the_week_on_thursday() -> None:
    # Good Friday 2026-04-03 is an exchange holiday: the trading week ends on Thursday 04-02.
    data = frames("2026-04-02")["1wk"]

    before = run(data, "1wk", "2026-04-02 15:59:00")
    after = run(data, "1wk", "2026-04-02 16:01:00")

    assert latest_value(before) == "2026-03-23 to 2026-03-27"
    assert latest_value(after) == "2026-03-30 to 2026-04-02"


def test_month_ending_on_a_weekend_completes_on_the_last_trading_friday() -> None:
    # May 31 2026 is a Sunday and Memorial Day (05-25) is a holiday: the last session is Fri 05-29.
    data = frames("2026-05-29")["1mo"]

    assert latest_value(run(data, "1mo", "2026-05-29 15:59:00")) == "2026-04"
    assert latest_value(run(data, "1mo", "2026-05-29 16:00:00")) == "2026-05"
    assert latest_value(run(data, "1mo", "2026-05-31 12:00:00")) == "2026-05"


def test_candle_clock_reports_calendar_source_and_weekday_fallback_warning() -> None:
    calendar_clock = CandleClock("1d", exchange_timezone=NY, exchange_calendar="NASDAQ")
    fallback_clock = CandleClock("1wk", exchange_timezone=NY, exchange_calendar=None)

    assert calendar_clock.source == "exchange_calendar"
    assert calendar_clock.warnings == []
    assert fallback_clock.source == "weekday_fallback"
    assert "weekday-based" in fallback_clock.warnings[0]


def test_unknown_exchange_falls_back_to_weekdays_with_a_warning() -> None:
    data = frames("2026-09-18")["1d"]
    unknown = ResolvedInstrument(input_identifier="XYZ", symbol="XYZ", name="XYZ", exchange=None)

    result = run(data, "1d", "2026-09-18 16:00:01", instrument=unknown)

    assert latest_value(result) == "2026-09-18"
    assert any("weekday-based" in warning for warning in result["warnings"])
    assert result["candle_completion_source"] == "weekday_fallback"


def test_intraday_intervals_keep_nominal_bar_end_arithmetic() -> None:
    clock = CandleClock("15m", exchange_timezone=NY)
    starts = pd.Series(pd.to_datetime(["2026-09-18 09:30", "2026-09-18 09:45"]).tz_localize(NY))

    ends = clock.bar_ends(starts)

    assert list(ends) == [starts[0] + pd.Timedelta(minutes=15), starts[1] + pd.Timedelta(minutes=15)]
    assert clock.source == "nominal"


# ---------------------------------------------------------------------------
# 8. Daylight-saving transitions (US and Israel)
# ---------------------------------------------------------------------------


def _bar_end_utc(clock: CandleClock, day: str, tz: str) -> pd.Timestamp:
    start = pd.Series(pd.to_datetime([day]).tz_localize(tz))
    return clock.bar_ends(start).iloc[0].tz_convert("UTC")


def test_us_close_stays_at_16_00_local_across_both_dst_changes() -> None:
    clock = CandleClock("1d", exchange_timezone=NY, exchange_calendar="NASDAQ")

    # Spring forward 2026-03-08: Friday 03-06 is EST (UTC-5), Monday 03-09 is EDT (UTC-4).
    assert _bar_end_utc(clock, "2026-03-06", NY) == pd.Timestamp("2026-03-06 21:00", tz="UTC")
    assert _bar_end_utc(clock, "2026-03-09", NY) == pd.Timestamp("2026-03-09 20:00", tz="UTC")
    # Fall back 2026-11-01: Friday 10-30 is EDT, Monday 11-02 is EST.
    assert _bar_end_utc(clock, "2026-10-30", NY) == pd.Timestamp("2026-10-30 20:00", tz="UTC")
    assert _bar_end_utc(clock, "2026-11-02", NY) == pd.Timestamp("2026-11-02 21:00", tz="UTC")


def test_us_weekly_and_monthly_ends_follow_dst() -> None:
    weekly = CandleClock("1wk", exchange_timezone=NY, exchange_calendar="NASDAQ")
    monthly = CandleClock("1mo", exchange_timezone=NY, exchange_calendar="NASDAQ")

    assert _bar_end_utc(weekly, "2026-03-02", NY) == pd.Timestamp("2026-03-06 21:00", tz="UTC")
    assert _bar_end_utc(weekly, "2026-03-09", NY) == pd.Timestamp("2026-03-13 20:00", tz="UTC")
    assert _bar_end_utc(monthly, "2026-10-01", NY) == pd.Timestamp("2026-10-30 20:00", tz="UTC")
    assert _bar_end_utc(monthly, "2026-11-01", NY) == pd.Timestamp("2026-11-30 21:00", tz="UTC")


def test_israel_close_keeps_the_same_local_time_across_dst() -> None:
    clock = CandleClock("1d", exchange_timezone=JERUSALEM, exchange_calendar="TASE")
    # Israel switched to summer time on Friday 2026-03-27.
    before = _bar_end_utc(clock, "2026-03-26", JERUSALEM)  # UTC+2
    after = _bar_end_utc(clock, "2026-03-30", JERUSALEM)  # UTC+3

    local_before = before.tz_convert(JERUSALEM)
    local_after = after.tz_convert(JERUSALEM)
    assert local_before.utcoffset() == pd.Timedelta(hours=2)
    assert local_after.utcoffset() == pd.Timedelta(hours=3)
    assert local_before.time() == local_after.time()
    assert (before.hour - after.hour) == 1  # one hour earlier in UTC after the change


def test_israel_daily_analysis_switches_cleanly_at_the_dst_change() -> None:
    data = frames("2026-03-30", calendar="XTAE", tz=JERUSALEM, start="2024-01-02")["1d"]
    close_utc = _bar_end_utc(
        CandleClock("1d", exchange_timezone=JERUSALEM, exchange_calendar="TASE"), "2026-03-30", JERUSALEM
    )

    before = analyze_dataframe(
        df=data.copy(),
        symbol=TEVA.symbol,
        interval="1d",
        as_of=close_utc - pd.Timedelta(minutes=1),
        display_timezone=NY,
        instrument=TEVA,
    )
    after = analyze_dataframe(
        df=data.copy(),
        symbol=TEVA.symbol,
        interval="1d",
        as_of=close_utc + pd.Timedelta(minutes=1),
        display_timezone=NY,
        instrument=TEVA,
    )

    # TASE trades Monday-Friday: the short Friday session (03-27) precedes Monday 03-30, which is
    # still forming one minute before its close and complete one minute after it.
    assert latest_value(before) == "2026-03-27"
    assert latest_value(after) == "2026-03-30"
    assert before["exchange_timezone"] == JERUSALEM
    assert before["exchange_calendar"] == "TASE"


def test_tase_short_friday_session_completes_early() -> None:
    clock = CandleClock("1d", exchange_timezone=JERUSALEM, exchange_calendar="TASE")
    friday = _bar_end_utc(clock, "2026-03-27", JERUSALEM).tz_convert(JERUSALEM)
    thursday = _bar_end_utc(clock, "2026-03-26", JERUSALEM).tz_convert(JERUSALEM)

    assert friday.time() < thursday.time()


# ---------------------------------------------------------------------------
# 9. Insufficient history for the trend horizons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,candles,expected_insufficient",
    [
        ("1d", 60, ["long"]),  # short 10 / medium 50 fit, long 100 does not
        ("1wk", 50, ["long"]),  # short 8 / medium 26 fit, long 104 does not
        ("1mo", 30, ["long"]),  # short 6 / medium 24 fit, long 60 does not
        ("1d", 30, ["medium", "long"]),
    ],
)
def test_horizons_without_enough_candles_are_reported_as_insufficient(
    interval: str, candles: int, expected_insufficient: list[str]
) -> None:
    through = "2026-09-30" if interval == "1mo" else "2026-09-18"
    as_of = "2026-09-30 16:00:01" if interval == "1mo" else "2026-09-18 16:00:01"
    data = frames(through)[interval].tail(candles)

    result = run(data, interval, as_of)

    assert result["insufficient_trend_horizons"] == expected_insufficient
    for name in expected_insufficient:
        assert result[f"{name}_term_trend"] == "Insufficient Data"
        assert result[f"{name}_term_trend_score"] is None
    for name in {"short", "medium", "long"} - set(expected_insufficient):
        assert result[f"{name}_term_trend"] != "Insufficient Data"
        assert result[f"{name}_term_trend_score"] is not None
    # The composite trend is still computed from the horizons that are valid.
    assert result["trend"] != "Insufficient Data"
    assert any("Insufficient Data" in warning for warning in result["warnings"])
    assert any(f"{expected_insufficient[0]}-term trend horizon needs" in warning for warning in result["warnings"])


def test_no_valid_horizon_yields_an_insufficient_composite_trend() -> None:
    data = frames("2026-09-18")["1d"].tail(8)  # fewer than the 10 candles of the short horizon

    result = run(data, "1d", "2026-09-18 16:00:01")

    assert result["trend"] == "Insufficient Data"
    assert result["insufficient_trend_horizons"] == ["short", "medium", "long"]
    assert result["trend_score"] == 0.0
    assert result["overall_bias"] in {"Neutral", "Bullish", "Bearish"}


def test_full_history_reports_every_horizon() -> None:
    data = frames("2026-09-18")["1d"].tail(130)

    result = run(data, "1d", "2026-09-18 16:00:01")

    assert result["insufficient_trend_horizons"] == []
    assert result["trend_horizon_candles"] == {"short": 10, "medium": 50, "long": 100}
    assert result["local_trend_lookback_bars"] == 20


def test_each_timeframe_uses_its_own_trend_horizons() -> None:
    expectations = {
        "1d": ({"short": 10, "medium": 50, "long": 100}, 20),
        "1wk": ({"short": 8, "medium": 26, "long": 104}, 13),
        "1mo": ({"short": 6, "medium": 24, "long": 60}, 12),
    }
    for interval, (horizons, local_window) in expectations.items():
        through = "2026-09-30" if interval == "1mo" else "2026-09-18"
        as_of = "2026-09-30 16:00:01" if interval == "1mo" else "2026-09-18 16:00:01"
        result = run(frames(through)[interval], interval, as_of)

        assert result["trend_horizon_candles"] == horizons
        assert result["local_trend_lookback_bars"] == local_window
        assert result["insufficient_trend_horizons"] == []


def test_text_output_marks_unavailable_horizons() -> None:
    data = frames("2026-09-18")["1d"].tail(60)

    text = format_analysis_text(run(data, "1d", "2026-09-18 16:00:01"))

    assert "Long-Term Trend: Insufficient Data (Not Available, 100 candles)" in text
    assert "Short-Term Trend:" in text and ", 10 candles)" in text


# ---------------------------------------------------------------------------
# 10. Patterns, trend, scoring and price only ever see completed candles
# ---------------------------------------------------------------------------

_COMPARED_KEYS = (
    "trend",
    "trend_score",
    "overall_bias",
    "market_state",
    "net_signal_score",
    "pattern_score",
    "volume_score",
    "latest_close",
    "short_term_trend_score",
    "medium_term_trend_score",
    "long_term_trend_score",
    "rule_confidence",
)


@pytest.mark.parametrize(
    "interval,through,forming_as_of",
    [
        ("1d", "2026-09-18", "2026-09-18 11:00:00"),
        ("1wk", "2026-09-16", "2026-09-16 11:00:00"),
        ("1mo", "2026-09-29", "2026-09-29 11:00:00"),
    ],
)
def test_the_forming_candle_has_no_influence_on_any_result(interval: str, through: str, forming_as_of: str) -> None:
    with_forming = frames(through)[interval]
    completed_only = with_forming.iloc[:-1]
    # An extreme forming candle would move price, trend and patterns if it leaked in.
    with_forming.loc[with_forming.index[-1], ["Open", "High", "Low", "Close"]] = [40.0, 900.0, 1.0, 800.0]

    polluted = run(with_forming, interval, forming_as_of)
    clean = run(completed_only, interval, forming_as_of)

    for key in _COMPARED_KEYS:
        assert polluted[key] == clean[key], key
    assert polluted["completed_candles_used"] == clean["completed_candles_used"] == len(completed_only)
    assert polluted["latest_completed_candle"] == clean["latest_completed_candle"]
    assert {event["event_id"] for event in polluted["all_detected_patterns"]} == {
        event["event_id"] for event in clean["all_detected_patterns"]
    }


@pytest.mark.parametrize(
    "interval,through,as_of",
    [
        ("1d", "2026-09-18", "2026-09-18 16:00:01"),
        ("1wk", "2026-09-18", "2026-09-18 16:00:01"),
        ("1mo", "2026-09-30", "2026-09-30 16:00:01"),
    ],
)
def test_every_pattern_timestamp_is_at_or_before_the_analysis_time(interval: str, through: str, as_of: str) -> None:
    result = run(frames(through)[interval], interval, as_of)
    cutoff = pd.Timestamp(as_of, tz=NY)

    events = result["all_detected_patterns"]
    assert events
    for event in events:
        assert pd.Timestamp(event["detected_at"]) <= cutoff
        assert int(event["detected_index"]) <= result["completed_candles_used"] - 1


def test_completed_candle_index_is_the_same_for_price_trend_and_patterns() -> None:
    data = frames("2026-09-18")["1d"]

    result = run(data, "1d", "2026-09-18 10:00:00")

    last_index = result["completed_candles_used"] - 1
    assert result["latest_close"] == round(float(data["Close"].iloc[last_index]), 2)
    for event in result["all_detected_patterns"]:
        assert int(event["last_completed_candle_index"]) == last_index


# ---------------------------------------------------------------------------
# 11. Recency and lifecycle are counted in candles, not calendar days
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,through,as_of",
    [
        ("1d", "2026-09-18", "2026-09-18 16:00:01"),
        ("1wk", "2026-09-18", "2026-09-18 16:00:01"),
        ("1mo", "2026-09-30", "2026-09-30 16:00:01"),
    ],
)
def test_pattern_age_and_lifecycle_are_candle_counts(interval: str, through: str, as_of: str) -> None:
    result = run(frames(through)[interval], interval, as_of)
    events = result["historical_lifecycle_events"] + result["current_relevant_patterns"]
    assert events

    calendar_day_differences = 0
    for event in events:
        distance = int(event["last_completed_candle_index"]) - int(event["completion_index"])
        assert isinstance(event["score_age_bars"], int)
        assert 0 <= event["score_age_bars"] <= distance
        # The same number of candles of expiry regardless of what one candle spans in calendar time.
        assert event["state_expiration_bars"] == 4
        if interval == "1d" and distance >= 3:
            completed = pd.Timestamp(event["pattern_completion"])
            latest = pd.Timestamp(result["latest_completed_candle"]["completed_at"])
            calendar_day_differences += int((latest - completed).days != distance)
    if interval == "1d":
        # Weekends and holidays make calendar distance differ from candle distance; age ignores it.
        assert calendar_day_differences > 0


@pytest.mark.parametrize("interval", ["1d", "1wk", "1mo"])
def test_pattern_recency_weights_do_not_depend_on_the_calendar_span_of_a_candle(interval: str) -> None:
    through = "2026-09-30" if interval == "1mo" else "2026-09-18"
    as_of = "2026-09-30 16:00:01" if interval == "1mo" else "2026-09-18 16:00:01"
    result = run(frames(through)[interval], interval, as_of)

    for event in result["current_relevant_patterns"]:
        assert 0.0 <= event["recency_weight"] <= 1.0
        assert event["score_age_bars"] <= event["score_max_age_bars"]


# ---------------------------------------------------------------------------
# Output fields and dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,through,as_of,timeframe,lookback,label,latest",
    [
        ("1d", "2026-09-18", "2026-09-18 16:00:01", "Daily", "6 months (6mo)", "Latest Completed Trading Session", "2026-09-18"),
        ("1wk", "2026-09-18", "2026-09-18 16:00:01", "Weekly", "5 years (5y)", "Latest Completed Trading Week", "2026-09-14 to 2026-09-18"),
        ("1mo", "2026-09-30", "2026-09-30 16:00:01", "Monthly", "10 years (10y)", "Latest Completed Trading Month", "2026-09"),
    ],
)
def test_text_output_shows_timeframe_interval_lookback_and_latest_completed_candle(
    interval, through, as_of, timeframe, lookback, label, latest
) -> None:
    period = TIMEFRAME_SPECS[spec_for_interval(interval).timeframe].lookback_period
    result = run(frames(through)[interval], interval, as_of, requested_period=period)

    text = format_analysis_text(result)

    assert f"Timeframe: {timeframe}" in text
    assert f"Candle Interval: {interval}" in text
    assert f"Historical Lookback: {lookback}" in text
    assert f"Completed Candles Used: {result['completed_candles_used']}" in text
    assert f"{label}: {latest}" in text
    assert "Exchange Timezone: America/New_York" in text
    assert "Display Timezone: Asia/Jerusalem" in text
    assert "Analysis Time:" in text
    assert "Selected timeframe" not in text
    assert "One day" not in text
    assert "Latest Completed Candle Start" not in text


def test_daily_candle_dates_are_never_shown_with_an_artificial_hour() -> None:
    result = run(frames("2026-09-18")["1d"], "1d", "2026-09-18 16:00:01")
    text = format_analysis_text(result)

    assert "07:00" not in text
    assert "Asia/Jerusalem" in text  # only in the Analysis Time / Display Timezone lines
    assert result["latest_bar_start"] == "2026-09-18"
    assert result["latest_bar_end"] == "2026-09-18"
    for event in result["all_detected_patterns"][:20]:
        assert len(event["detected_at_display"]) == len("2026-09-18")


def test_candle_dates_do_not_move_when_the_display_timezone_changes() -> None:
    data = frames("2026-09-18")["1d"]

    result_il = analyze_dataframe(
        df=data.copy(), symbol="PYPL", interval="1d", as_of=pd.Timestamp("2026-09-18 16:00:01", tz=NY),
        display_timezone="Asia/Jerusalem", instrument=PYPL,
    )
    result_la = analyze_dataframe(
        df=data.copy(), symbol="PYPL", interval="1d", as_of=pd.Timestamp("2026-09-18 16:00:01", tz=NY),
        display_timezone="America/Los_Angeles", instrument=PYPL,
    )

    assert latest_value(result_il) == latest_value(result_la) == "2026-09-18"
    assert result_il["latest_bar_start"] == result_la["latest_bar_start"]
    assert result_il["analysis_time"] != result_la["analysis_time"]


def test_json_result_carries_the_timeframe_fields() -> None:
    result = run(frames("2026-09-18")["1wk"], "1wk", "2026-09-18 16:00:01", requested_period="5y")

    assert result["timeframe"] == "WEEKLY"
    assert result["timeframe_label"] == "Weekly"
    assert result["candle_interval"] == "1wk"
    assert result["historical_lookback"] == "5y"
    assert result["historical_lookback_display"] == "5 years (5y)"
    assert result["completed_candles_used"] == result["data_quality_report"]["completed_row_count"]
    assert result["latest_completed_candle"]["first_trading_date"] == "2026-09-14"
    assert result["latest_completed_candle"]["last_trading_date"] == "2026-09-18"


def test_json_output_serializes_and_keeps_insufficient_scores_null() -> None:
    result = run(frames("2026-09-18")["1d"].tail(60), "1d", "2026-09-18 16:00:01", requested_period="6mo")

    payload = json.loads(format_analysis_json(result))

    assert payload["timeframe"] == "DAILY"
    assert payload["long_term_trend"] == "Insufficient Data"
    assert payload["long_term_trend_score"] is None
    assert payload["latest_completed_candle"]["value"] == "2026-09-18"
    assert payload["insufficient_trend_horizons"] == ["long"]


def test_explicit_timeframe_must_match_the_interval() -> None:
    with pytest.raises(ConfigurationError, match="uses 1wk candles"):
        run(frames("2026-09-18")["1d"], "1d", "2026-09-18 16:00:01", timeframe="WEEKLY")


def test_intraday_analysis_is_unchanged_by_the_timeframe_work() -> None:
    result_keys = run(frames("2026-09-18")["1d"], "1d", "2026-09-18 16:00:01").keys()
    assert "timeframe" in result_keys

    starts = pd.date_range("2026-09-18 09:30", periods=26, freq="15min", tz=NY)
    rng = np.random.default_rng(3)
    close = 50 + np.cumsum(rng.normal(0, 0.1, len(starts)))
    intraday = pd.DataFrame(
        {
            "Datetime": starts,
            "Open": close - 0.02,
            "High": close + 0.1,
            "Low": close - 0.1,
            "Close": close,
            "Volume": rng.integers(1000, 2000, len(starts)),
        }
    )
    result = analyze_dataframe(
        df=intraday,
        symbol="PYPL",
        interval="15m",
        as_of=pd.Timestamp("2026-09-18 16:00:01", tz=NY),
        instrument=PYPL,
    )

    assert result["timeframe"] is None
    assert result["latest_completed_candle"] is None
    text = format_analysis_text(result)
    assert "Interval: 15m" in text
    assert "Latest Completed Candle Start:" in text
    assert "Timeframe:" not in text


# ---------------------------------------------------------------------------
# 12. Symbol resolution, providers and the CLI keep working
# ---------------------------------------------------------------------------


def test_american_and_israeli_symbols_resolve_their_exchange_calendars() -> None:
    us = build_analysis_context(
        symbol="PYPL", interval="1d", display_timezone=JERUSALEM, session_mode="regular", instrument=PYPL
    )
    il = build_analysis_context(
        symbol="TEVA.TA", interval="1d", display_timezone=JERUSALEM, session_mode="regular", instrument=TEVA
    )

    assert (us.exchange_calendar, us.exchange_timezone) == ("NASDAQ", NY)
    assert (il.exchange_calendar, il.exchange_timezone) == ("TASE", JERUSALEM)


def _install_fake_download(monkeypatch: pytest.MonkeyPatch, through: str, calls: list) -> None:
    def fake_download(self, symbol, interval, period, start, end, *, include_extended_hours):
        calls.append((symbol, interval, period))
        frame = frames(through)[interval].rename(columns={"Datetime": "Date"})
        metadata = {
            "source": "yfinance",
            "symbol": symbol,
            "fast_info": None,
            "history_metadata": {
                "exchangeName": "NMS",
                "exchangeTimezoneName": NY,
                "instrumentType": "EQUITY",
                "currency": "USD",
            },
        }
        return frame, metadata

    monkeypatch.setattr(market_data_module.YFinanceProvider, "_download", fake_download)


@pytest.mark.parametrize(
    "timeframe,interval,period",
    [("DAILY", "1d", "6mo"), ("WEEKLY", "1wk", "5y"), ("MONTHLY", "1mo", "10y"), ("1_WEEK", "1wk", "5y")],
)
def test_analyze_stock_selects_interval_and_lookback_from_the_timeframe(
    monkeypatch: pytest.MonkeyPatch, timeframe: str, interval: str, period: str
) -> None:
    calls: list = []
    _install_fake_download(monkeypatch, "2026-09-30", calls)

    result = analyze_stock(
        "PYPL",
        timeframe=timeframe,
        instrument=PYPL,
        as_of=pd.Timestamp("2026-09-30 16:00:01", tz=NY),
        no_cache=True,
        session_mode="regular",
    )

    assert calls == [("PYPL", interval, period)]
    assert result["candle_interval"] == interval
    assert result["historical_lookback"] == period
    assert result["timeframe"] == parse_timeframe(timeframe).value


def test_analyze_stock_without_a_timeframe_keeps_the_legacy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_load(self, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing the request")

    monkeypatch.setattr(market_data_module.YFinanceProvider, "load", fake_load)

    with pytest.raises(RuntimeError):
        analyze_stock("PYPL", instrument=PYPL, no_cache=True)

    assert (captured["period"], captured["interval"]) == ("1mo", "15m")


def test_monthly_lookback_retries_with_max_history_when_too_few_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    full = frames("2026-09-30")["1mo"]

    def fake_download(self, symbol, interval, period, start, end, *, include_extended_hours):
        calls.append(period)
        frame = full.rename(columns={"Datetime": "Date"})
        if period != "max":
            frame = frame.tail(30)  # the provider returned less than the long trend horizon needs
        return frame.reset_index(drop=True), {
            "source": "yfinance",
            "symbol": symbol,
            "fast_info": None,
            "history_metadata": {"exchangeName": "NMS", "exchangeTimezoneName": NY, "currency": "USD"},
        }

    monkeypatch.setattr(market_data_module.YFinanceProvider, "_download", fake_download)

    result = analyze_stock(
        "PYPL",
        timeframe="MONTHLY",
        instrument=PYPL,
        as_of=pd.Timestamp("2026-09-30 16:00:01", tz=NY),
        no_cache=True,
    )

    assert calls == ["10y", "max"]
    assert result["historical_lookback"] == "max"
    assert result["insufficient_trend_horizons"] == []


def test_explicit_period_is_never_replaced_by_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    _install_fake_download(monkeypatch, "2026-09-30", calls)

    analyze_stock(
        "PYPL",
        period="3y",
        timeframe="MONTHLY",
        instrument=PYPL,
        as_of=pd.Timestamp("2026-09-30 16:00:01", tz=NY),
        no_cache=True,
    )

    assert calls == [("PYPL", "1mo", "3y")]


@pytest.mark.parametrize("timeframe,expected_line", [
    ("DAILY", "Latest Completed Trading Session: 2026-09-17"),
    # An offline data file has no exchange, hence no holiday calendar: the weekday-based fallback
    # is used (and reported), so the Labor Day week is shown as Monday-Friday.
    ("WEEKLY", "Latest Completed Trading Week: 2026-09-07 to 2026-09-11"),
    ("MONTHLY", "Latest Completed Trading Month: 2026-08"),
])
def test_cli_end_to_end_with_a_data_file(tmp_path: Path, capsys, timeframe: str, expected_line: str) -> None:
    interval = TIMEFRAME_SPECS[parse_timeframe(timeframe)].interval
    through = {"1d": "2026-09-18", "1wk": "2026-09-16", "1mo": "2026-09-29"}[interval]
    data_file = tmp_path / "PYPL.csv"
    frame = frames(through)[interval].copy()
    # A CSV holds one UTC offset per column, so write UTC (New York offsets change with DST).
    frame["Datetime"] = frame["Datetime"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%z")
    frame.to_csv(data_file, index=False)
    as_of = {"1d": "2026-09-18T10:00:00-04:00", "1wk": "2026-09-16T12:00:00-04:00", "1mo": "2026-09-29T12:00:00-04:00"}[interval]

    exit_code = main(
        [
            "analyze", "PYPL", "--timeframe", timeframe, "--data-file", str(data_file),
            "--exchange-timezone", NY, "--as-of", as_of, "--no-cache",
        ],
        interactive=False,
        validator=lambda instrument: None,  # no network lookup in tests
    )
    output = capsys.readouterr().out

    assert exit_code == ExitCode.SUCCESS
    assert f"Timeframe: {timeframe.capitalize()}" in output
    assert f"Candle Interval: {interval}" in output
    assert expected_line in output
    assert "weekday-based" in output  # the fallback is disclosed in the warnings
    assert "Selected timeframe: One day" not in output
    assert BAR_END_COLUMN not in output
