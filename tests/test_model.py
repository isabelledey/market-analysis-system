from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model import analyze_dataframe
from model import NoCompletedCandlesError


EXCHANGE_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ = ZoneInfo("Asia/Jerusalem")


def make_base_df(length: int = 30, start: str = "2026-07-10 09:30") -> pd.DataFrame:
    """Create a stable intraday OHLCV DataFrame with timezone-aware timestamps."""
    datetimes = pd.date_range(start=start, periods=length, freq="15min", tz=EXCHANGE_TZ)
    rows = []

    for timestamp in datetimes:
        rows.append(
            {
                "Datetime": timestamp,
                "Open": 100.00,
                "High": 100.60,
                "Low": 99.60,
                "Close": 100.10,
                "Volume": 1000,
            }
        )

    return pd.DataFrame(rows)


def test_incomplete_candle_is_removed_before_analysis() -> None:
    df = make_base_df(length=4)
    as_of = pd.Timestamp("2026-07-10 10:07", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)

    assert result["latest_datetime"] == "2026-07-10T09:45-04:00"
    assert result["latest_bar_start"] == "2026-07-10 16:45 Asia/Jerusalem"
    assert result["latest_bar_end"] == "2026-07-10 17:00 Asia/Jerusalem"


def test_raises_when_no_completed_candles_remain() -> None:
    df = make_base_df(length=1)
    as_of = pd.Timestamp("2026-07-10 09:35", tz=EXCHANGE_TZ)

    try:
        analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    except NoCompletedCandlesError as error:
        assert "No completed 15m candles" in str(error)
    else:
        raise AssertionError("Expected NoCompletedCandlesError for incomplete-only input.")


def test_engulfing_detected_at_equals_final_candle_close() -> None:
    df = make_base_df(length=25)
    df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 101.20, 99.80, 100.00, 3000]
    df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [99.90, 101.50, 99.70, 101.30, 3200]
    as_of = pd.Timestamp("2026-07-10 15:46", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    engulfing = next(
        pattern
        for pattern in result["all_detected_patterns"]
        if pattern["pattern_name"] == "Bullish Engulfing"
    )

    assert engulfing["pattern_start_at"] == "2026-07-10T15:15-04:00"
    assert engulfing["pattern_end_at"] == "2026-07-10T15:45-04:00"
    assert engulfing["bar_start_at"] == "2026-07-10T15:30-04:00"
    assert engulfing["bar_end_at"] == "2026-07-10T15:45-04:00"
    assert engulfing["detected_at"] == "2026-07-10T15:45-04:00"


def test_timezone_conversion_uses_asia_jerusalem_for_display() -> None:
    df = make_base_df(length=25)
    df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 101.20, 99.80, 100.00, 3000]
    df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [99.90, 101.50, 99.70, 101.30, 3200]
    as_of = pd.Timestamp("2026-07-10 15:46", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    engulfing = next(
        pattern
        for pattern in result["all_detected_patterns"]
        if pattern["pattern_name"] == "Bullish Engulfing"
    )

    assert result["display_timezone"] == "Asia/Jerusalem"
    assert engulfing["bar_start_display"] == "2026-07-10 22:30 Asia/Jerusalem"
    assert engulfing["bar_end_display"] == "2026-07-10 22:45 Asia/Jerusalem"
    assert engulfing["detected_at_display"] == "2026-07-10 22:45 Asia/Jerusalem"


def test_breakout_crossing_emits_one_event_not_repeated_events() -> None:
    df = make_base_df(length=30)
    df.loc[20, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 101.20, 100.00, 100.90, 1200]
    df.loc[21, ["Open", "High", "Low", "Close", "Volume"]] = [100.80, 101.40, 100.70, 101.10, 1200]
    df.loc[22, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 101.50, 100.90, 101.20, 1200]
    as_of = pd.Timestamp("2026-07-10 17:01", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    breakout_patterns = [
        pattern
        for pattern in result["all_detected_patterns"]
        if "Breakout" in pattern["pattern_name"]
    ]

    assert len(breakout_patterns) == 1
    assert breakout_patterns[0]["pattern_name"] == "20-Bar Breakout"


def test_strong_breakout_is_not_double_counted() -> None:
    df = make_base_df(length=30)
    df.loc[20, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 101.20, 100.00, 100.90, 2500]
    df.loc[21, ["Open", "High", "Low", "Close", "Volume"]] = [100.80, 101.40, 100.70, 101.10, 1200]
    as_of = pd.Timestamp("2026-07-10 16:46", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    pattern_names = [pattern["pattern_name"] for pattern in result["all_detected_patterns"]]

    assert pattern_names.count("Strong 20-Bar Breakout") == 1
    assert "20-Bar Breakout" not in pattern_names


def test_inside_bar_failure_requires_mother_bar_confirmation() -> None:
    df = make_base_df(length=25)
    df.loc[22, ["Open", "High", "Low", "Close", "Volume"]] = [100.00, 105.00, 95.00, 102.00, 2500]
    df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 104.00, 96.00, 101.50, 2500]
    df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [103.50, 104.50, 97.00, 97.50, 2600]
    as_of = pd.Timestamp("2026-07-10 15:46", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    pattern_names = [pattern["pattern_name"] for pattern in result["all_detected_patterns"]]

    assert "Inside Bar Failure Bearish Reversal" not in pattern_names


def test_inside_bar_failure_detects_confirmed_mother_bar_failure() -> None:
    df = make_base_df(length=25)
    df.loc[22, ["Open", "High", "Low", "Close", "Volume"]] = [100.00, 105.00, 95.00, 102.00, 2500]
    df.loc[23, ["Open", "High", "Low", "Close", "Volume"]] = [101.00, 104.00, 96.00, 101.50, 2500]
    df.loc[24, ["Open", "High", "Low", "Close", "Volume"]] = [104.80, 105.60, 97.00, 100.00, 2600]
    as_of = pd.Timestamp("2026-07-10 15:46", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)
    failure = next(
        pattern
        for pattern in result["all_detected_patterns"]
        if pattern["pattern_name"] == "Inside Bar Failure Bearish Reversal"
    )

    assert failure["pattern_start_at"] == "2026-07-10T15:00-04:00"
    assert failure["pattern_end_at"] == "2026-07-10T15:45-04:00"
    assert failure["detected_at"] == "2026-07-10T15:45-04:00"


def test_no_pattern_scoring_keeps_pattern_score_zero_and_confidence_low() -> None:
    df = make_base_df(length=30)
    as_of = pd.Timestamp("2026-07-10 17:01", tz=EXCHANGE_TZ)

    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)

    assert result["all_detected_patterns"] == []
    assert result["pattern_score"] == 0
    assert result["rule_confidence"] <= 12.0
    assert result["overall_bias"] == "Neutral"


def test_all_patterns_are_returned_while_top_patterns_remain_limited() -> None:
    df = make_base_df(length=40)
    for index, volume in zip((19, 22, 25, 37), (2600, 2700, 2800, 2900)):
        df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [
            100.10,
            100.25,
            99.00,
            100.20,
            volume,
        ]

    as_of = pd.Timestamp("2026-07-10 19:31", tz=EXCHANGE_TZ)
    result = analyze_dataframe(df=df, symbol="TEST", as_of=as_of)

    assert len(result["all_detected_patterns"]) == 4
    assert len(result["top_patterns"]) == 3
    assert result["top_patterns"][0]["pattern_name"] == "Bullish Pin Bar"
