from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.datetime_utils import convert_to_timezone
from stock_pattern_model.datetime_utils import format_display_datetime
from stock_pattern_model.features import add_features
from stock_pattern_model.formatters import format_analysis_text
from stock_pattern_model.pattern_detector import classify_intraday_trend


EXCHANGE_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ = ZoneInfo("Asia/Jerusalem")


def make_ohlcv_df(
    closes: list[float],
    *,
    start: str = "2026-07-10 09:30",
    volume: int = 1500,
) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range(start=start, periods=len(closes), freq="15min", tz=EXCHANGE_TZ)
    previous_close = closes[0]
    for index, (timestamp, close) in enumerate(zip(timestamps, closes)):
        open_ = previous_close if index > 0 else close - 0.05
        high = max(open_, close) + 0.25
        low = min(open_, close) - 0.25
        rows.append(
            {
                "Datetime": timestamp,
                "Open": round(open_, 4),
                "High": round(high, 4),
                "Low": round(low, 4),
                "Close": round(close, 4),
                "Volume": volume + (index * 5),
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=16)


def test_timezone_converts_new_york_summer_to_jerusalem() -> None:
    timestamp = pd.Timestamp("2026-07-20 16:00", tz=EXCHANGE_TZ)

    converted = convert_to_timezone(timestamp, DISPLAY_TZ)

    assert converted == pd.Timestamp("2026-07-20 23:00", tz=DISPLAY_TZ)
    assert format_display_datetime(timestamp, DISPLAY_TZ) == "2026-07-20 23:00:00+0300 Asia/Jerusalem"


def test_timezone_handles_date_rollover_and_analysis_time_conversion() -> None:
    timestamp = pd.Timestamp("2026-07-21 09:38", tz="UTC")

    converted = convert_to_timezone(timestamp, DISPLAY_TZ)

    assert converted == pd.Timestamp("2026-07-21 12:38", tz=DISPLAY_TZ)
    assert format_display_datetime(timestamp, DISPLAY_TZ) == "2026-07-21 12:38:00+0300 Asia/Jerusalem"

    rollover = pd.Timestamp("2026-07-20 19:30", tz=EXCHANGE_TZ)
    assert format_display_datetime(rollover, DISPLAY_TZ) == "2026-07-21 02:30:00+0300 Asia/Jerusalem"


def test_timezone_respects_dst_transition_gap_between_markets() -> None:
    before_us_shift = pd.Timestamp("2026-03-20 16:00", tz=EXCHANGE_TZ)
    after_us_shift = pd.Timestamp("2026-03-30 16:00", tz=EXCHANGE_TZ)

    assert format_display_datetime(before_us_shift, DISPLAY_TZ) == "2026-03-20 22:00:00+0200 Asia/Jerusalem"
    assert format_display_datetime(after_us_shift, DISPLAY_TZ) == "2026-03-30 23:00:00+0300 Asia/Jerusalem"


def test_naive_datetime_raises_explicit_error() -> None:
    try:
        convert_to_timezone(pd.Timestamp("2026-07-20 16:00"), DISPLAY_TZ)
    except ValueError as error:
        assert "timezone-aware" in str(error)
    else:
        raise AssertionError("Expected naive datetimes to raise an explicit error.")


def test_report_rendering_uses_display_timezone_for_normal_timestamps() -> None:
    closes = [100 + (index * 0.08) for index in range(30)]
    df = make_ohlcv_df(closes)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [102.5, 103.4, 102.3, 102.35, 4200]
    result = analyze_dataframe(df=df, symbol="TZREP", as_of=analysis_as_of(df))

    report = format_analysis_text(result, include_all_patterns=True)

    assert "Analysis Time:" in report
    assert "Latest Completed Candle Start: " in report
    assert "Detected at: " in report
    assert "Display Detected at:" not in report
    assert "EDT" not in report
    assert "EST" not in report
    assert "UTC" not in report
    assert "Latest Completed Candle Start:" in report
    assert "+0300 Asia/Jerusalem" in report
    assert "Detected at:" in report


def test_clear_uptrend_is_classified_as_uptrend() -> None:
    closes = [100 + (index * 0.35) for index in range(80)]
    df = make_ohlcv_df(closes)

    result = analyze_dataframe(df=df, symbol="UP", as_of=analysis_as_of(df))

    assert result["trend"] == "Uptrend"
    assert result["trend_score"] > 18
    assert result["short_term_trend"] == "Uptrend"
    assert any("positive" in item.lower() or "bullish" in item.lower() for item in result["trend_evidence"])


def test_clear_downtrend_is_classified_as_downtrend() -> None:
    closes = [140 - (index * 0.42) for index in range(80)]
    df = make_ohlcv_df(closes)

    result = analyze_dataframe(df=df, symbol="DOWN", as_of=analysis_as_of(df))

    assert result["trend"] == "Downtrend"
    assert result["trend_score"] < -18
    assert result["short_term_trend"] == "Downtrend"
    assert any("negative" in item.lower() or "bearish" in item.lower() for item in result["trend_evidence"])


def test_sideways_market_stays_neutral() -> None:
    closes = [100 + ((index % 4) - 1.5) * 0.06 for index in range(80)]
    df = make_ohlcv_df(closes)

    result = analyze_dataframe(df=df, symbol="SIDE", as_of=analysis_as_of(df))

    assert result["trend"] == "Neutral"
    assert abs(result["trend_score"]) < 18


def test_bearish_trend_with_one_bullish_pin_bar_stays_bearish() -> None:
    closes = [120 - (index * 0.30) for index in range(60)]
    df = make_ohlcv_df(closes)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [102.4, 102.55, 100.8, 102.5, 4200]

    result = analyze_dataframe(df=df, symbol="PIN", as_of=analysis_as_of(df))

    assert result["trend"] == "Downtrend"
    assert result["trend_score"] < -18
    assert any(pattern["pattern_name"] == "Bullish Pin Bar" for pattern in result["all_detected_patterns"])


def test_recent_bearish_reversal_outweighs_old_bullish_history() -> None:
    older_bullish = [90 + (index * 0.08) for index in range(160)]
    recent_bearish = [older_bullish[-1] - ((index + 1) * 0.55) for index in range(60)]
    df = make_ohlcv_df(older_bullish + recent_bearish)

    result = analyze_dataframe(df=df, symbol="REV", as_of=analysis_as_of(df))

    assert result["trend"] == "Downtrend"
    assert result["medium_term_trend"] == "Downtrend"
    assert result["trend_score"] < -18


def test_incomplete_candle_is_excluded_from_trend_reversal() -> None:
    closes = [130 - (index * 0.35) for index in range(40)]
    df = make_ohlcv_df(closes)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [116.0, 120.0, 115.8, 119.8, 5000]
    as_of = pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=5)

    result = analyze_dataframe(df=df, symbol="COMPLETE", as_of=as_of)

    assert result["trend"] == "Downtrend"
    assert result["latest_datetime"] == pd.Timestamp(df.iloc[-2]["Datetime"]).isoformat(timespec="minutes")


def test_breakdown_support_appears_in_trend_evidence() -> None:
    closes = [110 - (index * 0.02) for index in range(35)]
    df = make_ohlcv_df(closes)
    df.loc[20:24, "Close"] = [109.7, 109.68, 109.66, 109.64, 109.62]
    df.loc[25, ["Open", "High", "Low", "Close", "Volume"]] = [109.65, 109.7, 107.8, 107.9, 5000]
    df.loc[26, ["Open", "High", "Low", "Close", "Volume"]] = [107.95, 108.05, 107.4, 107.5, 4200]
    df.loc[27, ["Open", "High", "Low", "Close", "Volume"]] = [107.55, 107.6, 107.0, 107.1, 3500]
    df.loc[28, ["Open", "High", "Low", "Close", "Volume"]] = [107.15, 107.2, 106.7, 106.8, 3200]
    continuation_closes = [106.7, 106.6, 106.5, 106.4, 106.3, 106.2]
    for index, close in enumerate(continuation_closes, start=29):
        open_ = close + 0.18
        df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [
            open_,
            open_ + 0.12,
            close - 0.18,
            close,
            3000 - ((index - 29) * 80),
        ]

    result = analyze_dataframe(df=df, symbol="BD", as_of=analysis_as_of(df))

    assert result["trend"] == "Downtrend"
    assert any("downside break" in item.lower() for item in result["trend_evidence"])


def test_scale_independence_keeps_direction_consistent() -> None:
    low_price_df = make_ohlcv_df([10 * (0.995**index) for index in range(70)])
    high_price_df = make_ohlcv_df([1000 * (0.995**index) for index in range(70)])

    low_result = analyze_dataframe(low_price_df, symbol="LOW", as_of=analysis_as_of(low_price_df))
    high_result = analyze_dataframe(high_price_df, symbol="HIGH", as_of=analysis_as_of(high_price_df))

    assert low_result["trend"] == "Downtrend"
    assert high_result["trend"] == "Downtrend"


def test_overlapping_pin_bar_and_doji_are_grouped_as_one_evidence_event() -> None:
    closes = [100 + (index * 0.03) for index in range(30)]
    df = make_ohlcv_df(closes)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [100.88, 101.02, 99.9, 100.9, 4000]

    result = analyze_dataframe(df=df, symbol="OVERLAP", as_of=analysis_as_of(df), top_pattern_count=10)

    latest_bar_patterns = [
        pattern
        for pattern in result["all_detected_patterns"]
        if pattern["candles_ago"] == 0
    ]

    assert {pattern["pattern_name"] for pattern in latest_bar_patterns} >= {"Bullish Pin Bar", "Doji"}
    assert sum(1 for pattern in latest_bar_patterns if pattern["group_primary"]) == 1
    assert sum(1 for pattern in latest_bar_patterns if pattern["group_suppressed"]) >= 1


def test_trend_classifier_is_independent_from_pattern_scoring() -> None:
    closes = [115 - (index * 0.28) for index in range(60)]
    df = make_ohlcv_df(closes)
    base_result = analyze_dataframe(df=df, symbol="BASE", as_of=analysis_as_of(df))

    feature_df = add_features(df)
    classified = classify_intraday_trend(feature_df, lookback_bars=12, pivot_left_bars=2, pivot_right_bars=2, breakout_lookback=20)

    assert base_result["trend"] == str(classified.iloc[-1]["Trend"])
    assert round(base_result["trend_score"], 2) == round(float(classified.iloc[-1]["Trend_Score"]), 2)
