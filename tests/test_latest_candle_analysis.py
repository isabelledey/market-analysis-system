"""Tests covering the latest-candle-analysis diagnosis:

1. Local_Trend for daily/weekly intervals uses a real rolling multi-candle window
   (see test_trend_separation.py for the dedicated Local_Trend tests).
2. Latest Candle Direction is a single-candle classifier, kept separate from Local_Trend.
3. The reported PYPL candle does not match Shooting Star / Upper-Wick Rejection geometry,
   and every detected pattern's serialized OHLC matches its own source row exactly.
4. Incomplete (in-progress) candles are excluded from analysis and from pattern detection.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.config import PatternConfig
from stock_pattern_model.features import add_features
from stock_pattern_model.pattern_detector import ShootingStarDetector
from stock_pattern_model.pattern_detector import _long_upper_rejection_geometry
from stock_pattern_model.pattern_detector import classify_latest_candle_direction

EXCHANGE_TZ = ZoneInfo("America/New_York")


def _bar(ts, open_, high, low, close, volume) -> dict:
    return {
        "Datetime": ts,
        "Open": round(open_, 4),
        "High": round(high, 4),
        "Low": round(low, 4),
        "Close": round(close, 4),
        "Volume": volume,
    }


def make_daily_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def make_daily_uptrend_with_final_bar(final_bar: dict, length: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range(start="2026-05-01", periods=length, freq="1D", tz=EXCHANGE_TZ)
    rows = []
    price = 50.0
    for index, timestamp in enumerate(timestamps[:-1]):
        step = price + index * 0.4
        rows.append(_bar(timestamp, step, step + 0.3, step - 0.3, step + 0.1, 5_000_000))
    rows.append({**final_bar, "Datetime": timestamps[-1]})
    return make_daily_df(rows)


# ---------------------------------------------------------------------------
# Latest Candle Direction (single-candle classifier, kept separate from trend)
# ---------------------------------------------------------------------------


def test_latest_candle_direction_strong_bullish_for_marubozu_like_candle() -> None:
    row = pd.Series(
        {
            "Open": 60.00,
            "High": 63.00,
            "Low": 59.90,
            "Close": 62.90,
            "Candle_Range": 3.10,
            "Body_Ratio": (62.90 - 60.00) / 3.10,
            "Upper_Wick_Ratio": (63.00 - 62.90) / 3.10,
            "Lower_Wick_Ratio": (60.00 - 59.90) / 3.10,
        }
    )

    label, score = classify_latest_candle_direction(row)

    assert label == "Strong Bullish"
    assert score > 0.6


def test_latest_candle_direction_strong_bearish_for_inverse_marubozu_like_candle() -> None:
    row = pd.Series(
        {
            "Open": 62.90,
            "High": 63.00,
            "Low": 59.90,
            "Close": 60.00,
            "Candle_Range": 3.10,
            "Body_Ratio": (62.90 - 60.00) / 3.10,
            "Upper_Wick_Ratio": (63.00 - 62.90) / 3.10,
            "Lower_Wick_Ratio": (60.00 - 59.90) / 3.10,
        }
    )

    label, score = classify_latest_candle_direction(row)

    assert label == "Strong Bearish"
    assert score < -0.6


def test_latest_candle_direction_neutral_for_doji_like_candle() -> None:
    row = pd.Series(
        {
            "Open": 60.00,
            "High": 60.60,
            "Low": 59.40,
            "Close": 60.02,
            "Candle_Range": 1.20,
            "Body_Ratio": 0.02 / 1.20,
            "Upper_Wick_Ratio": 0.58 / 1.20,
            "Lower_Wick_Ratio": 0.60 / 1.20,
        }
    )

    label, score = classify_latest_candle_direction(row)

    assert label == "Neutral"
    assert abs(score) < 0.2


def test_latest_candle_direction_zero_range_candle_is_neutral() -> None:
    row = pd.Series(
        {
            "Open": 60.0,
            "High": 60.0,
            "Low": 60.0,
            "Close": 60.0,
            "Candle_Range": 0.0,
            "Body_Ratio": 0.0,
            "Upper_Wick_Ratio": 0.0,
            "Lower_Wick_Ratio": 0.0,
        }
    )

    label, score = classify_latest_candle_direction(row)

    assert label == "Neutral"
    assert score == 0.0


def test_strong_bullish_daily_breakout_reports_strong_bullish_latest_candle() -> None:
    breakout_bar = _bar(pd.Timestamp("2026-06-14", tz=EXCHANGE_TZ), 61.50, 64.50, 61.40, 64.30, 40_000_000)
    df = make_daily_uptrend_with_final_bar(breakout_bar)

    result = analyze_dataframe(
        df=df,
        symbol="BULLTEST",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )

    assert result["latest_candle_direction"] == "Strong Bullish"
    # This must be a separate field from local trend -- both should agree here, but they are
    # computed by entirely different functions.
    assert result["local_trend"] == "Uptrend"


def test_latest_candle_direction_field_present_in_text_output() -> None:
    from stock_pattern_model.formatters import format_analysis_text

    breakout_bar = _bar(pd.Timestamp("2026-06-14", tz=EXCHANGE_TZ), 61.50, 64.50, 61.40, 64.30, 40_000_000)
    df = make_daily_uptrend_with_final_bar(breakout_bar)

    result = analyze_dataframe(
        df=df,
        symbol="BULLTEST",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )
    text = format_analysis_text(result)

    assert "Latest Candle Direction: Strong Bullish" in text


# ---------------------------------------------------------------------------
# PYPL Shooting Star discrepancy
# ---------------------------------------------------------------------------

PYPL_CANDLE = {"Open": 60.56, "High": 61.83, "Low": 60.55, "Close": 61.25}


def test_pypl_candle_fails_shooting_star_geometry_gate() -> None:
    row = pd.Series(
        {
            **PYPL_CANDLE,
            "Candle_Range": PYPL_CANDLE["High"] - PYPL_CANDLE["Low"],
            "Body_Ratio": abs(PYPL_CANDLE["Close"] - PYPL_CANDLE["Open"])
            / (PYPL_CANDLE["High"] - PYPL_CANDLE["Low"]),
            "Upper_Wick_Ratio": (PYPL_CANDLE["High"] - max(PYPL_CANDLE["Open"], PYPL_CANDLE["Close"]))
            / (PYPL_CANDLE["High"] - PYPL_CANDLE["Low"]),
            "Lower_Wick_Ratio": (min(PYPL_CANDLE["Open"], PYPL_CANDLE["Close"]) - PYPL_CANDLE["Low"])
            / (PYPL_CANDLE["High"] - PYPL_CANDLE["Low"]),
            "Is_Significant_Candle": True,
        }
    )

    assert _long_upper_rejection_geometry(row) is False


def test_pypl_candle_is_not_classified_as_shooting_star_end_to_end() -> None:
    df = make_daily_uptrend_with_final_bar({**PYPL_CANDLE, "Volume": 15_627_600})

    result = analyze_dataframe(
        df=df,
        symbol="PYPL",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )

    pattern_names = {pattern["pattern_name"] for pattern in result["all_detected_patterns"]}
    assert "Shooting Star" not in pattern_names
    assert "Upper-Wick Rejection" not in pattern_names


def test_pypl_candle_never_triggers_shooting_star_detector_directly() -> None:
    df = make_daily_uptrend_with_final_bar({**PYPL_CANDLE, "Volume": 15_627_600})
    feature_df = add_features(df)

    events = ShootingStarDetector().detect(feature_df, PatternConfig(), "1d")

    assert events == []


# ---------------------------------------------------------------------------
# Follow-up investigation: the real PYPL candle behind the label is the day
# *before* the one quoted above (2026-08-18, not 2026-08-19). Confirmed against
# live market data: index 61 (2026-08-18) is Open 60.205002 / High 61.599998 /
# Low 59.959999 / Close 60.43, immediately followed by index 62 (2026-08-19,
# the latest completed bar) which is exactly PYPL_CANDLE above. The originally
# reported OHLC belonged to the *next* candle, not the one the pattern actually
# fired on.
# ---------------------------------------------------------------------------

PYPL_ACTUAL_SHOOTING_STAR_CANDLE = {
    "Open": 60.205002,
    "High": 61.599998,
    "Low": 59.959999,
    "Close": 60.43,
}


def test_pypl_actual_shooting_star_candle_passes_geometry_gate() -> None:
    o, h, l, c = (
        PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Open"],
        PYPL_ACTUAL_SHOOTING_STAR_CANDLE["High"],
        PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Low"],
        PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Close"],
    )
    row = pd.Series(
        {
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Candle_Range": h - l,
            "Body_Ratio": abs(c - o) / (h - l),
            "Upper_Wick_Ratio": (h - max(o, c)) / (h - l),
            "Lower_Wick_Ratio": (min(o, c) - l) / (h - l),
            "Is_Significant_Candle": True,
        }
    )

    assert _long_upper_rejection_geometry(row) is True


def test_pypl_shooting_star_pattern_is_linked_to_aug18_not_aug19() -> None:
    # Builds a realistic sequence ending in the two real, back-to-back PYPL daily candles:
    # 2026-08-18 (the real shooting-star geometry) followed by 2026-08-19 (PYPL_CANDLE, the
    # candle that was previously and incorrectly quoted as "the" shooting-star candle).
    timestamps = pd.date_range(start="2026-05-01", periods=32, freq="1D", tz=EXCHANGE_TZ)
    rows = []
    price = 55.0
    for index, timestamp in enumerate(timestamps[:-2]):
        step = price + index * 0.2
        rows.append(_bar(timestamp, step, step + 0.3, step - 0.3, step + 0.1, 5_000_000))
    rows.append({**PYPL_ACTUAL_SHOOTING_STAR_CANDLE, "Datetime": timestamps[-2], "Volume": 12_000_000})
    rows.append({**PYPL_CANDLE, "Datetime": timestamps[-1], "Volume": 15_627_600})
    df = make_daily_df(rows)

    result = analyze_dataframe(
        df=df,
        symbol="PYPL",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )

    shooting_star_patterns = [
        p for p in result["all_detected_patterns"]
        if p.get("pattern_name") in {"Shooting Star", "Upper-Wick Rejection"}
    ]
    assert shooting_star_patterns, "expected the real Aug-18 geometry to produce a Shooting Star label"

    expected_candle_date = timestamps[-2].strftime("%Y-%m-%d")
    next_day_date = timestamps[-1].strftime("%Y-%m-%d")
    for pattern in shooting_star_patterns:
        candle = pattern["pattern_candle"]
        assert candle["open"] == pytest.approx(PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Open"])
        assert candle["high"] == pytest.approx(PYPL_ACTUAL_SHOOTING_STAR_CANDLE["High"])
        assert candle["low"] == pytest.approx(PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Low"])
        assert candle["close"] == pytest.approx(PYPL_ACTUAL_SHOOTING_STAR_CANDLE["Close"])
        # It must NOT be attributed to the following day's candle (the one previously
        # misreported as the shooting-star candle).
        assert candle["close"] != pytest.approx(PYPL_CANDLE["Close"])
        assert pattern["pattern_start_at"].startswith(expected_candle_date)
        assert not pattern["pattern_start_at"].startswith(next_day_date)


def test_pypl_shooting_star_pattern_not_excluded_as_outside_current_trading_session() -> None:
    # Regression for the false "outside current trading session" exclusion on daily data:
    # the shooting-star candle is one bar before the latest completed bar, which used to
    # always fail the same-calendar-day check for daily intervals.
    timestamps = pd.date_range(start="2026-05-01", periods=32, freq="1D", tz=EXCHANGE_TZ)
    rows = []
    price = 55.0
    for index, timestamp in enumerate(timestamps[:-2]):
        step = price + index * 0.2
        rows.append(_bar(timestamp, step, step + 0.3, step - 0.3, step + 0.1, 5_000_000))
    rows.append({**PYPL_ACTUAL_SHOOTING_STAR_CANDLE, "Datetime": timestamps[-2], "Volume": 12_000_000})
    rows.append({**PYPL_CANDLE, "Datetime": timestamps[-1], "Volume": 15_627_600})
    df = make_daily_df(rows)

    result = analyze_dataframe(
        df=df,
        symbol="PYPL",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )

    shooting_star_patterns = [
        p for p in result["all_detected_patterns"]
        if p.get("pattern_name") in {"Shooting Star", "Upper-Wick Rejection"}
    ]
    assert shooting_star_patterns
    for pattern in shooting_star_patterns:
        assert pattern.get("exclusion_reason") != "outside current trading session"


# ---------------------------------------------------------------------------
# Pattern-to-candle OHLC integrity (general invariant, not specific to one candle)
# ---------------------------------------------------------------------------


def make_real_shooting_star_df() -> pd.DataFrame:
    timestamps = pd.date_range(start="2026-06-01", periods=25, freq="1D", tz=EXCHANGE_TZ)
    rows = []
    price = 50.0
    for index, timestamp in enumerate(timestamps[:-1]):
        step = price + index * 0.6
        rows.append(_bar(timestamp, step, step + 0.3, step - 0.3, step + 0.1, 5_000_000))
    last_open = rows[-1]["Close"]
    rows.append(
        _bar(timestamps[-1], last_open, last_open + 2.0, last_open - 0.05, last_open + 0.10, 9_000_000)
    )
    return make_daily_df(rows)


def test_every_detected_pattern_is_linked_to_its_own_source_row_ohlc() -> None:
    df = make_real_shooting_star_df()

    result = analyze_dataframe(
        df=df,
        symbol="INTEGRITY",
        interval="1d",
        as_of=pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(days=1),
    )

    assert result["all_detected_patterns"], "expected at least one detected pattern in this fixture"

    df_by_datetime = df.set_index(pd.to_datetime(df["Datetime"]))
    for pattern in result["all_detected_patterns"]:
        relevant_prices = pattern["relevant_prices"]
        bar_start = pd.Timestamp(pattern["bar_start_at"])
        source_row = df_by_datetime.loc[df_by_datetime.index == bar_start]
        assert len(source_row) == 1, f"no unique source row found for pattern bar_start {bar_start}"
        source_row = source_row.iloc[0]
        assert relevant_prices["open"] == pytest.approx(float(source_row["Open"]))
        assert relevant_prices["high"] == pytest.approx(float(source_row["High"]))
        assert relevant_prices["low"] == pytest.approx(float(source_row["Low"]))
        assert relevant_prices["close"] == pytest.approx(float(source_row["Close"]))


# ---------------------------------------------------------------------------
# Incomplete candles remain excluded
# ---------------------------------------------------------------------------


def test_incomplete_latest_daily_candle_is_excluded_from_analysis() -> None:
    df = make_daily_uptrend_with_final_bar({**PYPL_CANDLE, "Volume": 15_627_600}, length=30)

    # as_of falls inside the final bar's own day, before that daily candle has closed --
    # it must not be treated as completed.
    as_of_mid_session = pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(hours=6)

    result = analyze_dataframe(df=df, symbol="INCOMPLETE", interval="1d", as_of=as_of_mid_session)

    latest_bar_start = pd.Timestamp(result["latest_datetime"])
    assert latest_bar_start < pd.Timestamp(df.iloc[-1]["Datetime"])
    assert result["data_quality_report"]["completed_row_count"] == len(df) - 1
