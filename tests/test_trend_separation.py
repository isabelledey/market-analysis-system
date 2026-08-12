"""Tests for the broad / local-session / pattern-entry trend separation.

These cover Stage 1 of the pattern-analysis correction plan: the broad multi-horizon
trend (`classify_intraday_trend`) must never automatically determine the local
session trend (`classify_local_session_trend`), and the trend evaluated immediately
before a detected candlestick pattern (`pattern_entry_trend`, computed inside
`classify_prior_pattern_context`) must reflect the candles actually leading into that
pattern rather than the broad trend or the detector's "expected" direction.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.config import PatternConfig
from stock_pattern_model.exceptions import ConfigurationError
from stock_pattern_model.features import add_features
from stock_pattern_model.formatters import format_analysis_text
from stock_pattern_model.pattern_detector import classify_local_session_trend

EXCHANGE_TZ = ZoneInfo("America/New_York")


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bar(ts, open_, high, low, close, volume) -> dict:
    return {
        "Datetime": ts,
        "Open": round(open_, 4),
        "High": round(high, 4),
        "Low": round(low, 4),
        "Close": round(close, 4),
        "Volume": volume,
    }


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=16)


def make_broad_uptrend_local_decline_df() -> pd.DataFrame:
    """A long, brisk uptrend followed by a sharp 4-bar decline in the final session.

    Timestamps start at midnight so every 96 fifteen-minute bars is exactly one
    calendar day: 288 prior bars span three full "sessions" of a strong uptrend,
    and the final 4 bars are a fresh session that opens strong and immediately
    sells off to new lows with no real recovery.
    """
    prior_bars = 288
    decline_bars = 4
    prior_slope = 1.2
    decline_step = 2.2

    closes: list[float] = [100.0 + (i * prior_slope) for i in range(prior_bars)]
    last = closes[-1]
    for _ in range(decline_bars):
        last -= decline_step
        closes.append(last)

    timestamps = pd.date_range(
        start="2026-07-01 00:00",
        periods=prior_bars + decline_bars,
        freq="15min",
        tz=EXCHANGE_TZ,
    )
    rows = []
    previous_close = closes[0]
    for index, (timestamp, close) in enumerate(zip(timestamps, closes)):
        open_ = previous_close if index > 0 else close - 0.05
        high = max(open_, close) + 0.3
        low = min(open_, close) - 0.3
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 3))
        previous_close = close
    return _build_df(rows)


def test_broad_uptrend_and_local_downtrend_coexist() -> None:
    df = make_broad_uptrend_local_decline_df()

    result = analyze_dataframe(df=df, symbol="DIVERGE", as_of=analysis_as_of(df))

    assert result["trend"] == "Uptrend"
    assert result["broad_trend"] == "Uptrend"
    assert result["trend_score"] >= 18.0

    assert result["local_trend"] == "Downtrend"
    assert result["local_trend_score"] <= -18.0
    assert result["local_trend_evidence"]

    text = format_analysis_text(result)
    assert "Broad Trend: Uptrend" in text
    assert "Local Session Trend: Downtrend" in text


def test_local_trend_evidence_mentions_session_open_or_range() -> None:
    df = make_broad_uptrend_local_decline_df()

    result = analyze_dataframe(df=df, symbol="DIVERGE2", as_of=analysis_as_of(df))

    assert any(
        "session open" in item.lower() or "session" in item.lower() and "low" in item.lower()
        for item in result["local_trend_evidence"]
    )


def test_local_trend_lookback_is_configurable_and_session_capped() -> None:
    df = make_broad_uptrend_local_decline_df()
    feature_df = add_features(df)

    wide = classify_local_session_trend(feature_df, lookback_bars=20)
    narrow = classify_local_session_trend(feature_df, lookback_bars=3)

    # The final session only has 4 bars, so even the "wide" lookback is capped to
    # the number of bars available in the current session (never looks past it).
    assert int(wide.iloc[-1]["Local_Trend_Lookback_Bars"]) == 4
    # A narrower configured lookback further caps the window within the session.
    assert int(narrow.iloc[-1]["Local_Trend_Lookback_Bars"]) == 3
    # Both should still detect the sharp decline, regardless of window size.
    assert wide.iloc[-1]["Local_Trend"] == "Downtrend"
    assert narrow.iloc[-1]["Local_Trend"] == "Downtrend"


def test_local_trend_is_never_computed_from_broad_horizon_alone() -> None:
    # A brand new session (first bar of the day) has no local history at all, so it
    # must be Neutral even while sitting inside a long, strongly bullish broad trend.
    df = make_broad_uptrend_local_decline_df()
    feature_df = add_features(df)
    local = classify_local_session_trend(feature_df, lookback_bars=20)

    first_bar_of_final_session_index = 288
    assert local.iloc[first_bar_of_final_session_index]["Local_Trend_Lookback_Bars"] == 1
    assert local.iloc[first_bar_of_final_session_index]["Local_Trend"] == "Neutral"


def make_bearish_broad_trend_with_confirmed_bullish_pin_bar() -> pd.DataFrame:
    closes = [120 - (index * 0.30) for index in range(60)]
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=60, freq="15min", tz=EXCHANGE_TZ)
    rows = []
    previous_close = closes[0]
    for index, (timestamp, close) in enumerate(zip(timestamps, closes)):
        open_ = previous_close if index > 0 else close - 0.05
        high = max(open_, close) + 0.25
        low = min(open_, close) - 0.25
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 5))
        previous_close = close
    df = _build_df(rows)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [102.4, 102.55, 100.8, 102.5, 4200]
    return df


def test_pattern_entry_trend_matches_real_decline_for_confirmed_reversal() -> None:
    df = make_bearish_broad_trend_with_confirmed_bullish_pin_bar()

    result = analyze_dataframe(df=df, symbol="PIN", as_of=analysis_as_of(df))

    pin_bar = next(p for p in result["all_detected_patterns"] if p["pattern_name"] == "Bullish Pin Bar")
    assert pin_bar["pattern_entry_trend"] == "Downtrend"
    assert pin_bar["pattern_entry_trend_score"] < -18.0
    assert pin_bar["pattern_entry_trend_lookback_bars"] > 0


def make_bullish_broad_trend_with_lower_wick_rejection() -> pd.DataFrame:
    closes = [100 + (index * 0.35) for index in range(60)]
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=60, freq="15min", tz=EXCHANGE_TZ)
    rows = []
    previous_close = closes[0]
    for index, (timestamp, close) in enumerate(zip(timestamps, closes)):
        open_ = previous_close if index > 0 else close - 0.05
        high = max(open_, close) + 0.25
        low = min(open_, close) - 0.25
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 5))
        previous_close = close
    df = _build_df(rows)
    last_close = df.iloc[len(df) - 2]["Close"]
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [
        last_close + 0.3,
        last_close + 0.35,
        last_close - 3.0,
        last_close + 0.25,
        4000,
    ]
    return df


def test_pattern_entry_trend_reflects_actual_move_not_expected_direction() -> None:
    """A lower-wick-rejection candle inside an uptrend does not qualify as a classic
    bullish reversal (it wasn't preceded by a downtrend), so it should stay a neutral
    geometry-only candidate. Its pattern_entry_trend must still report the real
    immediate pre-pattern move (Uptrend) rather than inheriting the "downtrend"
    direction the detector was checking for, and must not simply mirror the broad
    trend either (it's evaluated independently from a short causal prefix window).
    """
    df = make_bullish_broad_trend_with_lower_wick_rejection()

    result = analyze_dataframe(df=df, symbol="UPREJ", as_of=analysis_as_of(df))

    assert result["broad_trend"] == "Uptrend"
    rejection = next(p for p in result["all_detected_patterns"] if p["pattern_name"] == "Lower-Wick Rejection")
    assert rejection["status"] == "candidate"
    assert rejection["bias"] == "Neutral"
    assert rejection["pattern_entry_trend"] == "Uptrend"
    assert rejection["pattern_entry_trend_score"] > 18.0


def test_pattern_config_validates_new_trend_thresholds() -> None:
    with pytest.raises(ConfigurationError):
        PatternConfig(local_trend_lookback_bars=4).validate()
    with pytest.raises(ConfigurationError):
        PatternConfig(pattern_entry_trend_lookback_bars=4).validate()
    with pytest.raises(ConfigurationError):
        PatternConfig(context_minimum_displacement_atr=0).validate()
    with pytest.raises(ConfigurationError):
        PatternConfig(resistance_proximity_lookback_bars=4).validate()
    with pytest.raises(ConfigurationError):
        PatternConfig(support_resistance_proximity_tolerance=-0.01).validate()
    with pytest.raises(ConfigurationError):
        PatternConfig(structural_duplicate_price_tolerance=-0.01).validate()

    # Sensible non-default values remain valid.
    PatternConfig(
        local_trend_lookback_bars=30,
        pattern_entry_trend_lookback_bars=10,
        context_minimum_displacement_atr=1.5,
        resistance_proximity_lookback_bars=20,
        support_resistance_proximity_tolerance=0.01,
        structural_duplicate_price_tolerance=0.01,
    ).validate()
