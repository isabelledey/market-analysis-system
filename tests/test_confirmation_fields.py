"""Tests for Stage 5: replacing the single ambiguous pattern status with four independent,
explicit fields -- Geometry Status, Context Status, Directional Confirmation, and Follow-Through.

"Status: confirmed" alone cannot tell a reader whether only the candle shape was validated,
whether the market context was appropriate, whether price action actually confirmed the expected
direction, or whether that move has continued since. These tests check both the structured result
and the rendered text distinguish all of that -- and specifically that a context-validated but
not-yet-directionally-confirmed pattern is never described as if its direction were confirmed.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.formatters import format_analysis_text

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


def analysis_as_of(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=16)


def make_real_local_uptrend_then_rejection_df(extra_bar: dict | None = None) -> pd.DataFrame:
    bar_count = 40
    total = bar_count + (1 if extra_bar else 0)
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=total, freq="15min", tz=EXCHANGE_TZ)
    previous_close = 100.0
    rows = []
    for index in range(bar_count - 1):
        close = 100.0 + index * 0.4
        open_ = previous_close
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        rows.append(_bar(timestamps[index], open_, high, low, close, 1500 + index * 5))
        previous_close = close
    reference_close = rows[-1]["Close"]
    rows.append(
        _bar(
            timestamps[bar_count - 1],
            reference_close + 0.1,
            reference_close + 3.0,
            reference_close - 0.1,
            reference_close + 0.15,
            4000,
        )
    )
    if extra_bar is not None:
        rows.append(_bar(timestamps[bar_count], **extra_bar))
    return pd.DataFrame(rows)


def _shooting_star(result: dict) -> dict:
    return next(p for p in result["all_detected_patterns"] if p["pattern_name"] == "Shooting Star")


def test_awaiting_shooting_star_shows_pending_directional_confirmation_not_confirmed() -> None:
    df = make_real_local_uptrend_then_rejection_df()

    result = analyze_dataframe(df=df, symbol="SS_PENDING", as_of=analysis_as_of(df))

    pattern = _shooting_star(result)
    assert pattern["status"] == "tentative"
    assert pattern["geometry_status"] == "Validated"
    assert pattern["context_status"] == "Validated"
    assert pattern["directional_confirmation"] == "Pending"
    assert pattern["follow_through"] == "Pending"

    # The dampener label must match this pattern's own bias (Bearish here, so it dampens the
    # *Bullish* side of the score) rather than a hardcoded "Bearish-Score Dampener" -- that
    # wording is only correct for a Bullish pattern (e.g. an unconfirmed Hammer).
    assert pattern["bias"] == "Bearish"
    text = format_analysis_text(result, include_all_patterns=True)
    assert "Bullish-Score Dampener: Active" in text
    assert "Bearish-Score Dampener: Active" not in text


def test_shooting_star_directionally_confirmed_by_close_below_its_low() -> None:
    df_probe = make_real_local_uptrend_then_rejection_df()

    # Detect directly to get the exact rejection candle's own low, needed to build a confirming bar.
    from stock_pattern_model.config import PatternConfig
    from stock_pattern_model.features import add_features
    from stock_pattern_model.pattern_detector import ShootingStarDetector
    from stock_pattern_model.pattern_detector import classify_intraday_trend
    from stock_pattern_model.pattern_detector import classify_local_session_trend

    feature_df = add_features(df_probe)
    feature_df = classify_intraday_trend(feature_df, lookback_bars=12, pivot_left_bars=2, pivot_right_bars=2, breakout_lookback=20)
    feature_df = classify_local_session_trend(feature_df, lookback_bars=20)
    rejection_event = ShootingStarDetector().detect(feature_df, PatternConfig(), "15m")[-1]
    low = rejection_event.relevant_prices["low"]

    last_timestamp = df_probe.iloc[-1]["Datetime"] + pd.Timedelta(minutes=15)
    confirming_bar = _bar(last_timestamp, low + 0.05, low + 0.1, low - 0.5, low - 0.4, 2000)
    df = pd.concat([df_probe, pd.DataFrame([confirming_bar])], ignore_index=True)

    result = analyze_dataframe(df=df, symbol="SS_CONFIRMED", as_of=analysis_as_of(df))

    pattern = _shooting_star(result)
    assert pattern["status"] == "tentative"  # status alone never changes -- event_state does
    assert pattern["directional_confirmation"] == "Confirmed"
    assert pattern["follow_through"] == "Present"
    # dampener_eligible is a static, detection-time flag that stays true forever once
    # context-validated -- rendered text must not claim the dampener is "Active" once the
    # pattern has actually been directionally confirmed and is scoring on its own full (if
    # tentative-discounted) contribution instead.
    assert pattern["dampener_eligible"] is True
    text = format_analysis_text(result, include_all_patterns=True)
    assert "Bearish-Score Dampener: Active" not in text


def test_shooting_star_invalidated_by_close_above_its_high() -> None:
    df_probe = make_real_local_uptrend_then_rejection_df()

    from stock_pattern_model.config import PatternConfig
    from stock_pattern_model.features import add_features
    from stock_pattern_model.pattern_detector import ShootingStarDetector
    from stock_pattern_model.pattern_detector import classify_intraday_trend
    from stock_pattern_model.pattern_detector import classify_local_session_trend

    feature_df = add_features(df_probe)
    feature_df = classify_intraday_trend(feature_df, lookback_bars=12, pivot_left_bars=2, pivot_right_bars=2, breakout_lookback=20)
    feature_df = classify_local_session_trend(feature_df, lookback_bars=20)
    rejection_event = ShootingStarDetector().detect(feature_df, PatternConfig(), "15m")[-1]
    high = rejection_event.relevant_prices["high"]

    last_timestamp = df_probe.iloc[-1]["Datetime"] + pd.Timedelta(minutes=15)
    invalidating_bar = _bar(last_timestamp, high - 0.05, high + 0.5, high - 0.1, high + 0.4, 2000)
    df = pd.concat([df_probe, pd.DataFrame([invalidating_bar])], ignore_index=True)

    result = analyze_dataframe(df=df, symbol="SS_INVALID", as_of=analysis_as_of(df))

    pattern = _shooting_star(result)
    assert pattern["directional_confirmation"] == "Failed"
    assert pattern["follow_through"] == "Failed"
    # Not closing above the invalidation level would only mean "not invalidated" -- it is a
    # different, separate bar here that genuinely broke back above the high.
    assert pattern["pattern_score_contribution"] == 0.0


def test_geometry_only_pattern_has_not_applicable_context_and_not_required_direction() -> None:
    df = make_real_local_uptrend_then_rejection_df()

    result = analyze_dataframe(df=df, symbol="DOJI_FIELDS", as_of=analysis_as_of(df))

    doji_patterns = [p for p in result["all_detected_patterns"] if p["pattern_name"] == "Doji"]
    assert doji_patterns
    for pattern in doji_patterns:
        assert pattern["geometry_status"] == "Validated"
        assert pattern["context_status"] == "Not Applicable"
        assert pattern["directional_confirmation"] == "Not Required"
        assert pattern["follow_through"] == "Not Applicable"


def _candle(open_price=100.0, high_price=100.8, low_price=99.6, close_price=100.2, volume=1000) -> dict:
    return {"Open": open_price, "High": high_price, "Low": low_price, "Close": close_price, "Volume": volume}


def make_df(rows: list[dict], start: str = "2026-07-10 09:30") -> pd.DataFrame:
    datetimes = pd.date_range(start=start, periods=len(rows), freq="15min", tz=EXCHANGE_TZ)
    out = []
    for timestamp, row in zip(datetimes, rows):
        record = dict(row)
        record["Datetime"] = timestamp
        out.append(record)
    return pd.DataFrame(out)


def make_double_bottom_df(confirmation_close: float | None) -> pd.DataFrame:
    post_confirmation_close = 103.0 if confirmation_close is not None else 102.1
    rows = [
        _candle(100.4, 101.0, 99.9, 100.7),
        _candle(100.8, 101.2, 99.5, 100.0),
        _candle(100.1, 100.3, 95.0, 95.6, 2300),
        _candle(95.8, 97.6, 95.3, 97.1),
        _candle(97.2, 99.3, 96.8, 99.0),
        _candle(99.1, 102.3, 98.9, 101.8, 2100),
        _candle(101.6, 101.8, 99.5, 100.1),
        _candle(100.0, 100.2, 98.0, 98.6),
        _candle(98.5, 99.0, 95.1, 95.8, 2200),
        _candle(95.9, 97.2, 95.6, 96.8),
        _candle(96.8, 98.5, 96.6, 97.8),
        _candle(97.9, 103.0, 97.7, confirmation_close if confirmation_close is not None else 101.6, 2500),
        _candle(102.9, 103.3, 102.0, post_confirmation_close),
    ]
    return make_df(rows)


def test_double_bottom_awaiting_neckline_break_shows_pending() -> None:
    df = make_double_bottom_df(confirmation_close=None)

    result = analyze_dataframe(df, symbol="DB_PENDING", as_of=analysis_as_of(df))

    pattern = next(p for p in result["all_detected_patterns"] if p["pattern_name"] == "Double Bottom")
    assert pattern["status"] == "tentative"
    assert pattern["context_status"] == "Not Applicable"
    assert pattern["directional_confirmation"] == "Pending"
    assert pattern["follow_through"] == "Pending"


def test_double_bottom_neckline_break_shows_confirmed() -> None:
    df = make_double_bottom_df(confirmation_close=102.8)

    result = analyze_dataframe(df, symbol="DB_CONFIRMED", as_of=analysis_as_of(df))

    pattern = next(
        p for p in result["all_detected_patterns"]
        if p["pattern_name"] == "Double Bottom" and p["status"] == "confirmed"
    )
    assert pattern["directional_confirmation"] == "Confirmed"
    assert pattern["follow_through"] == "Present"


def test_rendered_text_distinguishes_all_confirmation_stages_and_never_bare_confirms() -> None:
    df = make_real_local_uptrend_then_rejection_df()

    result = analyze_dataframe(df=df, symbol="TEXT", as_of=analysis_as_of(df), top_pattern_count=10)
    text = format_analysis_text(result, include_all_patterns=True)

    assert "Geometry Status: Validated" in text
    assert "Context Status: Validated" in text
    assert "Directional Confirmation: Pending" in text
    assert "Follow-Through: Pending" in text
    # A context-validated-but-unconfirmed pattern must never be rendered as if bare "confirmed"
    # were the full story -- the explicit Directional Confirmation line is always present too.
    for line_index, line in enumerate(text.splitlines()):
        if line.strip() == "Status: tentative":
            following_lines = text.splitlines()[line_index : line_index + 6]
            assert any("Directional Confirmation:" in later_line for later_line in following_lines)
