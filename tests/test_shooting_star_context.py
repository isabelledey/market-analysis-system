"""Tests for Stage 2: contextual classification of upper-wick rejection candles.

A Shooting Star must require more than bearish candle geometry: a genuine, sufficiently
displaced preceding uptrend, proximity to a recent swing high / resistance area, and no
contradictory local-session downtrend already in force. When any of those fail, the
geometry is preserved (never discarded) but renamed/downgraded to reflect what the
market context actually supports:

- Shooting Star: geometry + genuine uptrend + near resistance + no contradicting local trend.
- Upper-Wick Rejection (Bearish): geometry during an active local downtrend
  (`bearish_continuation_rejection`) -- bearish informational value is preserved.
- Resistance Rejection (Neutral): geometry near a recent high inside a sideways range.
- Upper-Wick Rejection (Neutral): geometry with no clear context at all (fallback).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe

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


def _latest_pattern(result: dict, pattern_name_prefix: str | None = None) -> dict:
    candidates = [p for p in result["all_detected_patterns"] if p["candles_ago"] == 0]
    if pattern_name_prefix is not None:
        candidates = [p for p in candidates if p["pattern_name"] == pattern_name_prefix]
    return candidates[0]


def make_real_local_uptrend_then_rejection_df() -> pd.DataFrame:
    """A genuine, brisk local rise that breaks to a new high and is rejected (long upper wick)."""
    rows = []
    bar_count = 40
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=bar_count, freq="15min", tz=EXCHANGE_TZ)
    previous_close = 100.0
    for index, timestamp in enumerate(timestamps[:-1]):
        close = 100.0 + index * 0.4
        open_ = previous_close
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 5))
        previous_close = close
    reference_close = rows[-1]["Close"]
    rows.append(
        _bar(
            timestamps[-1],
            reference_close + 0.1,
            reference_close + 3.0,
            reference_close - 0.1,
            reference_close + 0.15,
            4000,
        )
    )
    return pd.DataFrame(rows)


def make_broad_uptrend_then_decline_rejection_df(decline_bars: int, bounce_high_add: float) -> pd.DataFrame:
    """A long, brisk broad-trend uptrend (session-blocked to midnight boundaries), followed by a
    short sharp decline in the final session and a weak bounce attempt that gets rejected with a
    long upper wick near the new session low (not near the old, pre-decline resistance).
    """
    prior_bars = 288
    prior_slope = 1.2
    decline_step = 2.2
    total_bars = prior_bars + decline_bars + 1
    timestamps = pd.date_range(start="2026-07-01 00:00", periods=total_bars, freq="15min", tz=EXCHANGE_TZ)

    closes = [100.0 + (index * prior_slope) for index in range(prior_bars)]
    last = closes[-1]
    for _ in range(decline_bars):
        last -= decline_step
        closes.append(last)

    rows = []
    previous_close = closes[0]
    for index in range(prior_bars + decline_bars):
        timestamp = timestamps[index]
        close = closes[index]
        open_ = previous_close if index > 0 else close - 0.05
        high = max(open_, close) + 0.3
        low = min(open_, close) - 0.3
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 3))
        previous_close = close

    bounce_timestamp = timestamps[prior_bars + decline_bars]
    bounce_open = previous_close
    bounce_high = previous_close + bounce_high_add
    bounce_low = previous_close - 0.3
    bounce_close = previous_close - 0.1
    rows.append(_bar(bounce_timestamp, bounce_open, bounce_high, bounce_low, bounce_close, 2500))
    return pd.DataFrame(rows)


def make_sideways_range_then_rejection_df() -> pd.DataFrame:
    """A tight sideways range that suddenly breaks to a new local high and gets rejected."""
    rows = []
    bar_count = 40
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=bar_count, freq="15min", tz=EXCHANGE_TZ)
    previous_close = 100.0
    for index, timestamp in enumerate(timestamps[:-1]):
        close = 100.0 + ((index % 3) - 1) * 0.08
        open_ = previous_close
        high = max(open_, close) + 0.18
        low = min(open_, close) - 0.18
        rows.append(_bar(timestamp, open_, high, low, close, 1500))
        previous_close = close
    timestamp = timestamps[-1]
    high = previous_close + 1.0
    rows.append(_bar(timestamp, previous_close, high, previous_close - 0.2, previous_close + 0.05, 3000))
    return pd.DataFrame(rows)


def test_real_local_uptrend_and_resistance_break_yields_classic_shooting_star() -> None:
    df = make_real_local_uptrend_then_rejection_df()

    result = analyze_dataframe(df=df, symbol="SS_CLASSIC", as_of=analysis_as_of(df))

    pattern = _latest_pattern(result, "Shooting Star")
    assert pattern["bias"] == "Bearish"
    # Stage 5: a classic Shooting Star is context-validated but starts TENTATIVE/awaiting
    # directional confirmation -- it is never immediately "confirmed" from context alone.
    assert pattern["status"] == "tentative"
    assert pattern["dampener_eligible"] is True
    assert pattern["rejection_confirmation_state"] == "context_validated"
    assert pattern["context_quality"] == "validated"
    assert pattern["geometry_label"] == "long_upper_rejection"
    assert "shooting_star" in pattern["context_tags"]
    assert "near_resistance" in pattern["context_tags"]


def test_identical_geometry_during_local_decline_is_bearish_rejection_not_shooting_star() -> None:
    df = make_broad_uptrend_then_decline_rejection_df(decline_bars=4, bounce_high_add=2.0)

    result = analyze_dataframe(df=df, symbol="SS_DECLINE", as_of=analysis_as_of(df))

    assert result["local_trend"] == "Downtrend"
    pattern = _latest_pattern(result, "Upper-Wick Rejection")
    # Bearish informational value is preserved -- this is not silently flattened to Neutral.
    assert pattern["bias"] == "Bearish"
    # Stage 5: still TENTATIVE/awaiting directional confirmation, just without the classic label.
    assert pattern["status"] == "tentative"
    assert pattern["dampener_eligible"] is True
    assert pattern["context_quality"] == "geometry_only"
    assert pattern["geometry_label"] == "long_upper_rejection"
    assert "bearish_continuation_rejection" in pattern["context_tags"]
    assert "local_trend_downtrend" in pattern["context_tags"]
    assert not any(p["pattern_name"] == "Shooting Star" for p in result["all_detected_patterns"])


def test_upper_wick_inside_sideways_range_is_resistance_rejection() -> None:
    df = make_sideways_range_then_rejection_df()

    result = analyze_dataframe(df=df, symbol="SS_SIDEWAYS", as_of=analysis_as_of(df))

    assert abs(result["local_trend_score"]) < 18.0
    pattern = _latest_pattern(result, "Resistance Rejection")
    assert pattern["bias"] == "Neutral"
    assert pattern["status"] == "candidate"
    assert pattern["context_quality"] == "geometry_only"
    assert "resistance_rejection" in pattern["context_tags"]
    assert "near_resistance" in pattern["context_tags"]
    assert not any(p["pattern_name"] == "Shooting Star" for p in result["all_detected_patterns"])


def test_broad_uptrend_combined_with_local_decline_lets_local_context_decide_the_label() -> None:
    """Broad trend may legitimately stay bullish while the local session context still vetoes the
    classic reversal label -- local context, not the broad trend, determines the pattern label.
    """
    df = make_broad_uptrend_then_decline_rejection_df(decline_bars=3, bounce_high_add=2.0)

    result = analyze_dataframe(df=df, symbol="SS_MIXED", as_of=analysis_as_of(df))

    assert result["broad_trend"] == "Uptrend"
    assert result["trend_score"] >= 18.0
    assert result["local_trend"] == "Downtrend"

    pattern = _latest_pattern(result, "Upper-Wick Rejection")
    assert pattern["bias"] == "Bearish"
    assert "bearish_continuation_rejection" in pattern["context_tags"]
    assert not any(p["pattern_name"] == "Shooting Star" for p in result["all_detected_patterns"])
