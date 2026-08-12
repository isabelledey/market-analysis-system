"""Tests for Stage 4: staged confirmation model for lower-wick rejection (Hammer / Bullish Pin
Bar).

A lower-wick rejection near a support area, or after a genuinely displaced preceding downtrend,
is context-validated but must not immediately receive a full bullish score. It stages through:

- `detected`: geometry only, no context validation -- Neutral, informational.
- `context_validated`: near support or after a genuine downtrend -- becomes Bullish/TENTATIVE and
  either a small, bounded, configurable dampener (while awaiting confirmation) or a full
  contribution (once directionally confirmed) or nothing (once invalidated).
- `directionally_confirmed`: a later close above the rejection candle's high -- a real, if modest,
  bullish contribution (discounted by the same `tentative_signal_multiplier` used for all
  tentative patterns, since these patterns never leave `status=tentative`).
- `invalidated`: a later close below the rejection candle's low -- contributes nothing.

The forward-looking half of this state machine is resolved by the existing, pre-Stage-4 lifecycle
scanner in analysis.py (`_apply_generic_pattern_lifecycle`), extended with a new directional-
confirmation check, rather than a second, duplicate forward scan inside the detector.
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


def _hammer_pattern(result: dict) -> dict:
    return next(p for p in result["all_detected_patterns"] if p["pattern_id"] == "hammer")


def make_downtrend_then_hammer_df(*, extra_bar: dict | None = None) -> pd.DataFrame:
    """A genuine downtrend into a classic hammer candle, optionally followed by one more bar."""
    n = 60
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=n, freq="15min", tz=EXCHANGE_TZ)
    closes = [120 - (index * 0.30) for index in range(n - 1)]
    rows = []
    previous_close = closes[0]
    for index, timestamp in enumerate(timestamps[:-1]):
        close = closes[index]
        open_ = previous_close
        high = max(open_, close) + 0.25
        low = min(open_, close) - 0.25
        rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 5))
        previous_close = close
    # Classic hammer geometry (matches the existing Stage-1/2 fixtures): small body near the top,
    # long lower wick.
    rows.append(_bar(timestamps[n - 1], 102.4, 102.55, 100.8, 102.5, 4200))
    df = pd.DataFrame(rows)
    if extra_bar is not None:
        next_timestamp = timestamps[n - 1] + pd.Timedelta(minutes=15)
        df = pd.concat([df, pd.DataFrame([_bar(next_timestamp, **extra_bar)])], ignore_index=True)
    return df


def test_lower_wick_rejection_with_no_follow_through_stays_a_bounded_dampener() -> None:
    df = make_downtrend_then_hammer_df()

    result = analyze_dataframe(df=df, symbol="NOFOLLOW", as_of=analysis_as_of(df))

    hammer = _hammer_pattern(result)
    assert hammer["status"] == "tentative"
    assert hammer["event_state"] in {"new", "active"}
    assert hammer["dampener_eligible"] is True
    assert hammer["rejection_confirmation_state"] == "context_validated"
    # Not a full, unconditional bullish signal -- just the small configured dampener.
    assert 0 < hammer["pattern_score_contribution"] < 3.0


def test_lower_wick_rejection_confirmed_by_later_close_above_its_high() -> None:
    df = make_downtrend_then_hammer_df(
        extra_bar={"open_": 102.5, "high": 103.2, "low": 102.4, "close": 103.1, "volume": 2000}
    )

    result = analyze_dataframe(df=df, symbol="CONFIRMED", as_of=analysis_as_of(df))

    hammer = _hammer_pattern(result)
    assert hammer["status"] == "tentative"
    assert hammer["event_state"] == "directionally_confirmed"
    # A real, materially larger contribution than the awaiting-confirmation dampener -- but still
    # "modest": tentative patterns are always discounted by tentative_signal_multiplier.
    assert hammer["pattern_score_contribution"] > 3.0


def test_lower_wick_rejection_invalidated_by_later_close_below_its_low() -> None:
    df = make_downtrend_then_hammer_df(
        extra_bar={"open_": 102.3, "high": 102.4, "low": 99.5, "close": 100.0, "volume": 2000}
    )

    result = analyze_dataframe(df=df, symbol="INVALID", as_of=analysis_as_of(df))

    hammer = _hammer_pattern(result)
    assert hammer["status"] == "tentative"
    assert hammer["event_state"] == "invalidated"
    assert hammer["pattern_score_contribution"] == 0.0
    assert hammer["dampener_eligible"] is True  # was eligible; invalidation removes the effect
    assert not any(
        p["pattern_id"] == "hammer" and p["candles_ago"] == 0 and p["pattern_score_contribution"] != 0
        for p in result["all_detected_patterns"]
    )


def make_v_shaped_range_df() -> pd.DataFrame:
    """A decline to a real support level (~97), a recovery, and a large lower-wick candle whose
    low (99.0) sits comfortably above that support -- "mid-range", not testing any level.
    """
    n = 40
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=n, freq="15min", tz=EXCHANGE_TZ)
    closes = [101.0 - (index * 0.2) for index in range(20)]
    closes += [97.2 + (index * 0.2) for index in range(19)]
    rows = []
    previous_close = closes[0] + 0.05
    for index, timestamp in enumerate(timestamps[:-1]):
        close = closes[index]
        open_ = previous_close
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        rows.append(_bar(timestamp, open_, high, low, close, 1500))
        previous_close = close
    timestamp = timestamps[-1]
    rows.append(_bar(timestamp, previous_close, previous_close + 0.1, 99.0, previous_close - 0.05, 3000))
    return pd.DataFrame(rows)


def make_sideways_near_support_df() -> pd.DataFrame:
    """A tight sideways range where the lower wick tests right at the recent range low."""
    n = 40
    timestamps = pd.date_range(start="2026-07-10 09:30", periods=n, freq="15min", tz=EXCHANGE_TZ)
    rows = []
    previous_close = 100.0
    for index, timestamp in enumerate(timestamps[:-1]):
        close = 100.0 + ((index % 3) - 1) * 0.08
        open_ = previous_close
        high = max(open_, close) + 0.18
        low = min(open_, close) - 0.18
        rows.append(_bar(timestamp, open_, high, low, close, 1500))
        previous_close = close
    timestamp = timestamps[-1]
    low = previous_close - 0.3
    rows.append(_bar(timestamp, previous_close, previous_close + 0.1, low, previous_close + 0.05, 3000))
    return pd.DataFrame(rows)


def test_lower_wick_rejection_near_support_is_context_validated() -> None:
    df = make_sideways_near_support_df()

    result = analyze_dataframe(df=df, symbol="NEARSUPPORT", as_of=analysis_as_of(df))

    hammer = _hammer_pattern(result)
    assert hammer["bias"] == "Bullish"
    assert hammer["dampener_eligible"] is True
    assert hammer["rejection_confirmation_state"] == "context_validated"


def test_lower_wick_rejection_mid_range_is_not_context_validated() -> None:
    df = make_v_shaped_range_df()

    result = analyze_dataframe(df=df, symbol="MIDRANGE", as_of=analysis_as_of(df))

    hammer = _hammer_pattern(result)
    assert hammer["bias"] == "Neutral"
    assert hammer["dampener_eligible"] is False
    assert hammer["rejection_confirmation_state"] == "detected"
    assert hammer["pattern_score_contribution"] == 0.0


def test_dampener_is_a_small_bounded_offset_to_bearish_score() -> None:
    """The same bearish downtrend, with vs. without a trailing context-validated (but
    unconfirmed) hammer, should differ by exactly the small, bounded dampener -- not by a full
    base-score-sized contribution.
    """

    def make_bearish_trend(*, with_hammer: bool) -> pd.DataFrame:
        closes = [120 - (index * 0.30) for index in range(60)]
        timestamps = pd.date_range(start="2026-07-10 09:30", periods=60, freq="15min", tz=EXCHANGE_TZ)
        rows = []
        previous_close = closes[0]
        for index, timestamp in enumerate(timestamps):
            close = closes[index]
            open_ = previous_close if index > 0 else close - 0.05
            high = max(open_, close) + 0.25
            low = min(open_, close) - 0.25
            rows.append(_bar(timestamp, open_, high, low, close, 1500 + index * 5))
            previous_close = close
        df = pd.DataFrame(rows)
        if with_hammer:
            df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [
                102.4,
                102.55,
                100.8,
                102.5,
                4200,
            ]
        return df

    control_df = make_bearish_trend(with_hammer=False)
    dampened_df = make_bearish_trend(with_hammer=True)
    control_result = analyze_dataframe(df=control_df, symbol="CTRL", as_of=analysis_as_of(control_df))
    dampened_result = analyze_dataframe(df=dampened_df, symbol="DAMP", as_of=analysis_as_of(dampened_df))

    hammer = _hammer_pattern(dampened_result)
    assert hammer["dampener_eligible"] is True
    assert hammer["event_state"] in {"new", "active"}

    score_difference = dampened_result["net_signal_score"] - control_result["net_signal_score"]
    # The dampener nudges the net score toward neutral, but only by its own small, bounded amount
    # -- nowhere near a full, directionally confirmed bullish contribution (base_score=10).
    assert 0 < score_difference < 3.0
    assert score_difference == round(hammer["pattern_score_contribution"], 2)
