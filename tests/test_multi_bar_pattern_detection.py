"""Detection-level regression tests for every registered pattern detector.

These tests exist because of a real bug: `_same_pattern_session` keyed continuity
on the exchange calendar date, so any multi-bar pattern (Morning Star, Evening
Star, Engulfing, Inside Bar, Breakout/Breakdown) was structurally unreachable on
daily-or-larger intervals -- every daily bar sits in its own "session" by
construction, so the same-session check always failed and silently swallowed
every multi-bar detection. No existing test built real OHLC and asked a
detector to actually fire on `interval="1d"`, so this went unnoticed.

Each test below builds a minimal, hand-computed OHLC fixture that satisfies a
detector's exact formal definition and asserts detection succeeds (or, for the
near-miss / gap cases, correctly fails). This is deliberately at the detector
level (calling `SomeDetector().detect(df, config, interval)` directly) rather
than through the full `analyze_dataframe` pipeline, so these tests isolate
detection geometry and interval handling from unrelated context/lifecycle
filtering.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.config import PatternConfig
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.features import add_features
from stock_pattern_model.pattern_detector import (
    BearishEngulfingDetector,
    BreakdownDetector,
    BreakoutDetector,
    BullishEngulfingDetector,
    BullishPinBarDetector,
    DojiDetector,
    DoubleBottomDetector,
    DoubleTopDetector,
    EveningStarDetector,
    HammerDetector,
    InsideBarDetector,
    InsideBarFailureDetector,
    MorningStarDetector,
    ShootingStarDetector,
)

TZ = ZoneInfo("America/New_York")
CONFIG = PatternConfig()


def _bar(ts, open_, high, low, close, volume) -> dict:
    return {"Datetime": ts, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}


def _daily_timestamps(n: int, start: str = "2026-01-05") -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, periods=n, freq="1D", tz=TZ))


def _hourly_timestamps_same_session(n: int, *, start_hour: int = 13, start_minute: int = 30) -> list[pd.Timestamp]:
    base = pd.Timestamp("2026-01-05", tz=TZ) + pd.Timedelta(hours=start_hour, minutes=start_minute)
    return [base + pd.Timedelta(hours=i) for i in range(n)]


def _baseline_rows(timestamps: list[pd.Timestamp], *, price: float = 100.0, volume: int = 5_000_000) -> list[dict]:
    return [_bar(ts, price, price + 0.4, price - 0.4, price + 0.05, volume) for ts in timestamps]


def _build(rows: list[dict]) -> pd.DataFrame:
    return add_features(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Two-bar patterns, tested on both daily and intraday intervals -- this is the
# exact regression surface for the session-key bug.
# ---------------------------------------------------------------------------


def test_bullish_engulfing_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(10)
    rows = _baseline_rows(ts[:8])
    rows.append(_bar(ts[8], 100.6, 100.7, 99.0, 99.2, 5_000_000))
    rows.append(_bar(ts[9], 99.0, 101.5, 98.8, 101.2, 50_000_000))
    df = _build(rows)

    events = BullishEngulfingDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices["second_close"] == 101.2


def test_bullish_engulfing_detected_on_intraday_interval() -> None:
    ts = _hourly_timestamps_same_session(7)
    rows = _baseline_rows(ts[:5])
    rows.append(_bar(ts[5], 100.6, 100.7, 99.0, 99.2, 5_000_000))
    rows.append(_bar(ts[6], 99.0, 101.5, 98.8, 101.2, 50_000_000))
    df = _build(rows)

    events = BullishEngulfingDetector().detect(df, CONFIG, "1h")

    assert len(events) == 1


def test_bearish_engulfing_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(10)
    rows = _baseline_rows(ts[:8])
    rows.append(_bar(ts[8], 99.4, 100.9, 99.3, 100.6, 5_000_000))
    rows.append(_bar(ts[9], 100.8, 100.9, 98.0, 98.3, 50_000_000))
    df = _build(rows)

    events = BearishEngulfingDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1


def test_inside_bar_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(10)
    rows = _baseline_rows(ts[:9])
    rows[-1] = _bar(ts[8], 99.0, 103.0, 97.0, 100.0, 5_000_000)  # mother bar
    rows.append(_bar(ts[9], 99.5, 101.0, 98.5, 100.2, 5_000_000))  # inside bar
    df = _build(rows)

    events = InsideBarDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices == {
        "mother_high": 103.0,
        "mother_low": 97.0,
        "inside_high": 101.0,
        "inside_low": 98.5,
    }


def test_breakout_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(21)
    rows = _baseline_rows(ts[:20])
    rows.append(_bar(ts[20], 100.5, 105.0, 100.3, 104.5, 30_000_000))
    df = _build(rows)

    events = BreakoutDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices["close"] == 104.5


def test_breakdown_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(21)
    rows = _baseline_rows(ts[:20])
    rows.append(_bar(ts[20], 100.0, 100.2, 95.0, 95.5, 30_000_000))
    df = _build(rows)

    events = BreakdownDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices["close"] == 95.5


# ---------------------------------------------------------------------------
# Three-bar patterns (Morning Star / Evening Star / Inside Bar Failure).
# ---------------------------------------------------------------------------


def test_inside_bar_failure_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(11)
    rows = _baseline_rows(ts[:9])
    rows[-1] = _bar(ts[8], 99.0, 103.0, 97.0, 100.0, 5_000_000)  # mother bar
    rows.append(_bar(ts[9], 99.5, 101.0, 98.5, 100.2, 5_000_000))  # inside bar
    rows.append(_bar(ts[10], 100.5, 104.0, 99.5, 99.8, 50_000_000))  # bearish failure sweep
    df = _build(rows)

    events = InsideBarFailureDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].bias == "Bearish"


def test_morning_star_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(3)
    rows = [
        _bar(ts[0], 100.0, 100.3, 94.5, 95.0, 5_000_000),  # bearish impulse, body_ratio ~0.86
        _bar(ts[1], 94.9, 95.1, 94.2, 94.95, 5_000_000),  # small star, gapped below first close
        _bar(ts[2], 95.2, 98.5, 95.1, 98.0, 5_000_000),  # bullish recovery above midpoint (97.5)
    ]
    df = _build(rows)

    events = MorningStarDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices["recovery_close"] == 98.0


def test_morning_star_detected_on_intraday_interval() -> None:
    ts = _hourly_timestamps_same_session(3)
    rows = [
        _bar(ts[0], 100.0, 100.3, 94.5, 95.0, 5_000_000),
        _bar(ts[1], 94.9, 95.1, 94.2, 94.95, 5_000_000),
        _bar(ts[2], 95.2, 98.5, 95.1, 98.0, 5_000_000),
    ]
    df = _build(rows)

    events = MorningStarDetector().detect(df, CONFIG, "1h")

    assert len(events) == 1


def test_morning_star_near_miss_does_not_fire() -> None:
    """First candle's body ratio (0.41) falls just short of the 0.45 minimum --
    this mirrors the real near-miss found on PYPL's 2026-08-31 candle. The
    detector must reject it, not just accept anything that "looks like" a dip
    and bounce.
    """
    ts = _daily_timestamps(3)
    open0, high0, low0 = 100.0, 100.3, 96.8
    close0 = open0 - 0.41 * (high0 - low0)  # body_ratio == 0.41, below the 0.45 gate
    rows = [
        _bar(ts[0], open0, high0, low0, close0, 5_000_000),
        _bar(ts[1], close0 - 0.1, close0 + 0.1, close0 - 0.5, close0 - 0.05, 5_000_000),
        _bar(ts[2], close0 - 0.1, 99.0, close0 - 0.5, 98.8, 5_000_000),
    ]
    df = _build(rows)

    assert df.iloc[0]["Body_Ratio"] < 0.45
    events = MorningStarDetector().detect(df, CONFIG, "1d")

    assert events == []


def test_evening_star_detected_on_daily_interval() -> None:
    ts = _daily_timestamps(3)
    rows = [
        _bar(ts[0], 95.0, 100.5, 94.8, 100.0, 5_000_000),  # bullish impulse
        _bar(ts[1], 100.1, 100.8, 99.9, 100.05, 5_000_000),  # small star, gapped above first close
        _bar(ts[2], 99.9, 100.0, 96.0, 96.5, 5_000_000),  # bearish reversal below midpoint (97.5)
    ]
    df = _build(rows)

    events = EveningStarDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].relevant_prices["reversal_close"] == 96.5


# ---------------------------------------------------------------------------
# The interval-aware continuity guard itself: a legitimate multi-day gap
# (weekend) must still count as continuous, while an abnormally large gap
# (e.g. an extended trading halt) must not.
# ---------------------------------------------------------------------------


def test_daily_pattern_spans_a_weekend_gap() -> None:
    ts = [
        pd.Timestamp("2026-01-02", tz=TZ),  # Friday
        pd.Timestamp("2026-01-05", tz=TZ),  # following Monday
        pd.Timestamp("2026-01-06", tz=TZ),
    ]
    rows = [
        _bar(ts[0], 100.0, 100.3, 94.5, 95.0, 5_000_000),
        _bar(ts[1], 94.9, 95.1, 94.2, 94.95, 5_000_000),
        _bar(ts[2], 95.2, 98.5, 95.1, 98.0, 5_000_000),
    ]
    df = _build(rows)

    events = MorningStarDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1


def test_daily_pattern_rejected_across_an_extended_halt_gap() -> None:
    ts = [
        pd.Timestamp("2026-01-02", tz=TZ),
        pd.Timestamp("2026-01-05", tz=TZ),
        pd.Timestamp("2026-03-01", tz=TZ),  # ~8-week gap: not a normal trading cadence
    ]
    rows = [
        _bar(ts[0], 100.0, 100.3, 94.5, 95.0, 5_000_000),
        _bar(ts[1], 94.9, 95.1, 94.2, 94.95, 5_000_000),
        _bar(ts[2], 95.2, 98.5, 95.1, 98.0, 5_000_000),
    ]
    df = _build(rows)

    events = MorningStarDetector().detect(df, CONFIG, "1d")

    assert events == []


# ---------------------------------------------------------------------------
# Single-candle patterns -- unaffected by the session-key bug (no multi-bar
# continuity check), included here for full-registry coverage.
# ---------------------------------------------------------------------------


def test_doji_detected() -> None:
    ts = _daily_timestamps(3)
    rows = [
        _bar(ts[0], 100, 101.0, 99.0, 100.8, 5_000_000),  # real body
        _bar(ts[1], 100, 102.0, 98.0, 100.08, 5_000_000),  # doji: tiny body, wide range
        _bar(ts[2], 100, 101.0, 99.0, 99.2, 5_000_000),  # real body
    ]
    df = _build(rows)

    events = DojiDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].pattern_start_index == 1


def test_hammer_and_bullish_pin_bar_detected() -> None:
    ts = _daily_timestamps(3)
    rows = [
        _bar(ts[0], 100, 100.5, 99.5, 100.1, 5_000_000),
        _bar(ts[1], 100, 100.3, 99.7, 100.0, 5_000_000),
        _bar(ts[2], 100.0, 100.3, 95.0, 100.2, 5_000_000),  # long lower wick, close near high
    ]
    df = _build(rows)

    assert len(HammerDetector().detect(df, CONFIG, "1d")) == 1
    assert len(BullishPinBarDetector().detect(df, CONFIG, "1d")) == 1


def test_shooting_star_detected() -> None:
    ts = _daily_timestamps(3)
    rows = [
        _bar(ts[0], 100, 100.5, 99.5, 99.9, 5_000_000),
        _bar(ts[1], 100, 100.3, 99.7, 100.0, 5_000_000),
        _bar(ts[2], 100.0, 105.0, 99.7, 99.8, 5_000_000),  # long upper wick, close near low
    ]
    df = _build(rows)

    events = ShootingStarDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1


# ---------------------------------------------------------------------------
# Structural swing patterns (Double Top / Double Bottom). These use pivot
# geometry rather than `_same_pattern_session`, so they were never affected by
# the bug, but are included for full-registry coverage.
# ---------------------------------------------------------------------------


def test_double_top_detected() -> None:
    ts = _daily_timestamps(11)
    highs = [100.0, 100.5, 105.0, 101.0, 100.0, 99.0, 101.0, 102.0, 105.02, 101.0, 100.0]
    lows = [99.7, 100.0, 104.5, 100.5, 99.5, 90.0, 100.5, 101.5, 104.5, 100.5, 99.5]
    rows = [
        _bar(ts[i], (highs[i] + lows[i]) / 2, highs[i], lows[i], (highs[i] + lows[i]) / 2, 5_000_000)
        for i in range(11)
    ]
    df = _build(rows)

    events = DoubleTopDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].status == PatternStatus.TENTATIVE
    assert events[0].relevant_prices["first_peak"] == 105.0


def test_double_bottom_detected() -> None:
    ts = _daily_timestamps(11)
    lows = [100.0, 99.5, 95.0, 99.0, 100.0, 101.0, 99.0, 98.0, 94.98, 99.0, 100.0]
    highs = [100.3, 100.0, 95.5, 99.5, 100.5, 110.0, 99.5, 98.5, 95.5, 99.5, 100.5]
    rows = [
        _bar(ts[i], (highs[i] + lows[i]) / 2, highs[i], lows[i], (highs[i] + lows[i]) / 2, 5_000_000)
        for i in range(11)
    ]
    df = _build(rows)

    events = DoubleBottomDetector().detect(df, CONFIG, "1d")

    assert len(events) == 1
    assert events[0].status == PatternStatus.TENTATIVE
    assert events[0].relevant_prices["first_bottom"] == 95.0
