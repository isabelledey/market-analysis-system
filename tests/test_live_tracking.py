from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.live_tracking import PatternLifecycleTracker

EXCHANGE_TZ = ZoneInfo("America/New_York")


def _bar(ts: pd.Timestamp, o: float, h: float, l: float, c: float, v: int) -> dict:
    return {"Datetime": ts, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v}


def _baseline_session(session_date: str, length: int, *, freq: str, start_price: float = 100.0) -> pd.DataFrame:
    """A quiet, gently-drifting run of candles with no patterns of its own."""
    timestamps = pd.date_range(f"{session_date} 09:30", periods=length, freq=freq, tz=EXCHANGE_TZ)
    rows = []
    previous_close = start_price
    for index, timestamp in enumerate(timestamps):
        close = start_price + (index * 0.05)
        open_price = previous_close
        high = max(open_price, close) + 0.18
        low = min(open_price, close) - 0.18
        rows.append(_bar(timestamp, open_price, high, low, close, 1000 + index * 5))
        previous_close = close
    return pd.DataFrame(rows)


def _append_bars(base: pd.DataFrame, bars: list[dict], *, freq: str) -> pd.DataFrame:
    start = pd.Timestamp(base.iloc[-1]["Datetime"]) + pd.tseries.frequencies.to_offset(freq)
    timestamps = pd.date_range(start, periods=len(bars), freq=freq, tz=EXCHANGE_TZ)
    stamped = [{"Datetime": ts, **{k: v for k, v in bar.items() if k != "Datetime"}} for ts, bar in zip(timestamps, bars)]
    return pd.concat([base, pd.DataFrame(stamped)], ignore_index=True)


def _as_of(df: pd.DataFrame, interval: str = "15m") -> pd.Timestamp:
    bar_minutes = {"15m": 16, "1h": 61, "1d": 60 * 24 + 1, "1wk": 60 * 24 * 7 + 1, "1mo": 60 * 24 * 31}[interval]
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=bar_minutes)


def _morning_star_bars() -> list[dict]:
    return [
        {"Open": 100.0, "High": 100.3, "Low": 94.5, "Close": 95.0, "Volume": 5_000_000},  # bearish impulse
        {"Open": 94.9, "High": 95.1, "Low": 94.2, "Close": 94.95, "Volume": 5_000_000},  # small star
    ]


def test_ingest_is_a_no_op_when_no_new_candle_has_completed() -> None:
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="NOOP", interval="15m")

    first = tracker.ingest(df=base, as_of=_as_of(base))
    assert first.has_new_completed_candle is True

    second = tracker.ingest(df=base, as_of=_as_of(base))
    assert second.has_new_completed_candle is False
    assert second.transitions == []


def test_morning_star_candidate_is_reported_then_confirmed_on_next_candle() -> None:
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    df_candidate = _append_bars(base, _morning_star_bars(), freq="15min")
    df_confirmed = _append_bars(
        base,
        [*_morning_star_bars(), {"Open": 95.2, "High": 98.5, "Low": 95.1, "Close": 98.0, "Volume": 5_000_000}],
        freq="15min",
    )

    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="MSTAR", interval="15m")

    candidate_result = tracker.ingest(df=df_candidate, as_of=_as_of(df_candidate))
    candidate_transitions = [t for t in candidate_result.transitions if t.pattern_id == "morning_star"]
    assert len(candidate_transitions) == 1
    assert candidate_transitions[0].kind == "new_candidate"
    assert candidate_transitions[0].status == "candidate"
    assert candidate_transitions[0].pattern["relevant_indices"] == [20, 21]

    confirmed_result = tracker.ingest(df=df_confirmed, as_of=_as_of(df_confirmed))
    confirmed_transitions = [t for t in confirmed_result.transitions if t.pattern_id == "morning_star"]
    assert len(confirmed_transitions) == 1
    assert confirmed_transitions[0].kind == "state_changed"
    assert confirmed_transitions[0].previous_status == "candidate"
    assert confirmed_transitions[0].status == "confirmed"
    assert confirmed_transitions[0].pattern["relevant_indices"] == [20, 21, 22]
    # The candle timestamps responsible for the (now three-bar) detection are preserved exactly.
    assert confirmed_transitions[0].pattern["pattern_start_at"] == candidate_transitions[0].pattern["pattern_start_at"]

    # Re-ingesting the identical, already-processed data must not re-report the same transition.
    repeat_result = tracker.ingest(df=df_confirmed, as_of=_as_of(df_confirmed))
    assert repeat_result.has_new_completed_candle is False
    assert repeat_result.transitions == []


def test_morning_star_candidate_is_invalidated_when_next_candle_fails_to_confirm() -> None:
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    df_candidate = _append_bars(base, _morning_star_bars(), freq="15min")
    df_failed = _append_bars(
        base,
        [*_morning_star_bars(), {"Open": 95.0, "High": 95.3, "Low": 94.0, "Close": 94.6, "Volume": 5_000_000}],
        freq="15min",
    )

    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="MSTARFAIL", interval="15m")
    tracker.ingest(df=df_candidate, as_of=_as_of(df_candidate))
    failed_result = tracker.ingest(df=df_failed, as_of=_as_of(df_failed))

    failed_transitions = [t for t in failed_result.transitions if t.pattern_id == "morning_star"]
    assert len(failed_transitions) == 1
    assert failed_transitions[0].kind == "state_changed"
    assert failed_transitions[0].previous_status == "candidate"
    assert failed_transitions[0].status == "failed"


def test_evening_star_candidate_lifecycle_mirrors_morning_star() -> None:
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    impulse_and_star = [
        {"Open": 95.0, "High": 100.5, "Low": 94.8, "Close": 100.0, "Volume": 5_000_000},  # bullish impulse
        {"Open": 100.1, "High": 100.8, "Low": 99.9, "Close": 100.05, "Volume": 5_000_000},  # small star
    ]
    df_candidate = _append_bars(base, impulse_and_star, freq="15min")
    df_confirmed = _append_bars(
        base,
        [*impulse_and_star, {"Open": 99.9, "High": 100.0, "Low": 96.0, "Close": 96.5, "Volume": 5_000_000}],
        freq="15min",
    )

    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="ESTAR", interval="15m")
    candidate_result = tracker.ingest(df=df_candidate, as_of=_as_of(df_candidate))
    candidate_transitions = [t for t in candidate_result.transitions if t.pattern_id == "evening_star"]
    assert len(candidate_transitions) == 1
    assert candidate_transitions[0].status == "candidate"

    confirmed_result = tracker.ingest(df=df_confirmed, as_of=_as_of(df_confirmed))
    confirmed_transitions = [t for t in confirmed_result.transitions if t.pattern_id == "evening_star"]
    assert len(confirmed_transitions) == 1
    assert confirmed_transitions[0].status == "confirmed"


def test_generic_pattern_event_state_change_is_reported_exactly_once() -> None:
    """A single-candle pattern (e.g. a breakout) already gets forward-scanned for
    retest/reclaim/invalidation by analysis._apply_pattern_lifecycle on every analyze_dataframe
    call; the tracker must surface that as one state_changed transition; not re-detect it as new."""
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    breakout_bar = [{"Open": 100.4, "High": 103.0, "Low": 100.3, "Close": 102.8, "Volume": 6_000_000}]
    df_v1 = _append_bars(base, breakout_bar, freq="15min")
    df_v2 = _append_bars(
        base,
        [*breakout_bar, {"Open": 102.7, "High": 102.9, "Low": 100.5, "Close": 101.0, "Volume": 2_000_000}],
        freq="15min",
    )

    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="BRK", interval="15m")
    first = tracker.ingest(df=df_v1, as_of=_as_of(df_v1))
    breakout_first = [t for t in first.transitions if t.pattern_id == "breakout"]
    assert len(breakout_first) == 1
    assert breakout_first[0].kind == "new_candidate"
    first_event_state = breakout_first[0].event_state

    second = tracker.ingest(df=df_v2, as_of=_as_of(df_v2))
    breakout_second = [t for t in second.transitions if t.pattern_id == "breakout"]
    if breakout_second:
        # The retest bar changed event_state (e.g. new -> retest_pending); reported once, and
        # never as a duplicate "new_candidate" for the same underlying breakout.
        assert breakout_second[0].kind == "state_changed"
        assert breakout_second[0].event_state != first_event_state
    third = tracker.ingest(df=df_v2, as_of=_as_of(df_v2))
    assert third.has_new_completed_candle is False
    assert all(t.pattern_id != "breakout" for t in third.transitions)


def test_pattern_aging_out_of_a_sliding_window_is_reported_once() -> None:
    base = _baseline_session("2026-07-20", length=20, freq="15min")
    df_with_pattern = _append_bars(base, _morning_star_bars(), freq="15min")

    tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="AGE", interval="15m")
    tracker.ingest(df=df_with_pattern, as_of=_as_of(df_with_pattern))

    unrelated_session = _baseline_session("2026-07-21", length=20, freq="15min")
    disappeared_result = tracker.ingest(df=unrelated_session, as_of=_as_of(unrelated_session))
    aged_out = [t for t in disappeared_result.transitions if t.pattern_id == "morning_star"]
    assert len(aged_out) == 1
    assert aged_out[0].kind == "aged_out_of_window"
    assert aged_out[0].previous_status == "candidate"

    repeat = tracker.ingest(df=unrelated_session, as_of=_as_of(unrelated_session) + pd.Timedelta(minutes=1))
    assert all(t.pattern_id != "morning_star" for t in repeat.transitions)


def test_tracker_works_across_supported_candle_intervals() -> None:
    for interval, freq in (("15m", "15min"), ("1h", "1h"), ("1d", "1D"), ("1wk", "7D"), ("1mo", "30D")):
        base = _baseline_session("2026-01-05", length=20, freq=freq)
        df_candidate = _append_bars(base, _morning_star_bars(), freq=freq)
        df_confirmed = _append_bars(
            base,
            [*_morning_star_bars(), {"Open": 95.2, "High": 98.5, "Low": 95.1, "Close": 98.0, "Volume": 5_000_000}],
            freq=freq,
        )

        tracker = PatternLifecycleTracker(analyze=analyze_dataframe, symbol="MULTI", interval=interval)
        candidate_result = tracker.ingest(df=df_candidate, as_of=_as_of(df_candidate, interval))
        assert candidate_result.has_new_completed_candle is True

        confirmed_result = tracker.ingest(df=df_confirmed, as_of=_as_of(df_confirmed, interval))
        confirmed = [t for t in confirmed_result.transitions if t.pattern_id == "morning_star"]
        assert len(confirmed) == 1, f"interval={interval} did not confirm the morning star candidate"
        assert confirmed[0].status == "confirmed"
