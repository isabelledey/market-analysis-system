"""Tests for Stage 6: deduplicating near-duplicate structural (Double Top/Bottom) events.

Root cause: both `DoubleTopDetector` and `DoubleBottomDetector` pair every qualifying earlier
swing with the nearest valid later swing, without marking the later swing as "claimed" -- so when
more than one earlier swing qualifies against the same later swing and peak, the detector emits
multiple near-identical events for what is really one underlying setup. `deduplicate_structural_
events` runs immediately after the existing exact-duplicate `resolve_pattern_conflicts`, before
events are prepared, tracked, or scored, so no downstream layer (or the UI) ever sees the
duplicate -- and distinct setups are never incorrectly merged.
"""

from __future__ import annotations

import random
from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.config import PatternConfig
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.features import add_features
from stock_pattern_model.pattern_detector import DoubleBottomDetector
from stock_pattern_model.pattern_detector import deduplicate_structural_events
from stock_pattern_model.pattern_detector import resolve_pattern_conflicts

EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_double_bottom_event(
    *,
    pattern_start_at: str,
    setup_completion_at: str = "2026-07-10 12:30",
    status: PatternStatus = PatternStatus.TENTATIVE,
    first_bottom: float = 95.0,
    second_bottom: float = 95.1,
    neckline: float = 102.3,
    signal_strength: float = 3.5,
    relevant_indices: list[int] | None = None,
    detection_reason: str = "Two confirmed swing lows matched within tolerance.",
) -> PatternEvent:
    start = pd.Timestamp(pattern_start_at, tz=EXCHANGE_TZ)
    completion = pd.Timestamp(setup_completion_at, tz=EXCHANGE_TZ)
    return PatternEvent(
        pattern_id="double_bottom",
        pattern_name="Double Bottom",
        pattern_family=PatternFamily.DOUBLE_BOTTOM,
        bias="Bullish",
        status=status,
        pattern_start_at=start,
        pattern_end_at=completion,
        bar_start_at=completion,
        bar_end_at=completion,
        detected_at=completion,
        setup_completion_at=completion,
        relevant_prices={
            "first_bottom": first_bottom,
            "second_bottom": second_bottom,
            "neckline": neckline,
            "confirmation_price": neckline,
        },
        relevant_indices=relevant_indices or [0, 1, 2],
        detection_reason=detection_reason,
        signal_strength=signal_strength,
        base_score=20.0,
        exchange_timezone="America/New_York",
    )


def test_exact_duplicate_events_are_merged_into_one() -> None:
    original = make_double_bottom_event(pattern_start_at="2026-07-10 10:00")
    exact_duplicate = make_double_bottom_event(pattern_start_at="2026-07-10 10:00")

    deduplicated, removed_count = deduplicate_structural_events([original, exact_duplicate])

    assert len(deduplicated) == 1
    assert removed_count == 1


def test_equivalent_events_with_tiny_float_differences_are_merged() -> None:
    original = make_double_bottom_event(pattern_start_at="2026-07-10 10:00", first_bottom=95.000001, neckline=102.300004)
    near_duplicate = make_double_bottom_event(pattern_start_at="2026-07-10 10:45", first_bottom=95.05, neckline=102.30001)

    deduplicated, removed_count = deduplicate_structural_events([original, near_duplicate])

    assert len(deduplicated) == 1
    assert removed_count == 1
    # Merged provenance is preserved for auditability rather than silently discarded.
    assert "merged" in deduplicated[0].detection_reason.lower()
    assert set(deduplicated[0].relevant_indices) >= set(original.relevant_indices) | set(near_duplicate.relevant_indices)


def test_same_pattern_from_different_sources_is_merged() -> None:
    """Two events sharing the same canonical setup identity but produced with different
    `detector_version`/provenance metadata (simulating two detectors intentionally reporting the
    same underlying event) merge into one rather than staying duplicated.
    """
    from_detector_a = make_double_bottom_event(
        pattern_start_at="2026-07-10 10:00",
        detection_reason="Detected by rule-based swing matcher.",
    )
    from dataclasses import replace

    from_detector_b = replace(
        from_detector_a,
        detection_reason="Detected by an alternate structural scanner.",
        detector_version="v2",
        signal_strength=from_detector_a.signal_strength - 0.01,
    )

    deduplicated, removed_count = deduplicate_structural_events([from_detector_a, from_detector_b])

    assert len(deduplicated) == 1
    assert removed_count == 1


def test_distinct_double_bottoms_at_different_times_are_not_merged() -> None:
    earlier_setup = make_double_bottom_event(
        pattern_start_at="2026-07-10 10:00",
        setup_completion_at="2026-07-10 12:30",
        first_bottom=95.0,
        second_bottom=95.1,
        neckline=102.3,
    )
    later_setup = make_double_bottom_event(
        pattern_start_at="2026-07-11 10:00",
        setup_completion_at="2026-07-11 14:00",
        first_bottom=110.0,
        second_bottom=110.2,
        neckline=118.5,
    )

    deduplicated, removed_count = deduplicate_structural_events([earlier_setup, later_setup])

    assert len(deduplicated) == 2
    assert removed_count == 0


def test_different_lifecycle_stages_of_the_same_setup_are_not_merged() -> None:
    """TENTATIVE and CONFIRMED events for the same setup are deliberately different moments in
    the same setup's lifecycle and must never be collapsed into one.
    """
    tentative = make_double_bottom_event(pattern_start_at="2026-07-10 10:00", status=PatternStatus.TENTATIVE)
    confirmed = make_double_bottom_event(pattern_start_at="2026-07-10 10:00", status=PatternStatus.CONFIRMED)

    deduplicated, removed_count = deduplicate_structural_events([tentative, confirmed])

    assert len(deduplicated) == 2
    assert removed_count == 0


def test_deduplication_is_idempotent() -> None:
    events = [
        make_double_bottom_event(pattern_start_at="2026-07-10 10:00"),
        make_double_bottom_event(pattern_start_at="2026-07-10 10:45", first_bottom=95.05),
        make_double_bottom_event(
            pattern_start_at="2026-07-11 10:00",
            setup_completion_at="2026-07-11 14:00",
            first_bottom=110.0,
            second_bottom=110.2,
            neckline=118.5,
        ),
    ]

    once, removed_once = deduplicate_structural_events(events)
    twice, removed_twice = deduplicate_structural_events(once)

    assert len(once) == len(twice) == 2
    assert removed_once == 1
    assert removed_twice == 0
    assert [event.pattern_start_at for event in once] == [event.pattern_start_at for event in twice]


def test_deduplication_result_does_not_depend_on_input_order() -> None:
    events = [
        make_double_bottom_event(pattern_start_at="2026-07-10 10:00"),
        make_double_bottom_event(pattern_start_at="2026-07-10 10:45", first_bottom=95.05),
        make_double_bottom_event(
            pattern_start_at="2026-07-11 10:00",
            setup_completion_at="2026-07-11 14:00",
            first_bottom=110.0,
            second_bottom=110.2,
            neckline=118.5,
        ),
    ]
    shuffled = list(events)
    random.Random(7).shuffle(shuffled)

    forward_result, forward_removed = deduplicate_structural_events(events)
    shuffled_result, shuffled_removed = deduplicate_structural_events(shuffled)

    assert forward_removed == shuffled_removed
    assert [event.pattern_start_at for event in forward_result] == [
        event.pattern_start_at for event in shuffled_result
    ]


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


def make_double_bottom_with_two_qualifying_earlier_swings_df() -> pd.DataFrame:
    """Two separate earlier swing lows (index 2 and index 5) both qualify against the same later
    swing low (index 11) and the same intervening peak (index 8) -- the exact real-world detector
    bug that produced duplicate Double Bottom events.
    """
    rows = [
        _candle(100.4, 101.0, 99.9, 100.7),
        _candle(100.8, 101.2, 99.5, 100.0),
        _candle(100.1, 100.3, 95.0, 95.6, 2300),
        _candle(95.8, 97.6, 95.5, 97.1),
        _candle(97.2, 98.3, 97.0, 98.0),
        _candle(98.0, 98.3, 95.05, 95.7, 2200),
        _candle(95.9, 97.6, 95.6, 97.2),
        _candle(97.3, 99.5, 97.0, 99.2),
        _candle(99.3, 102.3, 99.0, 101.8, 2100),
        _candle(101.6, 101.8, 99.5, 100.1),
        _candle(100.0, 100.2, 98.0, 98.6),
        _candle(98.5, 99.0, 95.1, 95.8, 2200),
        _candle(95.9, 97.2, 95.6, 96.8),
        _candle(96.8, 98.5, 96.6, 97.8),
        _candle(97.9, 103.0, 97.7, 101.6, 2500),
        _candle(102.9, 103.3, 102.0, 102.1),
    ]
    return make_df(rows)


def test_detector_produces_the_real_bug_and_dedup_fixes_it_end_to_end() -> None:
    df = make_double_bottom_with_two_qualifying_earlier_swings_df()
    feature_df = add_features(df)

    raw_events = DoubleBottomDetector().detect(feature_df, PatternConfig(), "15m")
    # Confirms the actual detector-level bug this stage fixes: two earlier swings both pairing
    # with the same later swing/peak produce two near-identical raw events.
    assert len(raw_events) == 2
    assert raw_events[0].setup_completion_at == raw_events[1].setup_completion_at
    assert raw_events[0].relevant_prices["neckline"] == raw_events[1].relevant_prices["neckline"]

    resolved_events, _ = resolve_pattern_conflicts(raw_events)
    # The existing exact-key dedup does not catch this -- it is a genuine, correct fix target.
    assert len(resolved_events) == 2

    deduplicated, removed_count = deduplicate_structural_events(resolved_events)
    assert len(deduplicated) == 1
    assert removed_count == 1


def test_full_analysis_never_reports_the_duplicate_double_bottom() -> None:
    from stock_pattern_model.analysis import analyze_dataframe

    df = make_double_bottom_with_two_qualifying_earlier_swings_df()
    as_of = pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=16)

    result = analyze_dataframe(df=df, symbol="DBDEDUP", as_of=as_of, top_pattern_count=10)

    double_bottoms = [p for p in result["all_detected_patterns"] if p["pattern_name"] == "Double Bottom"]
    assert len(double_bottoms) == 1
