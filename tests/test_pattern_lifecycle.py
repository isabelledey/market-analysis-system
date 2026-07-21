from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.formatters import format_analysis_text


EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_session(
    session_date: str,
    *,
    length: int = 30,
    start_price: float = 100.0,
    step: float = 0.14,
) -> pd.DataFrame:
    timestamps = pd.date_range(f"{session_date} 09:30", periods=length, freq="15min", tz=EXCHANGE_TZ)
    rows = []
    previous_close = start_price
    for index, timestamp in enumerate(timestamps):
        close = start_price + (index * step)
        open_price = previous_close
        high = max(open_price, close) + 0.22
        low = min(open_price, close) - 0.22
        rows.append(
            {
                "Datetime": timestamp,
                "Open": round(open_price, 4),
                "High": round(high, 4),
                "Low": round(low, 4),
                "Close": round(close, 4),
                "Volume": 1500 + (index * 10),
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def combine_sessions(*sessions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(sessions, ignore_index=True)


def analysis_as_of(df: pd.DataFrame, *, minutes_after_last: int = 16) -> pd.Timestamp:
    return pd.Timestamp(df.iloc[-1]["Datetime"]) + pd.Timedelta(minutes=minutes_after_last)


def set_overlap_pinbar_doji(df: pd.DataFrame, index: int) -> None:
    df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.88, 101.02, 99.90, 100.90, 4000]


def set_bullish_pinbar_only(df: pd.DataFrame, index: int) -> None:
    df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.40, 100.60, 99.50, 100.55, 3600]


def set_doji(df: pd.DataFrame, index: int) -> None:
    df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.00, 101.00, 99.00, 100.02, 2200]


def set_bearish_engulfing(df: pd.DataFrame, bullish_index: int, bearish_index: int) -> None:
    df.loc[bullish_index, ["Open", "High", "Low", "Close", "Volume"]] = [99.80, 101.20, 99.70, 101.00, 2600]
    df.loc[bearish_index, ["Open", "High", "Low", "Close", "Volume"]] = [101.10, 101.30, 99.40, 99.60, 2800]


def make_breakdown_retest_session() -> pd.DataFrame:
    df = make_session("2026-07-20", length=30, start_price=100.0, step=0.0)
    for index in range(25):
        df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.30, 99.80, 100.05, 1200]
    df.loc[25, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.20, 99.00, 99.10, 2600]
    df.loc[26, ["Open", "High", "Low", "Close", "Volume"]] = [99.20, 99.30, 98.80, 98.90, 1400]
    df.loc[27, ["Open", "High", "Low", "Close", "Volume"]] = [98.95, 99.78, 98.70, 99.20, 1500]
    df.loc[28, ["Open", "High", "Low", "Close", "Volume"]] = [99.10, 99.25, 98.60, 98.75, 1300]
    df.loc[29, ["Open", "High", "Low", "Close", "Volume"]] = [98.80, 99.10, 98.35, 98.60, 1250]
    return df


def make_breakdown_pending_session() -> pd.DataFrame:
    df = make_session("2026-07-20", length=28, start_price=100.0, step=0.0)
    for index in range(25):
        df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.30, 99.80, 100.05, 1200]
    df.loc[25, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.20, 99.00, 99.10, 2600]
    df.loc[26, ["Open", "High", "Low", "Close", "Volume"]] = [99.20, 99.30, 98.80, 98.90, 1400]
    df.loc[27, ["Open", "High", "Low", "Close", "Volume"]] = [98.95, 99.95, 98.85, 99.86, 1450]
    return df


def make_breakdown_reclaim_session() -> pd.DataFrame:
    df = make_session("2026-07-20", length=31, start_price=100.0, step=0.0)
    for index in range(25):
        df.loc[index, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.30, 99.80, 100.05, 1200]
    df.loc[25, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.20, 99.00, 99.10, 2600]
    df.loc[26, ["Open", "High", "Low", "Close", "Volume"]] = [99.20, 99.30, 98.80, 98.90, 1400]
    df.loc[27, ["Open", "High", "Low", "Close", "Volume"]] = [98.95, 99.92, 98.85, 99.82, 1450]
    df.loc[28, ["Open", "High", "Low", "Close", "Volume"]] = [99.80, 100.35, 99.70, 100.12, 1550]
    df.loc[29, ["Open", "High", "Low", "Close", "Volume"]] = [100.08, 100.40, 99.95, 100.14, 1600]
    df.loc[30, ["Open", "High", "Low", "Close", "Volume"]] = [100.12, 100.44, 100.00, 100.18, 1650]
    return df


def make_double_bottom_session(*, confirmed: bool) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-20 09:30", periods=13, freq="15min", tz=EXCHANGE_TZ)
    rows = [
        [100.4, 101.0, 99.9, 100.7, 1200],
        [100.8, 101.2, 99.5, 100.0, 1250],
        [100.1, 100.3, 95.0, 95.6, 2300],
        [95.8, 97.6, 95.3, 97.1, 1300],
        [97.2, 99.3, 96.8, 99.0, 1350],
        [99.1, 102.3, 98.9, 101.8, 2100],
        [101.6, 101.8, 99.5, 100.1, 1400],
        [100.0, 100.2, 98.0, 98.6, 1450],
        [98.5, 99.0, 95.1, 95.8, 2200],
        [95.9, 97.2, 95.6, 96.8, 1500],
        [96.8, 98.5, 96.6, 97.8, 1550],
        [97.9, 103.0, 97.7, 102.8 if confirmed else 101.6, 2500],
        [102.9, 103.3, 101.6, 103.0 if confirmed else 101.7, 1600],
    ]
    return pd.DataFrame(
        [
            {
                "Datetime": timestamp,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
            for timestamp, (open_price, high_price, low_price, close_price, volume) in zip(timestamps, rows)
        ]
    )


def make_double_top_with_breakdown_session() -> pd.DataFrame:
    df = make_session("2026-07-20", length=30, start_price=100.0, step=0.0)
    structure_rows = [
        [99.5, 100.0, 99.0, 99.7, 1200],
        [100.0, 101.0, 99.4, 100.6, 1250],
        [100.6, 105.0, 100.1, 104.5, 2200],
        [104.1, 104.3, 101.8, 102.0, 1300],
        [102.0, 102.4, 99.8, 100.3, 1350],
        [100.1, 100.5, 98.4, 98.9, 1400],
        [98.9, 99.2, 97.8, 98.4, 2100],
        [98.5, 100.4, 98.2, 100.0, 1450],
        [100.0, 101.2, 99.6, 100.8, 1500],
        [101.0, 103.6, 100.4, 103.0, 1550],
        [103.1, 104.9, 102.5, 104.2, 2200],
        [104.0, 104.1, 101.8, 102.3, 1600],
        [102.2, 102.4, 100.0, 100.5, 1650],
        [100.3, 100.5, 96.8, 97.2, 2400],
        [98.2, 99.0, 97.4, 97.5, 1700],
    ]
    for offset, row in enumerate(structure_rows, start=15):
        df.loc[offset, ["Open", "High", "Low", "Close", "Volume"]] = row
    return df


def test_latest_completed_pinbar_stays_new_with_no_retest() -> None:
    df = make_session("2026-07-20")
    set_overlap_pinbar_doji(df, len(df) - 1)

    result = analyze_dataframe(df, symbol="LATE", as_of=analysis_as_of(df), top_pattern_count=10)

    canonical = result["current_relevant_patterns"][0]
    assert canonical["state"] == "new"
    assert canonical["retest_at"] is None

    latest_labels = [
        pattern for pattern in result["all_detected_patterns"]
        if pattern["pattern_completion_index"] == len(df) - 1
    ]
    assert {pattern["pattern_name"] for pattern in latest_labels} >= {"Bullish Pin Bar", "Doji"}
    assert all(pattern["event_state"] == "new" for pattern in latest_labels)


def test_incomplete_next_candle_cannot_create_retest() -> None:
    df = make_session("2026-07-20", length=31)
    set_bullish_pinbar_only(df, 29)
    df.loc[30, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.40, 99.80, 100.25, 2100]
    as_of = pd.Timestamp(df.iloc[30]["Datetime"]) + pd.Timedelta(minutes=5)

    result = analyze_dataframe(df, symbol="INCOMP", as_of=as_of, top_pattern_count=10)

    canonical = result["current_relevant_patterns"][0]
    assert canonical["state"] == "new"
    assert canonical["retest_at"] is None


def test_later_completed_pinbar_retest_changes_state() -> None:
    df = make_session("2026-07-20")
    set_bullish_pinbar_only(df, 28)
    df.loc[29, ["Open", "High", "Low", "Close", "Volume"]] = [100.10, 100.45, 99.80, 100.20, 2200]

    result = analyze_dataframe(df, symbol="RETEST", as_of=analysis_as_of(df), top_pattern_count=10)

    canonical = next(
        pattern for pattern in result["current_relevant_patterns"]
        if pattern["primary_pattern_name"] == "Bullish Pin Bar"
    )
    assert canonical["state"] == "retested"
    assert canonical["retest_at"] is not None
    assert canonical["completion_index"] < canonical["last_completed_candle_index"]


def test_breakdown_retest_requires_later_completed_candle() -> None:
    df = make_breakdown_retest_session()

    result = analyze_dataframe(df, symbol="BREAK", as_of=analysis_as_of(df), top_pattern_count=10)

    breakdown = next(
        pattern for pattern in result["current_relevant_patterns"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    assert breakdown["state"] == "retest_rejected"
    assert breakdown["retest_at"] is not None
    assert breakdown["included_in_current_score"] is True


def test_breakdown_touch_inside_zone_stays_retest_pending() -> None:
    df = make_breakdown_pending_session()

    result = analyze_dataframe(df, symbol="PENDING", as_of=analysis_as_of(df), top_pattern_count=10)

    breakdown = next(
        pattern for pattern in result["current_relevant_patterns"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    assert breakdown["state"] == "retest_pending"
    assert breakdown["retest_at"] is not None
    assert breakdown["current_score_exclusion_reason"] is None


def test_breakdown_reclaim_and_followthrough_become_failed_breakdown() -> None:
    df = make_breakdown_reclaim_session()

    reclaim_result = analyze_dataframe(
        df.iloc[:29].copy(),
        symbol="RECLAIM",
        as_of=analysis_as_of(df.iloc[:29].copy()),
        top_pattern_count=10,
    )
    reclaim_breakdown = next(
        pattern for pattern in reclaim_result["current_relevant_patterns"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    assert reclaim_breakdown["state"] == "reclaimed"
    assert reclaim_breakdown["included_in_current_score"] is False
    assert reclaim_breakdown["current_score_exclusion_reason"] == "level reclaimed"

    failed_result = analyze_dataframe(df, symbol="FAILED", as_of=analysis_as_of(df), top_pattern_count=10)
    failed_breakdown = next(
        pattern for pattern in failed_result["session_pattern_history"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    assert failed_breakdown["state"] == "failed_breakdown"
    assert failed_breakdown["included_in_current_score"] is False
    assert failed_breakdown["current_score_exclusion_reason"] == "failed breakdown"


def test_family_without_retest_semantics_does_not_become_retested() -> None:
    df = make_session("2026-07-20")
    set_doji(df, 28)
    df.loc[29, ["Open", "High", "Low", "Close", "Volume"]] = [100.01, 100.80, 99.10, 100.10, 2300]

    result = analyze_dataframe(df, symbol="DOJI", as_of=analysis_as_of(df), top_pattern_count=10)

    doji_events = [pattern for pattern in result["all_detected_patterns"] if pattern["pattern_name"] == "Doji"]
    assert doji_events
    assert all(pattern["event_state"] != "retested" for pattern in doji_events)


def test_overlapping_pinbar_and_doji_form_one_canonical_event() -> None:
    df = make_session("2026-07-20")
    set_overlap_pinbar_doji(df, len(df) - 1)

    result = analyze_dataframe(df, symbol="OVER", as_of=analysis_as_of(df), top_pattern_count=10)

    assert len(result["current_relevant_patterns"]) == 1
    canonical = result["current_relevant_patterns"][0]
    assert canonical["pattern_labels"] == ["Bullish Pin Bar", "Doji"]
    assert canonical["label_count"] == 2
    assert canonical["overlap_label_count"] == 1
    assert canonical["state"] == "new"
    assert "grouped into 1 candlestick event" in canonical["overlap_note"]


def test_bullish_pinbar_hammer_and_doji_share_one_canonical_candle_event() -> None:
    df = make_session("2026-07-20", start_price=101.5, step=-0.18)
    df.loc[len(df) - 1, ["Open", "High", "Low", "Close", "Volume"]] = [96.00, 96.05, 94.50, 96.02, 4200]

    result = analyze_dataframe(df, symbol="HAMMER", as_of=analysis_as_of(df), top_pattern_count=10)

    canonical = result["current_relevant_patterns"][0]
    assert canonical["primary_pattern_name"] == "Bullish Pin Bar"
    assert canonical["pattern_labels"] == ["Bullish Pin Bar", "Doji", "Hammer"]
    assert canonical["included_in_current_score"] is True


def test_early_session_bearish_engulfing_remains_visible_in_session_history() -> None:
    df = make_session("2026-07-20")
    set_bearish_engulfing(df, 3, 4)
    set_bullish_pinbar_only(df, 29)

    result = analyze_dataframe(df, symbol="HIST", as_of=analysis_as_of(df), top_pattern_count=3)

    history_item = next(
        item for item in result["session_pattern_history"]
        if item["primary_pattern_name"] == "Bearish Engulfing"
    )
    assert history_item["included_in_current_score"] is False
    assert history_item["current_score_exclusion_reason"] in {"expired", "invalidated"}
    assert history_item["detected_at_display"].endswith("Asia/Jerusalem")


def test_multiple_session_events_are_ordered_and_detection_is_independent_of_summary_limit() -> None:
    df = make_session("2026-07-20")
    set_doji(df, 5)
    set_doji(df, 10)
    set_doji(df, 15)
    set_bullish_pinbar_only(df, 29)

    result_top_1 = analyze_dataframe(df, symbol="COUNT1", as_of=analysis_as_of(df), top_pattern_count=1)
    result_top_5 = analyze_dataframe(df, symbol="COUNT5", as_of=analysis_as_of(df), top_pattern_count=5)

    assert result_top_1["session_history_total"] == result_top_5["session_history_total"]
    doji_history = [
        item for item in result_top_5["session_pattern_history"]
        if item["primary_pattern_name"] == "Doji"
    ]
    assert len(doji_history) >= 3
    detected_times = [item["detected_at"] for item in result_top_5["session_pattern_history"]]
    assert detected_times == sorted(detected_times)


def test_current_relevant_patterns_are_sorted_by_latest_transition_time() -> None:
    df = make_session("2026-07-20")
    set_doji(df, 27)
    set_overlap_pinbar_doji(df, 29)

    result = analyze_dataframe(df, symbol="ORDER", as_of=analysis_as_of(df), top_pattern_count=10)

    timestamps = [
        pattern["state_updated_at"] or pattern["detected_at"]
        for pattern in result["current_relevant_patterns"]
    ]
    assert timestamps == sorted(timestamps, reverse=True)

    leading_labels = ", ".join(result["current_relevant_patterns"][0]["pattern_labels"])
    assert leading_labels in result["structured_explanation"]["lifecycle_note"]


def test_previous_session_warmup_data_is_not_listed_in_current_session_history() -> None:
    previous_session = make_session("2026-07-17")
    set_bearish_engulfing(previous_session, 3, 4)
    current_session = make_session("2026-07-20")
    set_bullish_pinbar_only(current_session, 29)
    df = combine_sessions(previous_session, current_session)

    result = analyze_dataframe(df, symbol="WARM", as_of=analysis_as_of(df), top_pattern_count=10)

    assert result["relevant_session"]["exchange_date"] == "2026-07-20"
    assert all(item["session_date"] == "2026-07-20" for item in result["session_pattern_history"])


def test_before_market_and_weekend_use_most_recent_completed_session() -> None:
    friday_session = make_session("2026-07-17")
    set_bullish_pinbar_only(friday_session, 29)

    before_market_result = analyze_dataframe(
        friday_session,
        symbol="PRE",
        as_of=pd.Timestamp("2026-07-20 08:00", tz=EXCHANGE_TZ),
        top_pattern_count=10,
    )
    weekend_result = analyze_dataframe(
        friday_session,
        symbol="WKND",
        as_of=pd.Timestamp("2026-07-19 12:00", tz=EXCHANGE_TZ),
        top_pattern_count=10,
    )

    assert before_market_result["relevant_session"]["exchange_date"] == "2026-07-17"
    assert weekend_result["relevant_session"]["exchange_date"] == "2026-07-17"


def test_premarket_bars_do_not_replace_latest_regular_session_by_default() -> None:
    friday_session = make_session("2026-07-17")
    monday_premarket = pd.DataFrame(
        [
            {
                "Datetime": pd.Timestamp("2026-07-20 08:00", tz=EXCHANGE_TZ),
                "Open": 100.0,
                "High": 100.3,
                "Low": 99.8,
                "Close": 100.1,
                "Volume": 800,
            },
            {
                "Datetime": pd.Timestamp("2026-07-20 08:15", tz=EXCHANGE_TZ),
                "Open": 100.1,
                "High": 100.4,
                "Low": 99.9,
                "Close": 100.2,
                "Volume": 820,
            },
        ]
    )
    combined = pd.concat([friday_session, monday_premarket], ignore_index=True)

    result = analyze_dataframe(
        combined,
        symbol="PREMKT",
        as_of=pd.Timestamp("2026-07-20 08:31", tz=EXCHANGE_TZ),
        top_pattern_count=10,
        include_extended_hours=False,
    )

    assert result["relevant_session"]["exchange_date"] == "2026-07-17"
    assert result["relevant_session"]["previous_exchange_date"] is None


def test_session_history_is_rendered_in_display_timezone() -> None:
    df = make_session("2026-07-20")
    set_bearish_engulfing(df, 3, 4)
    set_overlap_pinbar_doji(df, 29)

    result = analyze_dataframe(df, symbol="TEXT", as_of=analysis_as_of(df), top_pattern_count=10)
    report = format_analysis_text(result)

    assert "Current Active Evidence (" in report
    assert "Historical Session Detections (" in report
    assert "Asia/Jerusalem" in report
    assert "grouped into 1 candlestick event" in report


def test_invalidation_conditions_are_exposed_in_results_and_text() -> None:
    df = make_breakdown_retest_session()

    result = analyze_dataframe(df, symbol="INVAL", as_of=analysis_as_of(df), top_pattern_count=10)
    breakdown = next(
        pattern for pattern in result["current_relevant_patterns"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    serialized_breakdown = next(
        pattern for pattern in result["all_detected_patterns"]
        if "Breakdown" in pattern["pattern_name"]
    )
    report = format_analysis_text(result)

    assert "completed close above" in breakdown["invalidation_condition"].lower()
    assert "completed close above" in serialized_breakdown["invalidation_condition"].lower()
    assert "Invalidation Condition:" in report


def test_reclaimed_and_failed_transition_times_are_exposed_in_results_and_text() -> None:
    df = make_breakdown_reclaim_session()

    reclaim_result = analyze_dataframe(
        df.iloc[:29].copy(),
        symbol="RECLTXT",
        as_of=analysis_as_of(df.iloc[:29].copy()),
        top_pattern_count=10,
    )
    reclaim_breakdown = next(
        pattern for pattern in reclaim_result["current_relevant_patterns"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    reclaim_report = format_analysis_text(reclaim_result)

    assert reclaim_breakdown["reclaimed_at_display"] is not None
    assert reclaim_breakdown["state_updated_at_display"] == reclaim_breakdown["reclaimed_at_display"]
    assert "Reclaimed Time:" in reclaim_report

    failed_result = analyze_dataframe(df, symbol="FAILTXT", as_of=analysis_as_of(df), top_pattern_count=10)
    failed_breakdown = next(
        pattern for pattern in failed_result["session_pattern_history"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    failed_report = format_analysis_text(failed_result)

    assert failed_breakdown["failed_at_display"] is not None
    assert failed_breakdown["state_updated_at_display"] == failed_breakdown["failed_at_display"]
    assert "Failed Time:" in failed_report


def test_double_bottom_awaiting_confirmation_uses_confirmation_reason_not_horizon() -> None:
    df = make_double_bottom_session(confirmed=False)

    result = analyze_dataframe(df, symbol="DBOT", as_of=analysis_as_of(df), top_pattern_count=10)

    double_bottom = next(
        pattern for pattern in result["current_relevant_patterns"]
        if pattern["primary_pattern_name"] == "Double Bottom"
    )
    assert double_bottom["state"] == "awaiting_confirmation"
    assert double_bottom["included_in_current_score"] is False
    assert double_bottom["current_score_exclusion_reason"] == "awaiting neckline confirmation"


def test_double_bottom_awaiting_confirmation_expires_after_structural_window() -> None:
    df = make_double_bottom_session(confirmed=False)
    last_timestamp = pd.Timestamp(df.iloc[-1]["Datetime"])
    trailing_rows = []
    for offset in range(1, 9):
        timestamp = last_timestamp + pd.Timedelta(minutes=15 * offset)
        trailing_rows.append(
            {
                "Datetime": timestamp,
                "Open": 101.1,
                "High": 101.4,
                "Low": 100.8,
                "Close": 101.2,
                "Volume": 1400 + offset,
            }
        )
    extended = pd.concat([df, pd.DataFrame(trailing_rows)], ignore_index=True)

    result = analyze_dataframe(extended, symbol="DBOTX", as_of=analysis_as_of(extended), top_pattern_count=10)

    double_bottom = next(
        pattern for pattern in result["session_pattern_history"]
        if pattern["primary_pattern_name"] == "Double Bottom"
    )
    assert double_bottom["state"] == "expired"
    assert double_bottom["included_in_current_score"] is False
    assert double_bottom["current_score_exclusion_reason"] == "expired"


def test_confirmed_double_bottom_keeps_confirmation_timestamp() -> None:
    df = make_double_bottom_session(confirmed=True)

    result = analyze_dataframe(df, symbol="DBOTC", as_of=analysis_as_of(df), top_pattern_count=10)

    double_bottom = next(
        pattern for pattern in result["session_pattern_history"]
        if pattern["primary_pattern_name"] == "Double Bottom"
    )
    assert double_bottom["confirmation_at"] is not None
    assert double_bottom["confirmation_at"] >= double_bottom["setup_completion"]


def test_structural_pattern_and_breakdown_stay_separate_but_related() -> None:
    df = make_double_top_with_breakdown_session()

    result = analyze_dataframe(df, symbol="REL", as_of=analysis_as_of(df), top_pattern_count=10)

    double_top = next(
        pattern for pattern in result["session_pattern_history"]
        if pattern["primary_pattern_name"] == "Double Top"
    )
    breakdown = next(
        pattern for pattern in result["session_pattern_history"]
        if "Breakdown" in pattern["primary_pattern_name"]
    )
    assert double_top["event_id"] != breakdown["event_id"]
    assert double_top["relationship_type"] == "confirmed_by"
    assert breakdown["relationship_type"] == "confirms"
    assert double_top["overlap_note"] is None
    assert breakdown["overlap_note"] is None
