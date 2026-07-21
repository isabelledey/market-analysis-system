"""Core analysis entry points for stock pattern detection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from typing import Optional

import pandas as pd

from stock_pattern_model.config import AnalysisConfig
from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.config import PatternConfig
from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.context import AnalysisContext
from stock_pattern_model.context import build_analysis_context
from stock_pattern_model.context import dataframe_identity
from stock_pattern_model.datetime_utils import convert_to_timezone
from stock_pattern_model.datetime_utils import format_display_datetime
from stock_pattern_model.datetime_utils import format_iso_timestamp
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.exceptions import NoCompletedBarsError
from stock_pattern_model.features import add_features
from stock_pattern_model.market_data import FileDataProvider
from stock_pattern_model.market_data import MarketDataProvider
from stock_pattern_model.market_data import YFinanceProvider
from stock_pattern_model.market_data import validate_market_data
from stock_pattern_model.pattern_detector import DEFAULT_PATTERN_REGISTRY
from stock_pattern_model.pattern_detector import PatternRegistry
from stock_pattern_model.pattern_detector import classify_intraday_trend
from stock_pattern_model.pattern_detector import detect_patterns
from stock_pattern_model.pattern_detector import resolve_pattern_conflicts
from stock_pattern_model.scoring import build_evidence_group
from stock_pattern_model.scoring import build_event_id
from stock_pattern_model.scoring import build_setup_id
from stock_pattern_model.scoring import pattern_max_age_bars
from stock_pattern_model.scoring import ScoringService
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_END
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_START
from stock_pattern_model.session_utils import DEFAULT_SESSION_MODE
from stock_pattern_model.session_utils import session_date_series


def _get_recency_weight(candles_ago: int) -> float:
    """Legacy display weighting retained for ranking transparency."""
    if candles_ago == 0:
        return 1.0
    if 1 <= candles_ago <= 3:
        return 0.85
    if 4 <= candles_ago <= 6:
        return 0.65
    return 0.40


def _get_bar_timedelta(interval: str) -> pd.Timedelta:
    return pd.to_timedelta(interval)


def _get_bar_end(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    return timestamp + _get_bar_timedelta(interval)


def _normalize_as_of(as_of: Optional[pd.Timestamp]) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz="UTC")

    normalized = pd.Timestamp(as_of)
    if normalized.tzinfo is None:
        raise DataValidationError("as_of must be timezone-aware.")
    return normalized


def _get_exchange_timezone(df: pd.DataFrame) -> str:
    if "Datetime" not in df.columns:
        raise DataValidationError("Input DataFrame must contain a Datetime column.")

    datetime_series = pd.to_datetime(df["Datetime"])
    timezone = datetime_series.dt.tz
    if timezone is None:
        raise DataValidationError("Datetime values must be timezone-aware for intraday analysis.")
    return str(timezone)


def _update_completed_row_count(
    report: DataQualityReport,
    completed_row_count: int,
) -> DataQualityReport:
    return DataQualityReport(
        row_count=report.row_count,
        completed_row_count=completed_row_count,
        duplicate_count=report.duplicate_count,
        missing_value_count=report.missing_value_count,
        invalid_ohlc_count=report.invalid_ohlc_count,
        irregular_gap_count=report.irregular_gap_count,
        warnings=list(report.warnings),
        cleaning_actions=list(report.cleaning_actions),
    )


def _filter_completed_candles(
    df: pd.DataFrame,
    interval: str,
    as_of: Optional[pd.Timestamp],
    quality_report: DataQualityReport,
) -> tuple[pd.DataFrame, pd.Timestamp, DataQualityReport]:
    normalized_as_of = _normalize_as_of(as_of)
    filtered_df = df.copy()
    filtered_df["Datetime"] = pd.to_datetime(filtered_df["Datetime"])
    filtered_df["Bar_End"] = filtered_df["Datetime"] + _get_bar_timedelta(interval)
    filtered_df = filtered_df.loc[filtered_df["Bar_End"] <= normalized_as_of].copy()
    filtered_df = filtered_df.drop(columns=["Bar_End"]).reset_index(drop=True)

    if filtered_df.empty:
        raise NoCompletedBarsError(
            f"No completed {interval} candles are available for analysis as of "
            f"{normalized_as_of.isoformat(timespec='minutes')}."
        )

    updated_report = _update_completed_row_count(quality_report, len(filtered_df))
    return filtered_df, normalized_as_of, updated_report


def _run_pattern_pipeline(
    df: pd.DataFrame,
    config: AnalysisConfig,
    registry: PatternRegistry,
    *,
    exchange_timezone: str | None,
    regular_session_start: str,
    regular_session_end: str,
) -> tuple[pd.DataFrame, list[PatternEvent]]:
    pattern_df = classify_intraday_trend(
        add_features(
            df,
            exchange_timezone=exchange_timezone,
            regular_session_start=regular_session_start,
            regular_session_end=regular_session_end,
        ),
        lookback_bars=config.scoring.lookback_bars,
        pivot_left_bars=config.pattern.pivot_left_bars,
        pivot_right_bars=config.pattern.pivot_right_bars,
        breakout_lookback=config.pattern.breakout_lookback,
    )
    pattern_events = detect_patterns(pattern_df, config.pattern, config.interval, registry=registry)
    return pattern_df, pattern_events


def _detected_bar_start(pattern: PatternEvent, interval: str) -> pd.Timestamp:
    return pattern.detected_at - _get_bar_timedelta(interval)


def _prepare_pattern_records(
    df: pd.DataFrame,
    events: list[PatternEvent],
    interval: str,
    registry: PatternRegistry,
    score_tentative_patterns: bool,
) -> list[dict[str, Any]]:
    details = registry.details()
    latest_index = len(df) - 1
    index_lookup = {
        pd.Timestamp(row["Datetime"]).isoformat(): int(index)
        for index, row in df.iterrows()
    }
    prepared: list[dict[str, Any]] = []

    for event in events:
        detected_bar_key = _detected_bar_start(event, interval).isoformat()
        final_bar_key = event.bar_start_at.isoformat()
        setup_completion_key = (
            event.setup_completion_at - _get_bar_timedelta(interval)
            if event.setup_completion_at is not None
            else event.pattern_end_at - _get_bar_timedelta(interval)
        ).isoformat()
        confirmation_key = (
            (event.confirmation_at - _get_bar_timedelta(interval)).isoformat()
            if event.confirmation_at is not None
            else None
        )
        detected_index = index_lookup.get(detected_bar_key)
        final_index = index_lookup.get(final_bar_key)
        setup_completion_index = index_lookup.get(setup_completion_key)
        confirmation_index = index_lookup.get(confirmation_key) if confirmation_key is not None else None
        if detected_index is None or final_index is None:
            continue

        metadata = details.get(event.pattern_id, {})
        detected_row = df.iloc[detected_index]
        candles_ago = latest_index - detected_index
        recency_weight = _get_recency_weight(candles_ago)
        weighted_score = round(float(event.base_score) * recency_weight, 2)
        strong_signal = event.strength_label == "strong"
        prepared.append(
            {
                "event": event,
                "pattern_id": event.pattern_id,
                "pattern_name": event.pattern_name,
                "bias": event.bias,
                "status": event.status.value,
                "pattern_family": event.pattern_family.value,
                "priority": int(metadata.get("priority", 99)),
                "base_score": float(event.base_score),
                "weighted_score": weighted_score,
                "candles_ago": candles_ago,
                "detection_reason": event.detection_reason,
                "exchange_timezone": event.exchange_timezone,
                "volume_confirmed": bool(detected_row.get("Volume_Strength", 0) >= 1.0),
                "strong_signal": strong_signal,
                "signal_strength": float(event.signal_strength),
                "strength_label": event.strength_label,
                "volume_baseline_source": event.volume_baseline_source,
                "pattern_start_index": int(event.pattern_start_index if event.pattern_start_index is not None else min(event.relevant_indices)),
                "pattern_completion_index": int(event.pattern_completion_index if event.pattern_completion_index is not None else final_index),
                "detected_index": int(event.detected_index if event.detected_index is not None else detected_index),
                "setup_completion_index": int(setup_completion_index if setup_completion_index is not None else final_index),
                "confirmation_index": int(confirmation_index) if confirmation_index is not None else None,
                "score_ineligibility_reason": (
                    "awaiting neckline confirmation"
                    if event.status is PatternStatus.TENTATIVE and event.pattern_id in {"double_top", "double_bottom"}
                    else (
                        "unconfirmed structural pattern"
                        if event.status is PatternStatus.TENTATIVE
                        else (
                            "failed pattern"
                            if event.status is PatternStatus.FAILED
                            else (
                                "expired"
                                if event.status is PatternStatus.EXPIRED
                                else None
                            )
                        )
                    )
                ),
                "score_eligible": event.status is PatternStatus.CONFIRMED
                or (
                    score_tentative_patterns
                    and event.status is PatternStatus.TENTATIVE
                ),
            }
        )

    return prepared


def _annotate_pattern_identity(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for pattern in patterns:
        item = dict(pattern)
        item["event_id"] = build_event_id(item)
        item["setup_id"] = build_setup_id(item)
        item["evidence_group"] = build_evidence_group(item)
        annotated.append(item)
    return annotated


def _latest_completed_session_info(
    df: pd.DataFrame,
    interval: str,
    *,
    context: AnalysisContext,
) -> dict[str, Any]:
    exchange_timezone = context.exchange_timezone or _get_exchange_timezone(df)
    session_dates = session_date_series(df["Datetime"], exchange_timezone)
    if "Session_Segment" in df.columns:
        segment_mask = df["Session_Segment"].isin(context.included_segments)
    else:
        segment_mask = pd.Series([True] * len(df), index=df.index)
    scoped_session_dates = session_dates.loc[segment_mask]
    relevant_source = scoped_session_dates if not scoped_session_dates.empty else session_dates
    relevant_session_date = str(relevant_source.iloc[-1])
    unique_session_dates = list(dict.fromkeys(relevant_source.tolist()))
    previous_session_date = unique_session_dates[-2] if len(unique_session_dates) > 1 else None
    session_mask = session_dates == relevant_session_date
    session_df = df.loc[session_mask].copy().reset_index(drop=True)
    session_start = pd.Timestamp(session_df.iloc[0]["Datetime"])
    session_end = _get_bar_end(pd.Timestamp(session_df.iloc[-1]["Datetime"]), interval)
    return {
        "exchange_timezone": exchange_timezone,
        "session_date": relevant_session_date,
        "previous_session_date": previous_session_date,
        "session_start": session_start,
        "session_end": session_end,
        "session_row_count": len(session_df),
        "session_index_start": int(df.index[session_mask][0]),
        "session_index_end": int(df.index[session_mask][-1]),
        "session_mode": context.session_mode,
        "included_segments": list(context.included_segments),
    }


def _exchange_session_date(timestamp: pd.Timestamp, exchange_timezone: str) -> str:
    return pd.Timestamp(timestamp).tz_convert(exchange_timezone).date().isoformat()


def _bar_end_for_index(df: pd.DataFrame, index: int, interval: str) -> pd.Timestamp:
    return _get_bar_end(pd.Timestamp(df.iloc[index]["Datetime"]), interval)


def _completion_reference_index(pattern: dict[str, Any]) -> int:
    return int(pattern["pattern_completion_index"])


def _transition_timestamp(df: pd.DataFrame, index: int | None, interval: str) -> pd.Timestamp | None:
    if index is None:
        return None
    return _bar_end_for_index(df, index, interval)


def _price_tolerance(
    df: pd.DataFrame,
    index: int,
    reference_level: float,
    config: PatternConfig,
) -> float:
    row = df.iloc[index]
    avg_range = row.get("Avg_Range_20_Bars")
    candle_range = float(row["High"]) - float(row["Low"])
    baseline = float(avg_range) if pd.notna(avg_range) and float(avg_range) > 0 else candle_range
    return max(
        baseline * config.atr_tolerance_multiplier,
        abs(reference_level) * config.percentage_tolerance,
        0.01,
    )


def _family_supports_retest(pattern: dict[str, Any]) -> bool:
    return pattern["pattern_id"] in {"breakout", "breakdown", "bullish_pin_bar", "shooting_star"}


def _pattern_extreme(df: pd.DataFrame, pattern: dict[str, Any], direction: str) -> float:
    indices = [int(index) for index in pattern["event"].relevant_indices]
    if direction == "low":
        return float(df.iloc[indices]["Low"].min())
    return float(df.iloc[indices]["High"].max())


def _is_retest_candle(
    pattern: dict[str, Any],
    row: pd.Series,
    df: pd.DataFrame,
    config: PatternConfig,
) -> bool:
    relevant_prices = pattern["event"].relevant_prices
    completion_index = _completion_reference_index(pattern)
    if pattern["pattern_id"] == "breakout":
        level = float(relevant_prices.get("breakout_level") or relevant_prices.get("confirmation_price") or row["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return bool(float(row["Low"]) <= level + tolerance and float(row["Close"]) >= level)
    if pattern["pattern_id"] == "breakdown":
        level = float(relevant_prices.get("breakdown_level") or relevant_prices.get("confirmation_price") or row["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return bool(float(row["High"]) >= level - tolerance and float(row["Close"]) <= level)
    if pattern["pattern_id"] == "bullish_pin_bar":
        low = float(relevant_prices["low"])
        high = float(relevant_prices["high"])
        zone_high = low + ((high - low) * 0.35)
        return bool(float(row["Low"]) <= zone_high and float(row["Close"]) > zone_high)
    if pattern["pattern_id"] == "shooting_star":
        high = float(relevant_prices["high"])
        low = float(relevant_prices["low"])
        zone_low = high - ((high - low) * 0.35)
        return bool(float(row["High"]) >= zone_low and float(row["Close"]) < zone_low)
    return False


def _is_invalidated_candle(
    pattern: dict[str, Any],
    row: pd.Series,
    df: pd.DataFrame,
    config: PatternConfig,
) -> bool:
    relevant_prices = pattern["event"].relevant_prices
    completion_index = _completion_reference_index(pattern)
    pattern_id = pattern["pattern_id"]
    bias = pattern["bias"]

    if pattern_id == "breakout":
        level = float(relevant_prices.get("breakout_level") or relevant_prices.get("confirmation_price") or row["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return bool(float(row["Close"]) < level - tolerance)
    if pattern_id == "breakdown":
        level = float(relevant_prices.get("breakdown_level") or relevant_prices.get("confirmation_price") or row["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return bool(float(row["Close"]) > level + tolerance)
    if pattern_id == "bullish_pin_bar":
        return bool(float(row["Close"]) < float(relevant_prices["low"]))
    if pattern_id == "shooting_star":
        return bool(float(row["Close"]) > float(relevant_prices["high"]))
    if bias == "Bullish":
        invalidation_level = _pattern_extreme(df, pattern, "low")
        tolerance = _price_tolerance(df, completion_index, invalidation_level, config)
        return bool(float(row["Close"]) < invalidation_level - tolerance)
    if bias == "Bearish":
        invalidation_level = _pattern_extreme(df, pattern, "high")
        tolerance = _price_tolerance(df, completion_index, invalidation_level, config)
        return bool(float(row["Close"]) > invalidation_level + tolerance)
    return False


def _expiration_transition_index(
    last_completed_index: int,
    reference_index: int,
    expiration_bars: int,
) -> int | None:
    if last_completed_index - reference_index <= expiration_bars:
        return None
    return min(last_completed_index, reference_index + expiration_bars + 1)


def _select_group_primary(group_patterns: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        group_patterns,
        key=lambda item: (
            item["status"] != "confirmed",
            item["bias"] == "Neutral",
            -abs(float(item["base_score"])),
            int(item["priority"]),
            int(item["candles_ago"]),
            item["pattern_name"],
        ),
    )
    return ranked[0]


def _latest_state_reference_index(
    *indices: int | None,
) -> int | None:
    valid_indices = [index for index in indices if index is not None]
    return max(valid_indices) if valid_indices else None


def _level_from_pattern(pattern: dict[str, Any]) -> float | None:
    prices = pattern["event"].relevant_prices
    for key in ("breakout_level", "breakdown_level", "neckline", "confirmation_price"):
        value = prices.get(key)
        if value is not None:
            return float(value)
    return None


def _apply_break_level_lifecycle(
    df: pd.DataFrame,
    pattern: dict[str, Any],
    *,
    interval: str,
    config: PatternConfig,
    scoring_config: ScoringConfig,
) -> dict[str, Any]:
    completion_index = _completion_reference_index(pattern)
    last_completed_index = len(df) - 1
    level = _level_from_pattern(pattern)
    if level is None:
        level = float(df.iloc[completion_index]["Close"])
    tolerance = _price_tolerance(df, completion_index, level, config)
    zone_lower = level - tolerance
    zone_upper = level + tolerance
    direction = "down" if pattern["pattern_id"] == "breakdown" else "up"

    retest_pending_index: int | None = None
    retest_rejected_index: int | None = None
    reclaimed_index: int | None = None
    failed_index: int | None = None
    reclaim_streak = 0

    for scan_index in range(completion_index + 1, len(df)):
        row = df.iloc[scan_index]
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])

        in_zone = high_price >= zone_lower and low_price <= zone_upper
        if retest_pending_index is None and in_zone:
            retest_pending_index = scan_index

        if direction == "down":
            bearish_rejection = (
                retest_pending_index is not None
                and close_price <= level
                and (
                    close_price < zone_lower
                    or (close_price < open_price and high_price >= level)
                )
            )
            reclaimed = close_price > zone_upper
            if retest_rejected_index is None and reclaimed_index is None and bearish_rejection:
                retest_rejected_index = scan_index
            if reclaimed:
                reclaimed_index = scan_index
                reclaim_streak += 1
            else:
                reclaim_streak = 0
            if reclaimed_index is not None and reclaim_streak >= config.reclaim_confirmation_bars:
                failed_index = scan_index
                break
        else:
            bullish_rejection = (
                retest_pending_index is not None
                and close_price >= level
                and (
                    close_price > zone_upper
                    or (close_price > open_price and low_price <= level)
                )
            )
            reclaimed = close_price < zone_lower
            if retest_rejected_index is None and reclaimed_index is None and bullish_rejection:
                retest_rejected_index = scan_index
            if reclaimed:
                reclaimed_index = scan_index
                reclaim_streak += 1
            else:
                reclaim_streak = 0
            if reclaimed_index is not None and reclaim_streak >= config.reclaim_confirmation_bars:
                failed_index = scan_index
                break

    transition_index = _latest_state_reference_index(
        failed_index,
        reclaimed_index,
        retest_rejected_index,
        retest_pending_index,
        completion_index,
    )
    expiration_index = None
    if failed_index is None and reclaimed_index is None:
        expiration_bars = pattern_max_age_bars(pattern, scoring_config)
        expiration_index = _expiration_transition_index(
            last_completed_index,
            transition_index if transition_index is not None else completion_index,
            expiration_bars,
        )

    if failed_index is not None:
        state = "failed_breakdown" if direction == "down" else "failed_breakout"
        state_reference_index = failed_index
    elif reclaimed_index is not None:
        state = "reclaimed"
        state_reference_index = reclaimed_index
    elif retest_rejected_index is not None:
        state = "retest_rejected"
        state_reference_index = retest_rejected_index
    elif retest_pending_index is not None:
        state = "retest_pending"
        state_reference_index = retest_pending_index
    elif expiration_index is not None:
        state = "expired"
        state_reference_index = expiration_index
    elif completion_index == last_completed_index:
        state = "new"
        state_reference_index = completion_index
    else:
        state = "active"
        state_reference_index = completion_index

    return {
        "event_state": state,
        "state_reference_index": state_reference_index,
        "retest_index": retest_pending_index,
        "retest_at": _transition_timestamp(df, retest_pending_index, interval),
        "rejection_index": retest_rejected_index,
        "rejection_at": _transition_timestamp(df, retest_rejected_index, interval),
        "reclaimed_index": reclaimed_index,
        "reclaimed_at": _transition_timestamp(df, reclaimed_index, interval),
        "failed_index": failed_index,
        "failed_at": _transition_timestamp(df, failed_index, interval),
        "invalidated_at": None,
        "expired_at": _transition_timestamp(df, expiration_index, interval) if state == "expired" else None,
        "lifecycle_note": (
            "Price returned to the break level and entered the tolerance zone, but confirmation of the retest outcome is still pending."
            if state == "retest_pending"
            else None
        ),
    }


def _apply_generic_pattern_lifecycle(
    df: pd.DataFrame,
    pattern: dict[str, Any],
    *,
    interval: str,
    config: PatternConfig,
    scoring_config: ScoringConfig,
) -> dict[str, Any]:
    last_completed_index = len(df) - 1
    completion_index = _completion_reference_index(pattern)
    retest_index: int | None = None
    invalidation_index: int | None = None

    for scan_index in range(completion_index + 1, len(df)):
        row = df.iloc[scan_index]
        if _is_invalidated_candle(pattern, row, df, config):
            invalidation_index = scan_index
            break
        if retest_index is None and _family_supports_retest(pattern) and _is_retest_candle(pattern, row, df, config):
            retest_index = scan_index

    if invalidation_index is not None:
        state = "invalidated"
        state_reference_index = invalidation_index
    else:
        retest_reference_index = retest_index if retest_index is not None else completion_index
        expiration_bars = pattern_max_age_bars(pattern, scoring_config)
        expiration_index = _expiration_transition_index(
            last_completed_index,
            retest_reference_index,
            expiration_bars,
        )
        if expiration_index is not None:
            state = "expired"
            state_reference_index = expiration_index
        elif completion_index == last_completed_index:
            state = "new"
            state_reference_index = completion_index
        elif retest_index is not None:
            state = "retested"
            state_reference_index = retest_index
        else:
            state = "active"
            state_reference_index = completion_index

    return {
        "event_state": state,
        "state_reference_index": state_reference_index,
        "retest_index": retest_index,
        "retest_at": _transition_timestamp(df, retest_index, interval),
        "rejection_index": None,
        "rejection_at": None,
        "reclaimed_index": None,
        "reclaimed_at": None,
        "failed_index": None,
        "failed_at": None,
        "invalidated_at": _transition_timestamp(df, invalidation_index, interval),
        "expired_at": (
            _transition_timestamp(df, state_reference_index, interval)
            if state == "expired"
            else None
        ),
        "lifecycle_note": None,
    }


def _apply_pattern_lifecycle(
    df: pd.DataFrame,
    patterns: list[dict[str, Any]],
    *,
    interval: str,
    pattern_config: PatternConfig,
    scoring_config: ScoringConfig,
) -> list[dict[str, Any]]:
    last_completed_index = len(df) - 1
    groups: dict[str, list[dict[str, Any]]] = {}
    for pattern in patterns:
        groups.setdefault(pattern["evidence_group"], []).append(pattern)

    lifecycle_patterns: list[dict[str, Any]] = []
    for group_patterns in groups.values():
        primary = _select_group_primary(group_patterns)
        if primary["pattern_id"] in {"breakout", "breakdown"}:
            lifecycle = _apply_break_level_lifecycle(
                df,
                primary,
                interval=interval,
                config=pattern_config,
                scoring_config=scoring_config,
            )
        elif primary["pattern_id"] in {"double_top", "double_bottom"} and primary["status"] == "tentative":
            state_reference_index = int(primary.get("setup_completion_index") or primary.get("detected_index") or _completion_reference_index(primary))
            expiration_bars = pattern_max_age_bars(primary, scoring_config)
            expiration_index = _expiration_transition_index(
                last_completed_index,
                state_reference_index,
                expiration_bars,
            )
            if expiration_index is not None:
                lifecycle = {
                    "event_state": "expired",
                    "state_reference_index": expiration_index,
                    "retest_index": None,
                    "retest_at": None,
                    "rejection_index": None,
                    "rejection_at": None,
                    "reclaimed_index": None,
                    "reclaimed_at": None,
                    "failed_index": None,
                    "failed_at": None,
                    "invalidated_at": None,
                    "expired_at": _transition_timestamp(df, expiration_index, interval),
                    "lifecycle_note": "The tentative structural setup expired before confirmation arrived.",
                }
            else:
                lifecycle = {
                    "event_state": "awaiting_confirmation",
                    "state_reference_index": state_reference_index,
                    "retest_index": None,
                    "retest_at": None,
                    "rejection_index": None,
                    "rejection_at": None,
                    "reclaimed_index": None,
                    "reclaimed_at": None,
                    "failed_index": None,
                    "failed_at": None,
                    "invalidated_at": None,
                    "expired_at": None,
                    "lifecycle_note": "The structural setup completed, but neckline confirmation has not yet occurred.",
                }
        elif primary["status"] == "failed":
            lifecycle = {
                "event_state": "failed",
                "state_reference_index": primary.get("confirmation_index")
                or primary.get("detected_index")
                or _completion_reference_index(primary),
                "retest_index": None,
                "retest_at": None,
                "rejection_index": None,
                "rejection_at": None,
                "reclaimed_index": None,
                "reclaimed_at": None,
                "failed_index": primary.get("confirmation_index") or primary.get("detected_index"),
                "failed_at": _transition_timestamp(
                    df,
                    primary.get("confirmation_index") or primary.get("detected_index"),
                    interval,
                ),
                "invalidated_at": None,
                "expired_at": None,
                "lifecycle_note": None,
            }
        elif primary["status"] == "expired":
            state_reference_index = primary.get("confirmation_index") or primary.get("detected_index")
            lifecycle = {
                "event_state": "expired",
                "state_reference_index": state_reference_index,
                "retest_index": None,
                "retest_at": None,
                "rejection_index": None,
                "rejection_at": None,
                "reclaimed_index": None,
                "reclaimed_at": None,
                "failed_index": None,
                "failed_at": None,
                "invalidated_at": None,
                "expired_at": _transition_timestamp(df, state_reference_index, interval),
                "lifecycle_note": None,
            }
        else:
            lifecycle = _apply_generic_pattern_lifecycle(
                df,
                primary,
                interval=interval,
                config=pattern_config,
                scoring_config=scoring_config,
            )

        group_state = str(lifecycle["event_state"])
        state_reference_index = lifecycle["state_reference_index"]
        state_updated_at = _transition_timestamp(df, state_reference_index, interval)

        for pattern in group_patterns:
            family_state = group_state
            if group_state == "retested" and not _family_supports_retest(pattern):
                family_state = "active"
            updated = dict(pattern)
            updated["event_state"] = family_state
            updated["state_updated_at"] = state_updated_at
            updated["retest_index"] = lifecycle["retest_index"]
            updated["retest_at"] = (
                lifecycle["retest_at"]
                if family_state in {"retested", "retest_pending", "retest_rejected", "reclaimed", "failed_breakout", "failed_breakdown"}
                else None
            )
            updated["rejection_index"] = lifecycle["rejection_index"]
            updated["rejection_at"] = lifecycle["rejection_at"] if family_state == "retest_rejected" else None
            updated["reclaimed_index"] = lifecycle["reclaimed_index"]
            updated["reclaimed_at"] = lifecycle["reclaimed_at"] if family_state == "reclaimed" else None
            updated["failed_index"] = lifecycle["failed_index"]
            updated["failed_at"] = lifecycle["failed_at"] if family_state in {"failed", "failed_breakout", "failed_breakdown"} else None
            updated["invalidation_index"] = lifecycle.get("invalidation_index")
            updated["invalidated_at"] = lifecycle["invalidated_at"] if family_state == "invalidated" else None
            updated["expired_at"] = lifecycle["expired_at"] if family_state == "expired" else None
            updated["last_completed_candle_index"] = last_completed_index
            updated["lifecycle_transition_timestamp"] = state_updated_at
            updated["lifecycle_note"] = lifecycle.get("lifecycle_note")
            lifecycle_patterns.append(updated)

    return lifecycle_patterns


def _link_related_patterns(
    patterns: list[dict[str, Any]],
    *,
    pattern_config: PatternConfig,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    annotated = [dict(pattern) for pattern in patterns]
    by_event_id = {pattern["event_id"]: pattern for pattern in annotated}
    structural_ids = {"double_top": "breakdown", "double_bottom": "breakout"}

    for pattern in annotated:
        trigger_id = structural_ids.get(pattern["pattern_id"])
        if trigger_id is None or pattern["status"] != "confirmed":
            continue

        setup_completion_index = int(pattern.get("setup_completion_index") or pattern["pattern_completion_index"])
        confirmation_index = int(pattern.get("confirmation_index") or pattern["detected_index"])
        neckline = _level_from_pattern(pattern)
        if neckline is None:
            continue
        tolerance = _price_tolerance(df, setup_completion_index, neckline, pattern_config)

        for candidate in annotated:
            if candidate["pattern_id"] != trigger_id or candidate["status"] != "confirmed":
                continue
            trigger_index = int(candidate["detected_index"])
            trigger_level = _level_from_pattern(candidate)
            if trigger_level is None:
                continue
            if trigger_index < setup_completion_index or abs(trigger_index - confirmation_index) > 1:
                continue
            if abs(trigger_level - neckline) > tolerance:
                continue

            pattern["related_event_ids"] = sorted(
                set((pattern.get("related_event_ids") or []) + [candidate["event_id"]])
            )
            pattern["relationship_type"] = "confirmed_by"
            candidate["related_event_ids"] = sorted(
                set((candidate.get("related_event_ids") or []) + [pattern["event_id"]])
            )
            candidate["relationship_type"] = "confirms"
            candidate["confirms_pattern_id"] = pattern["event_id"]
            candidate["parent_pattern_id"] = pattern["setup_id"]
            by_event_id[pattern["event_id"]] = pattern
            by_event_id[candidate["event_id"]] = candidate

    return list(by_event_id.values())


def _current_score_exclusion_reason(pattern: dict[str, Any]) -> str | None:
    if pattern.get("group_suppressed"):
        return "overlap duplicate"
    if pattern.get("dependency_suppressed"):
        return "linked confirmation duplicate"
    if pattern["event_state"] == "invalidated":
        return "invalidated"
    if pattern["event_state"] == "expired":
        return "expired"
    if pattern.get("score_ineligibility_reason") and not pattern["score_eligible"]:
        return str(pattern["score_ineligibility_reason"])
    if not pattern["score_eligible"]:
        return "outside scoring horizon"
    if pattern["bias"] == "Neutral":
        return "informational only"
    if abs(float(pattern.get("pattern_score_contribution", 0.0))) <= 0 and abs(float(pattern.get("volume_score_contribution", 0.0))) <= 0:
        return "informational only"
    return None


def _invalidation_condition_text(
    pattern: dict[str, Any],
    *,
    df: pd.DataFrame,
    config: PatternConfig,
) -> str | None:
    completion_index = _completion_reference_index(pattern)
    event = pattern["event"]
    relevant_prices = event.relevant_prices
    pattern_id = str(pattern["pattern_id"])

    if pattern_id == "breakout":
        level = float(relevant_prices.get("breakout_level") or relevant_prices.get("confirmation_price") or df.iloc[completion_index]["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return f"A completed close below {level - tolerance:.2f} would invalidate this breakout."
    if pattern_id == "breakdown":
        level = float(relevant_prices.get("breakdown_level") or relevant_prices.get("confirmation_price") or df.iloc[completion_index]["Close"])
        tolerance = _price_tolerance(df, completion_index, level, config)
        return f"A completed close above {level + tolerance:.2f} would invalidate this breakdown."
    if pattern_id == "bullish_pin_bar":
        return f"A completed close below {float(relevant_prices['low']):.2f} would invalidate this bullish pin bar."
    if pattern_id == "shooting_star":
        return f"A completed close above {float(relevant_prices['high']):.2f} would invalidate this shooting star."

    bias = str(pattern.get("bias") or "")
    if bias == "Bullish":
        invalidation_level = _pattern_extreme(df, pattern, "low")
        tolerance = _price_tolerance(df, completion_index, invalidation_level, config)
        return f"A completed close below {invalidation_level - tolerance:.2f} would invalidate this bullish setup."
    if bias == "Bearish":
        invalidation_level = _pattern_extreme(df, pattern, "high")
        tolerance = _price_tolerance(df, completion_index, invalidation_level, config)
        return f"A completed close above {invalidation_level + tolerance:.2f} would invalidate this bearish setup."
    return None


def _build_canonical_event_groups(
    patterns: list[dict[str, Any]],
    *,
    display_timezone,
    interval: str,
    df: pd.DataFrame,
    pattern_config: PatternConfig,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pattern in patterns:
        grouped.setdefault(pattern["evidence_group"], []).append(pattern)

    canonical_events: list[dict[str, Any]] = []
    for group_patterns in grouped.values():
        primary = _select_group_primary(group_patterns)
        event = primary["event"]
        labels = list(
            dict.fromkeys(
                pattern["pattern_name"]
                for pattern in sorted(group_patterns, key=lambda item: item["pattern_name"])
            )
        )
        primary_completion = pd.Timestamp(event.pattern_end_at)
        primary_detected = pd.Timestamp(event.detected_at)
        state_updated_at = primary.get("state_updated_at")
        retest_at = primary.get("retest_at")
        rejection_at = primary.get("rejection_at")
        reclaimed_at = primary.get("reclaimed_at")
        failed_at = primary.get("failed_at")
        invalidated_at = primary.get("invalidated_at")
        expired_at = primary.get("expired_at")
        inclusion_reason = _current_score_exclusion_reason(primary)
        included_in_current_score = inclusion_reason is None and bool(primary.get("group_primary", False))
        current_weighted_score = (
            round(float(primary.get("pattern_score_contribution", 0.0)) + float(primary.get("volume_score_contribution", 0.0)), 2)
            if included_in_current_score
            else 0.0
        )
        invalidation_condition = _invalidation_condition_text(
            primary,
            df=df,
            config=pattern_config,
        )
        overlap_label_count = max(0, len(labels) - 1)
        canonical_events.append(
            {
                "event_id": f"canonical:{primary['evidence_group']}",
                "primary_pattern_name": primary["pattern_name"],
                "pattern_labels": labels,
                "family": primary["pattern_family"],
                "bias": primary["bias"],
                "pattern_start": format_iso_timestamp(event.pattern_start_at),
                "setup_completion": format_iso_timestamp(event.setup_completion_at or event.pattern_end_at),
                "pattern_completion": format_iso_timestamp(primary_completion),
                "detected_at": format_iso_timestamp(primary_detected),
                "confirmation_at": (
                    format_iso_timestamp(event.confirmation_at)
                    if event.confirmation_at is not None
                    else None
                ),
                "completion_index": int(primary["pattern_completion_index"]),
                "last_completed_candle_index": int(primary["last_completed_candle_index"]),
                "state": primary["event_state"],
                "state_updated_at": format_iso_timestamp(state_updated_at) if state_updated_at is not None else None,
                "retest_at": format_iso_timestamp(retest_at) if retest_at is not None else None,
                "rejection_at": format_iso_timestamp(rejection_at) if rejection_at is not None else None,
                "reclaimed_at": format_iso_timestamp(reclaimed_at) if reclaimed_at is not None else None,
                "failed_at": format_iso_timestamp(failed_at) if failed_at is not None else None,
                "invalidated_at": format_iso_timestamp(invalidated_at) if invalidated_at is not None else None,
                "expired_at": format_iso_timestamp(expired_at) if expired_at is not None else None,
                "signal_strength": float(primary["signal_strength"]),
                "raw_score": float(primary["base_score"]),
                "current_weighted_score": current_weighted_score,
                "evidence_group": primary["evidence_group"],
                "included_in_current_score": included_in_current_score,
                "exclusion_reason": inclusion_reason,
                "exchange_timezone": primary["exchange_timezone"],
                "display_timezone": str(display_timezone),
                "pattern_start_display": format_display_datetime(event.pattern_start_at, display_timezone),
                "setup_completion_display": format_display_datetime(
                    event.setup_completion_at or event.pattern_end_at,
                    display_timezone,
                ),
                "pattern_completion_display": format_display_datetime(primary_completion, display_timezone),
                "detected_at_display": format_display_datetime(primary_detected, display_timezone),
                "confirmation_at_display": (
                    format_display_datetime(event.confirmation_at, display_timezone)
                    if event.confirmation_at is not None
                    else None
                ),
                "state_updated_at_display": (
                    format_display_datetime(state_updated_at, display_timezone)
                    if state_updated_at is not None
                    else None
                ),
                "retest_at_display": (
                    format_display_datetime(retest_at, display_timezone)
                    if retest_at is not None
                    else None
                ),
                "rejection_at_display": (
                    format_display_datetime(rejection_at, display_timezone)
                    if rejection_at is not None
                    else None
                ),
                "reclaimed_at_display": (
                    format_display_datetime(reclaimed_at, display_timezone)
                    if reclaimed_at is not None
                    else None
                ),
                "failed_at_display": (
                    format_display_datetime(failed_at, display_timezone)
                    if failed_at is not None
                    else None
                ),
                "invalidated_at_display": (
                    format_display_datetime(invalidated_at, display_timezone)
                    if invalidated_at is not None
                    else None
                ),
                "expired_at_display": (
                    format_display_datetime(expired_at, display_timezone)
                    if expired_at is not None
                    else None
                ),
                "label_count": len(labels),
                "overlap_label_count": overlap_label_count,
                "overlap_note": (
                    f"{len(labels)} overlapping candle labels were grouped into 1 candlestick event."
                    if overlap_label_count > 0
                    else None
                ),
                "relationship_type": primary.get("relationship_type"),
                "related_event_ids": primary.get("related_event_ids") or [],
                "related_note": (
                    "This event confirmed a previously identified structural setup."
                    if primary.get("relationship_type") == "confirms"
                    else (
                        "This structural setup remains separate from its confirmation trigger, and dependency-aware scoring prevented double counting."
                        if primary.get("relationship_type") == "confirmed_by"
                        else None
                    )
                ),
                "current_score_exclusion_reason": inclusion_reason,
                "invalidation_condition": invalidation_condition,
                "session_date": _exchange_session_date(primary_completion, primary["exchange_timezone"]),
            }
        )

    return sorted(
        canonical_events,
        key=lambda item: (item["detected_at"], item["primary_pattern_name"]),
    )


def _augment_scoring_annotations(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_groups = {
        str(pattern["evidence_group"])
        for pattern in patterns
        if pattern.get("group_primary")
    }
    annotated: list[dict[str, Any]] = []
    for pattern in patterns:
        item = dict(pattern)
        if (
            not item.get("group_primary")
            and not item.get("group_suppressed")
            and str(item["evidence_group"]) in primary_groups
        ):
            item["group_suppressed"] = True
        exclusion_reason = _current_score_exclusion_reason(item)
        item["included_in_current_score"] = exclusion_reason is None and bool(item.get("group_primary", False))
        item["exclusion_reason"] = exclusion_reason
        annotated.append(item)
    return annotated


def _canonical_display_timestamp(event: dict[str, Any], *field_names: str) -> pd.Timestamp:
    for field_name in field_names:
        value = event.get(field_name)
        if value:
            return pd.Timestamp(value)
    return pd.Timestamp.min.tz_localize("UTC")


def _build_session_context(
    canonical_events: list[dict[str, Any]],
    session_date: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_history = [event for event in canonical_events if event["session_date"] == session_date]
    current_relevant = [
        event
        for event in canonical_events
        if event["state"] in {"new", "active", "retested", "retest_pending", "retest_rejected", "reclaimed", "awaiting_confirmation"}
    ]
    current_relevant = sorted(
        current_relevant,
        key=lambda event: (
            -_canonical_display_timestamp(
                event,
                "state_updated_at",
                "reclaimed_at",
                "rejection_at",
                "retest_at",
                "confirmation_at",
                "detected_at",
            ).value,
            event["primary_pattern_name"],
        ),
    )
    return current_relevant, session_history


def _build_explanation_sections(
    structured_explanation: dict[str, Any],
    *,
    current_relevant_patterns: list[dict[str, Any]],
    session_pattern_history: list[dict[str, Any]],
    latest_canonical_labels: list[str],
) -> dict[str, Any]:
    current_pattern_lines: list[str] = []
    for event in current_relevant_patterns[:5]:
        line = (
            f"{event['primary_pattern_name']} [{event['state']}] detected at "
            f"{event['detected_at_display']}."
        )
        state_updated_display = event.get("state_updated_at_display")
        detected_display = event.get("detected_at_display")
        if state_updated_display and state_updated_display != detected_display:
            line = f"{line[:-1]} State last changed at {state_updated_display}."
        if not event.get("included_in_current_score") and event.get("current_score_exclusion_reason"):
            line = (
                f"{line[:-1]} It is not part of the current score because "
                f"{event['current_score_exclusion_reason']}."
            )
        current_pattern_lines.append(line)
    state_counts: dict[str, int] = {}
    for event in session_pattern_history:
        state_counts[event["state"]] = state_counts.get(event["state"], 0) + 1

    session_context_lines = [
        f"{len(session_pattern_history)} canonical pattern event(s) were detected during the relevant session.",
        f"{sum(1 for event in session_pattern_history if event['included_in_current_score'])} currently contribute to the latest signal.",
    ]
    if state_counts:
        state_summary = ", ".join(f"{count} {state}" for state, count in sorted(state_counts.items()))
        session_context_lines.append(f"Session lifecycle summary: {state_summary}.")
    if current_relevant_patterns:
        latest_event = current_relevant_patterns[0]
        latest_labels = ", ".join(latest_event["pattern_labels"])
        latest_transition = latest_event.get("state_updated_at_display") or latest_event.get("detected_at_display")
        lifecycle_note = (
            f"The latest current pattern event ({latest_labels}) was last updated at {latest_transition}, "
            "and only completed candles were allowed to change lifecycle states."
        )
    elif latest_canonical_labels:
        lifecycle_note = (
            f"The latest overlapping candle labels ({', '.join(latest_canonical_labels)}) were evaluated only on completed candles."
        )
    else:
        lifecycle_note = "Only completed candles were allowed to change lifecycle states."
    lifecycle_note += " The current incomplete candle was excluded from lifecycle transitions."

    enriched = dict(structured_explanation)
    enriched["current_pattern_evidence"] = current_pattern_lines
    enriched["session_context"] = session_context_lines
    enriched["lifecycle_note"] = lifecycle_note
    return enriched


def _serialize_pattern_event(
    pattern: dict[str, Any],
    display_timezone,
    *,
    df: pd.DataFrame,
    pattern_config: PatternConfig,
) -> dict[str, Any]:
    event: PatternEvent = pattern["event"]
    pattern_start_exchange = event.pattern_start_at
    pattern_end_exchange = event.pattern_end_at
    bar_start_exchange = event.bar_start_at
    bar_end_exchange = event.bar_end_at
    detected_at_exchange = event.detected_at
    setup_completion_exchange = event.setup_completion_at or event.pattern_end_at
    confirmation_at_exchange = event.confirmation_at

    pattern_start_display = convert_to_timezone(pattern_start_exchange, display_timezone)
    pattern_end_display = convert_to_timezone(pattern_end_exchange, display_timezone)
    bar_start_display = convert_to_timezone(bar_start_exchange, display_timezone)
    bar_end_display = convert_to_timezone(bar_end_exchange, display_timezone)
    detected_at_display = convert_to_timezone(detected_at_exchange, display_timezone)
    setup_completion_display = convert_to_timezone(setup_completion_exchange, display_timezone)
    confirmation_at_display = (
        convert_to_timezone(confirmation_at_exchange, display_timezone)
        if confirmation_at_exchange is not None
        else None
    )
    invalidation_condition = _invalidation_condition_text(
        pattern,
        df=df,
        config=pattern_config,
    )

    return {
        "event_id": pattern["event_id"],
        "setup_id": pattern["setup_id"],
        "evidence_group": pattern["evidence_group"],
        "event_state": pattern["event_state"],
        "pattern_id": event.pattern_id,
        "pattern_name": event.pattern_name,
        "pattern_family": event.pattern_family.value,
        "bias": event.bias,
        "status": event.status.value,
        "pattern_start_at": format_iso_timestamp(pattern_start_exchange),
        "pattern_end_at": format_iso_timestamp(pattern_end_exchange),
        "bar_start_at": format_iso_timestamp(bar_start_exchange),
        "bar_end_at": format_iso_timestamp(bar_end_exchange),
        "detected_at": format_iso_timestamp(detected_at_exchange),
        "setup_completion_at": format_iso_timestamp(setup_completion_exchange),
        "confirmation_at": (
            format_iso_timestamp(confirmation_at_exchange)
            if confirmation_at_exchange is not None
            else None
        ),
        "pattern_start_at_utc": format_iso_timestamp(pattern_start_exchange, timezone="UTC"),
        "pattern_end_at_utc": format_iso_timestamp(pattern_end_exchange, timezone="UTC"),
        "bar_start_at_utc": format_iso_timestamp(bar_start_exchange, timezone="UTC"),
        "bar_end_at_utc": format_iso_timestamp(bar_end_exchange, timezone="UTC"),
        "detected_at_utc": format_iso_timestamp(detected_at_exchange, timezone="UTC"),
        "setup_completion_at_utc": format_iso_timestamp(setup_completion_exchange, timezone="UTC"),
        "confirmation_at_utc": (
            format_iso_timestamp(confirmation_at_exchange, timezone="UTC")
            if confirmation_at_exchange is not None
            else None
        ),
        "exchange_timezone": event.exchange_timezone,
        "display_timezone": str(display_timezone),
        "pattern_start_exchange": format_display_datetime(pattern_start_exchange, event.exchange_timezone),
        "pattern_end_exchange": format_display_datetime(pattern_end_exchange, event.exchange_timezone),
        "bar_start_exchange": format_display_datetime(bar_start_exchange, event.exchange_timezone),
        "bar_end_exchange": format_display_datetime(bar_end_exchange, event.exchange_timezone),
        "detected_at_exchange": format_display_datetime(detected_at_exchange, event.exchange_timezone),
        "setup_completion_exchange": format_display_datetime(setup_completion_exchange, event.exchange_timezone),
        "confirmation_at_exchange": (
            format_display_datetime(confirmation_at_exchange, event.exchange_timezone)
            if confirmation_at_exchange is not None
            else None
        ),
        "pattern_start_display": format_display_datetime(pattern_start_display, display_timezone),
        "pattern_end_display": format_display_datetime(pattern_end_display, display_timezone),
        "bar_start_display": format_display_datetime(bar_start_display, display_timezone),
        "bar_end_display": format_display_datetime(bar_end_display, display_timezone),
        "detected_at_display": format_display_datetime(detected_at_display, display_timezone),
        "setup_completion_display": format_display_datetime(setup_completion_display, display_timezone),
        "confirmation_at_display": (
            format_display_datetime(confirmation_at_display, display_timezone)
            if confirmation_at_display is not None
            else None
        ),
        "candles_ago": pattern["candles_ago"],
        "base_score": pattern["base_score"],
        "weighted_score": pattern["weighted_score"],
        "pattern_score_contribution": pattern["pattern_score_contribution"],
        "volume_score_contribution": pattern["volume_score_contribution"],
        "detection_reason": pattern["detection_reason"],
        "signal_strength": pattern["signal_strength"],
        "strength_label": pattern["strength_label"],
        "volume_baseline_source": pattern["volume_baseline_source"],
        "score_anchor_type": pattern.get("score_anchor_type"),
        "score_anchor_index": pattern.get("score_anchor_index"),
        "score_anchor_candles_ago": pattern.get("score_anchor_candles_ago"),
        "score_max_age_bars": pattern.get("score_max_age_bars"),
        "score_eligibility": pattern.get("score_eligibility"),
        "group_primary": pattern["group_primary"],
        "group_suppressed": pattern["group_suppressed"],
        "pattern_start_index": pattern.get("pattern_start_index"),
        "pattern_completion_index": pattern.get("pattern_completion_index"),
        "detected_index": pattern.get("detected_index"),
        "setup_completion_index": pattern.get("setup_completion_index"),
        "confirmation_index": pattern.get("confirmation_index"),
        "last_completed_candle_index": pattern.get("last_completed_candle_index"),
        "state_updated_at": (
            format_iso_timestamp(pattern["state_updated_at"])
            if pattern.get("state_updated_at") is not None
            else None
        ),
        "state_updated_at_display": (
            format_display_datetime(pattern["state_updated_at"], display_timezone)
            if pattern.get("state_updated_at") is not None
            else None
        ),
        "retest_index": pattern.get("retest_index"),
        "retest_at": (
            format_iso_timestamp(pattern["retest_at"])
            if pattern.get("retest_at") is not None
            else None
        ),
        "retest_at_display": (
            format_display_datetime(pattern["retest_at"], display_timezone)
            if pattern.get("retest_at") is not None
            else None
        ),
        "invalidation_index": pattern.get("invalidation_index"),
        "rejection_index": pattern.get("rejection_index"),
        "rejection_at": (
            format_iso_timestamp(pattern["rejection_at"])
            if pattern.get("rejection_at") is not None
            else None
        ),
        "rejection_at_display": (
            format_display_datetime(pattern["rejection_at"], display_timezone)
            if pattern.get("rejection_at") is not None
            else None
        ),
        "reclaimed_index": pattern.get("reclaimed_index"),
        "reclaimed_at": (
            format_iso_timestamp(pattern["reclaimed_at"])
            if pattern.get("reclaimed_at") is not None
            else None
        ),
        "reclaimed_at_display": (
            format_display_datetime(pattern["reclaimed_at"], display_timezone)
            if pattern.get("reclaimed_at") is not None
            else None
        ),
        "failed_index": pattern.get("failed_index"),
        "failed_at": (
            format_iso_timestamp(pattern["failed_at"])
            if pattern.get("failed_at") is not None
            else None
        ),
        "failed_at_display": (
            format_display_datetime(pattern["failed_at"], display_timezone)
            if pattern.get("failed_at") is not None
            else None
        ),
        "invalidated_at": (
            format_iso_timestamp(pattern["invalidated_at"])
            if pattern.get("invalidated_at") is not None
            else None
        ),
        "invalidated_at_display": (
            format_display_datetime(pattern["invalidated_at"], display_timezone)
            if pattern.get("invalidated_at") is not None
            else None
        ),
        "expired_at": (
            format_iso_timestamp(pattern["expired_at"])
            if pattern.get("expired_at") is not None
            else None
        ),
        "expired_at_display": (
            format_display_datetime(pattern["expired_at"], display_timezone)
            if pattern.get("expired_at") is not None
            else None
        ),
        "included_in_current_score": pattern.get("included_in_current_score", False),
        "exclusion_reason": pattern.get("exclusion_reason"),
        "relationship_type": pattern.get("relationship_type"),
        "related_event_ids": pattern.get("related_event_ids") or [],
        "confirms_pattern_id": pattern.get("confirms_pattern_id"),
        "parent_pattern_id": pattern.get("parent_pattern_id"),
        "lifecycle_note": pattern.get("lifecycle_note"),
        "invalidation_condition": invalidation_condition,
        "relevant_prices": event.relevant_prices,
        "relevant_indices": event.relevant_indices,
    }


def _collect_analysis_validation_warnings(
    patterns: list[dict[str, Any]],
    score: dict[str, float],
) -> list[str]:
    warnings: list[str] = []
    included_patterns = [pattern for pattern in patterns if pattern.get("included_in_current_score")]

    bullish_total = round(
        sum(max(float(pattern.get("pattern_score_contribution", 0.0)), 0.0) for pattern in included_patterns),
        2,
    )
    bearish_total = round(
        sum(abs(min(float(pattern.get("pattern_score_contribution", 0.0)), 0.0)) for pattern in included_patterns),
        2,
    )
    pattern_total = round(
        sum(float(pattern.get("pattern_score_contribution", 0.0)) for pattern in included_patterns),
        2,
    )
    volume_total = round(
        sum(float(pattern.get("volume_score_contribution", 0.0)) for pattern in included_patterns),
        2,
    )
    net_total = round(pattern_total + volume_total + float(score["trend_score"]), 2)
    if bullish_total != round(float(score["bullish_score"]), 2):
        warnings.append("Internal validation: bullish_score did not match included bullish pattern contributions.")
    if bearish_total != round(float(score["bearish_score"]), 2):
        warnings.append("Internal validation: bearish_score did not match included bearish pattern contributions.")
    if pattern_total != round(float(score["pattern_score"]), 2):
        warnings.append("Internal validation: pattern_score did not match included pattern contributions.")
    if volume_total != round(float(score["volume_score"]), 2):
        warnings.append("Internal validation: volume_score did not match included volume contributions.")
    if net_total != round(float(score["net_signal_score"]), 2):
        warnings.append("Internal validation: net_signal_score did not match the reconciled score components.")

    for pattern in patterns:
        event = pattern["event"]
        if pattern.get("included_in_current_score") and pattern.get("exclusion_reason"):
            warnings.append(
                f"Internal validation: {pattern['pattern_name']} was included in score despite exclusion reason "
                f"'{pattern['exclusion_reason']}'."
            )
        if pattern.get("included_in_current_score") and pattern["event_state"] in {"expired", "invalidated", "failed", "failed_breakout", "failed_breakdown", "awaiting_confirmation"}:
            warnings.append(
                f"Internal validation: {pattern['pattern_name']} was included in score despite ineligible state "
                f"'{pattern['event_state']}'."
            )
        if not pattern.get("included_in_current_score") and (
            abs(float(pattern.get("pattern_score_contribution", 0.0))) > 0
            or abs(float(pattern.get("volume_score_contribution", 0.0))) > 0
        ):
            warnings.append(
                f"Internal validation: {pattern['pattern_name']} was excluded from score but kept a nonzero contribution."
            )
        if event.confirmation_at is not None and event.setup_completion_at is not None and event.confirmation_at < event.setup_completion_at:
            warnings.append(
                f"Internal validation: {pattern['pattern_name']} had confirmation_at earlier than setup_completion_at."
            )
        for timestamp in (
            event.pattern_start_at,
            event.pattern_end_at,
            event.bar_start_at,
            event.bar_end_at,
            event.detected_at,
            event.setup_completion_at,
            event.confirmation_at,
        ):
            if timestamp is not None and pd.Timestamp(timestamp).tzinfo is None:
                warnings.append(
                    f"Internal validation: {pattern['pattern_name']} contained a naive timestamp."
                )
                break

    return warnings


def _collect_history_warnings(
    completed_df: pd.DataFrame,
    registry: PatternRegistry,
) -> list[str]:
    if completed_df.empty:
        return []
    available_bars = len(completed_df)
    required_histories = sorted(
        {
            int(detector.minimum_required_history)
            for detector in registry.detectors
            if available_bars < int(detector.minimum_required_history)
        }
    )
    if not required_histories:
        return []
    return [
        "Only "
        f"{available_bars} completed bar(s) were available, so detectors requiring "
        f"{required_histories[0]} to {required_histories[-1]} bars could not participate fully."
    ]


def analyze_dataframe(
    df: pd.DataFrame,
    symbol: str = "DATAFRAME",
    interval: str = "15m",
    as_of: Optional[pd.Timestamp] = None,
    display_timezone: str = "Asia/Jerusalem",
    lookback_bars: int = 12,
    top_pattern_count: int = 3,
    instrument: Optional[ResolvedInstrument] = None,
    exchange_timezone: str | None = None,
    strict_data: bool = True,
    data_quality_report: DataQualityReport | None = None,
    validate_data: bool = True,
    metadata: dict[str, Any] | None = None,
    registry: PatternRegistry | None = None,
    include_extended_hours: bool = True,
    session_mode: str | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> dict[str, Any]:
    """Analyze a prepared OHLCV DataFrame using completed candles only."""
    base_config = AnalysisConfig()
    config = AnalysisConfig(
        interval=interval,
        period=base_config.period,
        pattern=base_config.pattern,
        scoring=base_config.scoring.__class__(
            lookback_bars=lookback_bars,
            top_pattern_count=top_pattern_count,
            pattern_max_age_bars=lookback_bars,
        ),
        timezone=base_config.timezone.__class__(display_timezone=display_timezone),
    )
    config.validate()
    display_zone = config.timezone.to_zoneinfo()
    active_registry = registry or DEFAULT_PATTERN_REGISTRY
    provider_name = str((metadata or {}).get("source", "dataframe"))
    effective_session_mode = session_mode or ("extended" if include_extended_hours else "regular")
    context = build_analysis_context(
        symbol=symbol,
        interval=interval,
        display_timezone=display_timezone,
        session_mode=effective_session_mode,
        instrument=instrument,
        provider=provider_name,
        provider_metadata=metadata,
        exchange_timezone_override=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
        cache_config={"strict_data": strict_data},
    )

    if validate_data:
        validated_df, quality_report = validate_market_data(
            df,
            interval=interval,
            exchange_timezone=exchange_timezone,
            as_of=as_of,
            strict_data=strict_data,
            include_extended_hours=context.include_extended_hours,
            session_mode=context.session_mode,
            regular_session_start=context.regular_session_start,
            regular_session_end=context.regular_session_end,
            context=context,
        )
    else:
        validated_df = df.copy()
        quality_report = data_quality_report or DataQualityReport(
            row_count=len(validated_df),
            completed_row_count=len(validated_df),
            duplicate_count=0,
            missing_value_count=0,
            invalid_ohlc_count=0,
            irregular_gap_count=0,
            warnings=[],
            cleaning_actions=[],
        )

    completed_df, normalized_as_of, quality_report = _filter_completed_candles(
        validated_df,
        interval,
        as_of,
        quality_report,
    )
    context = context.with_runtime_state(
        analysis_time=normalized_as_of,
        latest_completed_candle_start=pd.Timestamp(completed_df.iloc[-1]["Datetime"]),
        latest_completed_candle_end=_get_bar_end(pd.Timestamp(completed_df.iloc[-1]["Datetime"]), interval),
        dataframe_identity_value=dataframe_identity(completed_df),
        warnings=list(context.warnings),
    )
    pattern_df, raw_events = _run_pattern_pipeline(
        completed_df,
        config,
        active_registry,
        exchange_timezone=context.exchange_timezone,
        regular_session_start=context.regular_session_start,
        regular_session_end=context.regular_session_end,
    )

    latest_row = pattern_df.iloc[-1]
    trend = str(latest_row["Trend"])
    trend_structure_score = round(float(latest_row.get("Trend_Score", 0.0)), 2)
    trend_evidence = list(latest_row.get("Trend_Evidence", []))
    trend_horizon = str(latest_row.get("Trend_Horizon", "Short-to-medium term"))
    session_info = _latest_completed_session_info(pattern_df, interval, context=context)
    resolved_events, ignored_patterns_count = resolve_pattern_conflicts(raw_events)
    prepared_patterns = _prepare_pattern_records(
        pattern_df,
        resolved_events,
        interval,
        active_registry,
        score_tentative_patterns=config.pattern.score_tentative_patterns,
    )
    prepared_patterns = _annotate_pattern_identity(prepared_patterns)
    prepared_patterns = _apply_pattern_lifecycle(
        pattern_df,
        prepared_patterns,
        interval=interval,
        pattern_config=config.pattern,
        scoring_config=config.scoring,
    )
    prepared_patterns = _link_related_patterns(
        prepared_patterns,
        pattern_config=config.pattern,
        df=pattern_df,
    )
    latest_bar_start_exchange = pd.Timestamp(latest_row["Datetime"])
    latest_bar_end_exchange = _get_bar_end(latest_bar_start_exchange, interval)
    latest_bar_start_display = convert_to_timezone(latest_bar_start_exchange, display_zone)
    latest_bar_end_display = convert_to_timezone(latest_bar_end_exchange, display_zone)
    latest_close = round(float(latest_row["Close"]), 2)
    latest_volume_baseline_source = str(latest_row.get("Volume_Baseline_Source", "unknown"))

    scoring_result = ScoringService(config.scoring).evaluate(
        symbol=symbol,
        trend=trend,
        trend_structure_score=trend_structure_score,
        trend_evidence=trend_evidence,
        trend_horizon=trend_horizon,
        display_timezone=str(display_zone),
        patterns=prepared_patterns,
        quality_report=quality_report,
        latest_close=latest_close,
        latest_bar_start_display=format_display_datetime(latest_bar_start_display, display_zone),
        latest_bar_end_display=format_display_datetime(latest_bar_end_display, display_zone),
        interval=interval,
        latest_volume_baseline_source=latest_volume_baseline_source,
    )
    score = scoring_result["score"]
    market_state = scoring_result["market_state"]
    overall_bias = scoring_result["overall_bias"]
    rule_confidence = scoring_result["rule_confidence"]
    ranked_patterns = _augment_scoring_annotations(scoring_result["patterns"])
    top_patterns_internal = ranked_patterns[: config.scoring.top_pattern_count]
    canonical_events = _build_canonical_event_groups(
        ranked_patterns,
        display_timezone=display_zone,
        interval=interval,
        df=pattern_df,
        pattern_config=config.pattern,
    )
    current_relevant_patterns, session_pattern_history = _build_session_context(
        canonical_events,
        session_info["session_date"],
    )
    latest_canonical_labels = current_relevant_patterns[0]["pattern_labels"] if current_relevant_patterns else []
    structured_explanation = _build_explanation_sections(
        scoring_result["structured_explanation"],
        current_relevant_patterns=current_relevant_patterns,
        session_pattern_history=session_pattern_history,
        latest_canonical_labels=latest_canonical_labels,
    )
    explanation = scoring_result["explanation"]
    if structured_explanation.get("current_pattern_evidence"):
        explanation += " Current pattern evidence: " + "; ".join(structured_explanation["current_pattern_evidence"]) + "."
    if structured_explanation.get("session_context"):
        explanation += " Session context: " + " ".join(structured_explanation["session_context"])
    if structured_explanation.get("lifecycle_note"):
        explanation += " Lifecycle note: " + structured_explanation["lifecycle_note"]
    all_detected_patterns = [
        _serialize_pattern_event(
            pattern,
            display_zone,
            df=pattern_df,
            pattern_config=config.pattern,
        )
        for pattern in ranked_patterns
    ]
    top_patterns = [
        _serialize_pattern_event(
            pattern,
            display_zone,
            df=pattern_df,
            pattern_config=config.pattern,
        )
        for pattern in top_patterns_internal
    ]

    warnings = list(quality_report.warnings)
    warnings.extend(context.warnings)
    warnings.extend(_collect_history_warnings(completed_df, active_registry))
    if latest_volume_baseline_source == "rolling_20":
        warnings.append(
            "Rolling 20-bar volume baseline used because time-of-day history was insufficient."
        )
    warnings.extend(_collect_analysis_validation_warnings(ranked_patterns, score))

    analysis_time_display = format_display_datetime(normalized_as_of, display_zone)
    exchange_timezone_name = context.exchange_timezone or _get_exchange_timezone(pattern_df)
    analysis_time_exchange = format_display_datetime(normalized_as_of, exchange_timezone_name)
    instrument_payload = context.instrument.to_dict()
    instrument_payload.setdefault("symbol", context.instrument.canonical_symbol)

    return {
        "instrument": instrument_payload,
        "analysis_context": context.to_dict(),
        "symbol": context.instrument.canonical_symbol.upper(),
        "as_of": format_iso_timestamp(normalized_as_of, timezone="UTC"),
        "analysis_time": analysis_time_display,
        "analysis_time_display": analysis_time_display,
        "analysis_time_exchange": analysis_time_exchange,
        "exchange_timezone": exchange_timezone_name,
        "display_timezone": display_timezone,
        "session_mode": context.session_mode,
        "included_segments": list(context.included_segments),
        "excluded_segments": [
            segment for segment in ("premarket", "regular", "afterhours")
            if segment not in context.included_segments
        ],
        "exchange_calendar": context.exchange_calendar,
        "latest_datetime": format_iso_timestamp(latest_bar_start_exchange),
        "latest_bar_start": format_display_datetime(latest_bar_start_display, display_zone),
        "latest_bar_end": format_display_datetime(latest_bar_end_display, display_zone),
        "latest_bar_start_exchange": format_display_datetime(latest_bar_start_exchange, exchange_timezone_name),
        "latest_bar_end_exchange": format_display_datetime(latest_bar_end_exchange, exchange_timezone_name),
        "latest_close": latest_close,
        "interval": interval,
        "trend": trend,
        "trend_score": trend_structure_score,
        "trend_signal_score": score["trend_score"],
        "trend_horizon": trend_horizon,
        "trend_lookback_bars": int(latest_row.get("Trend_Lookback_Bars", config.scoring.lookback_bars)),
        "short_term_trend": str(latest_row.get("Short_Term_Trend", trend)),
        "medium_term_trend": str(latest_row.get("Medium_Term_Trend", trend)),
        "long_term_trend": str(latest_row.get("Long_Term_Trend", trend)),
        "short_term_trend_score": round(float(latest_row.get("Short_Term_Trend_Score", trend_structure_score)), 2),
        "medium_term_trend_score": round(float(latest_row.get("Medium_Term_Trend_Score", trend_structure_score)), 2),
        "long_term_trend_score": round(float(latest_row.get("Long_Term_Trend_Score", trend_structure_score)), 2),
        "trend_evidence": trend_evidence,
        "pattern_score": score["pattern_score"],
        "volume_score": score["volume_score"],
        "bullish_score": score["bullish_score"],
        "bearish_score": score["bearish_score"],
        "net_signal_score": score["net_signal_score"],
        "rule_confidence": rule_confidence,
        "market_state": market_state,
        "overall_bias": overall_bias,
        "ignored_patterns_count": ignored_patterns_count,
        "top_patterns": top_patterns,
        "all_detected_patterns": all_detected_patterns,
        "current_relevant_patterns": current_relevant_patterns,
        "session_pattern_history": session_pattern_history,
        "session_history_total": len(session_pattern_history),
        "session_history_shown": len(session_pattern_history),
        "relevant_session": {
            "exchange_date": session_info["session_date"],
            "previous_exchange_date": session_info["previous_session_date"],
            "session_mode": context.session_mode,
            "included_segments": list(context.included_segments),
            "session_start_exchange": format_display_datetime(session_info["session_start"], session_info["exchange_timezone"]),
            "session_end_exchange": format_display_datetime(session_info["session_end"], session_info["exchange_timezone"]),
            "session_start_display": format_display_datetime(session_info["session_start"], display_zone),
            "session_end_display": format_display_datetime(session_info["session_end"], display_zone),
            "session_row_count": session_info["session_row_count"],
            "history_ordering": "chronological by detected_at",
        },
        "warnings": warnings,
        "data_quality_report": quality_report.to_dict(),
        "latest_volume_baseline_source": latest_volume_baseline_source,
        "market_data_metadata": metadata or {},
        "structured_explanation": structured_explanation,
        "explanation": explanation,
    }


def analyze_stock(
    symbol: str,
    period: str = "1mo",
    interval: str = "15m",
    as_of: Optional[pd.Timestamp] = None,
    lookback_bars: int = 12,
    top_pattern_count: int = 3,
    display_timezone: str = "Asia/Jerusalem",
    instrument: Optional[ResolvedInstrument] = None,
    provider: MarketDataProvider | None = None,
    data_file: str | None = None,
    exchange_timezone: str | None = None,
    cache_dir: str | None = None,
    cache_ttl: int = 3600,
    no_cache: bool = False,
    strict_data: bool = True,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    registry: PatternRegistry | None = None,
    include_extended_hours: bool = False,
    session_mode: str | None = DEFAULT_SESSION_MODE,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> dict[str, Any]:
    """Analyze one symbol using completed intraday candles only."""
    effective_session_mode = session_mode or ("extended" if include_extended_hours else "regular")
    request_context = build_analysis_context(
        symbol=symbol,
        interval=interval,
        display_timezone=display_timezone,
        session_mode=effective_session_mode,
        instrument=instrument,
        provider="file" if data_file else "yfinance",
        requested_period=period,
        requested_start=start,
        requested_end=end,
        exchange_timezone_override=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
        cache_config={
            "cache_dir": cache_dir,
            "cache_ttl": cache_ttl,
            "use_cache": not no_cache,
        },
    )
    if provider is None:
        if data_file:
            provider = FileDataProvider(data_file)
        else:
            provider = YFinanceProvider(
                config=MarketDataConfig(
                    timeout_seconds=timeout_seconds,
                    retry_attempts=retry_attempts,
                    cache_dir=cache_dir,
                    cache_ttl_seconds=cache_ttl,
                    use_cache=not no_cache,
                    strict_data=strict_data,
                    exchange_timezone=request_context.exchange_timezone,
                    include_extended_hours=request_context.include_extended_hours,
                    session_mode=request_context.session_mode,
                    regular_session_start=request_context.regular_session_start,
                    regular_session_end=request_context.regular_session_end,
                )
            )

    payload = provider.load(
        symbol=symbol,
        interval=interval,
        period=period,
        start=start,
        end=end,
        exchange_timezone=exchange_timezone,
        as_of=as_of,
        strict_data=strict_data,
        bypass_cache=no_cache,
        include_extended_hours=request_context.include_extended_hours,
        session_mode=request_context.session_mode,
        context=request_context,
    )
    return analyze_dataframe(
        df=payload.dataframe,
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        display_timezone=display_timezone,
        lookback_bars=lookback_bars,
        top_pattern_count=top_pattern_count,
        instrument=instrument,
        exchange_timezone=payload.exchange_timezone,
        strict_data=strict_data,
        data_quality_report=payload.quality_report,
        validate_data=False,
        metadata=payload.metadata,
        registry=registry,
        include_extended_hours=request_context.include_extended_hours,
        session_mode=request_context.session_mode,
        regular_session_start=request_context.regular_session_start,
        regular_session_end=request_context.regular_session_end,
    )
