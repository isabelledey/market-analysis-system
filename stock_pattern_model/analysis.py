"""Core analysis entry points for stock pattern detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stock_pattern_model.config import (
    AnalysisConfig,
    MarketDataConfig,
    PatternConfig,
    ScoringConfig,
)
from stock_pattern_model.context import (
    AnalysisContext,
    build_analysis_context,
    dataframe_identity,
)
from stock_pattern_model.datetime_utils import (
    convert_to_timezone,
    format_display_datetime,
    format_iso_timestamp,
    interval_to_timedelta,
)
from stock_pattern_model.domain import (
    DataQualityReport,
    PatternEvent,
    PatternStatus,
    ResolvedInstrument,
)
from stock_pattern_model.exceptions import DataValidationError, NoCompletedBarsError
from stock_pattern_model.features import add_features
from stock_pattern_model.market_data import (
    FileDataProvider,
    MarketDataProvider,
    YFinanceProvider,
    validate_market_data,
)
from stock_pattern_model.pattern_detector import (
    DEFAULT_PATTERN_REGISTRY,
    PatternRegistry,
    classify_intraday_trend,
    classify_latest_candle_direction,
    classify_local_session_trend,
    deduplicate_structural_events,
    detect_patterns,
    resolve_pattern_conflicts,
)
from stock_pattern_model.scoring import (
    ScoringService,
    analytical_family,
    build_event_id,
    build_evidence_group,
    build_setup_id,
    pattern_max_age_bars,
    resolved_pattern_sort_key,
)
from stock_pattern_model.session_utils import (
    DEFAULT_REGULAR_SESSION_END,
    DEFAULT_REGULAR_SESSION_START,
    DEFAULT_SESSION_MODE,
    session_date_series,
)


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
    return interval_to_timedelta(interval)


def _get_bar_end(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    return timestamp + _get_bar_timedelta(interval)


def _normalize_as_of(as_of: pd.Timestamp | None) -> pd.Timestamp:
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
    as_of: pd.Timestamp | None,
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
    pattern_df = classify_local_session_trend(
        pattern_df,
        interval=config.interval,
        lookback_bars=config.pattern.local_trend_lookback_bars,
        pivot_left_bars=config.pattern.pivot_left_bars,
        pivot_right_bars=config.pattern.pivot_right_bars,
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
                "detector_label": str(metadata.get("label", event.pattern_name)),
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
                            "unconfirmed candidate geometry"
                            if event.status is PatternStatus.CANDIDATE
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
                    )
                ),
                "score_eligible": event.status is PatternStatus.CONFIRMED
                or (
                    score_tentative_patterns
                    and event.status is PatternStatus.TENTATIVE
                )
                # A dampener-eligible TENTATIVE rejection (Stage 4) is a coarse "plausibly
                # scoreable" candidate regardless of `score_tentative_patterns`; the precise
                # eligibility (pending dampener vs. directionally confirmed vs. invalidated) is
                # decided later by ScoringService.evaluate_pattern_eligibility using event_state.
                or bool(event.dampener_eligible),
                "pattern_entry_trend": event.pattern_entry_trend,
                "pattern_entry_trend_score": float(event.pattern_entry_trend_score),
                "pattern_entry_trend_lookback_bars": int(event.pattern_entry_trend_lookback_bars),
                "rejection_confirmation_state": event.rejection_confirmation_state,
                "dampener_eligible": bool(event.dampener_eligible),
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
    return pattern["pattern_id"] in {"breakout", "breakdown", "bullish_pin_bar", "shooting_star", "hammer"}


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
    if pattern["pattern_id"] in {"bullish_pin_bar", "hammer"}:
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


def _is_directionally_confirmed_candle(
    pattern: dict[str, Any],
    row: pd.Series,
    df: pd.DataFrame,
    config: PatternConfig,
) -> bool:
    """A later close beyond the rejection candle's own extreme directionally confirms a lower-
    wick rejection (Hammer / Bullish Pin Bar, close above the high) or an upper-wick rejection
    (Shooting Star / bearish continuation rejection, close below the low). Before this, the
    pattern stays a bounded, unconfirmed signal (see `dampener_eligible` in scoring.py); this is
    what promotes it. Merely failing to invalidate (Stage 5) is not, by itself, confirmation.
    """
    relevant_prices = pattern["event"].relevant_prices
    completion_index = _completion_reference_index(pattern)
    if pattern["pattern_id"] in {"hammer", "bullish_pin_bar"}:
        high = float(relevant_prices["high"])
        tolerance = _price_tolerance(df, completion_index, high, config)
        return bool(float(row["Close"]) > high + tolerance)
    if pattern["pattern_id"] == "shooting_star":
        low = float(relevant_prices["low"])
        tolerance = _price_tolerance(df, completion_index, low, config)
        return bool(float(row["Close"]) < low - tolerance)
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
    ranked = sorted(group_patterns, key=resolved_pattern_sort_key)
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
    confirmation_index: int | None = None

    for scan_index in range(completion_index + 1, len(df)):
        row = df.iloc[scan_index]
        if _is_invalidated_candle(pattern, row, df, config):
            invalidation_index = scan_index
            break
        if _is_directionally_confirmed_candle(pattern, row, df, config):
            confirmation_index = scan_index
            break
        if retest_index is None and _family_supports_retest(pattern) and _is_retest_candle(pattern, row, df, config):
            retest_index = scan_index

    if invalidation_index is not None:
        state = "invalidated"
        state_reference_index = invalidation_index
    elif confirmation_index is not None:
        state = "directionally_confirmed"
        state_reference_index = confirmation_index
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
        "confirmation_index": confirmation_index,
        "confirmed_at": (
            _transition_timestamp(df, confirmation_index, interval)
            if state == "directionally_confirmed"
            else None
        ),
        "expired_at": (
            _transition_timestamp(df, state_reference_index, interval)
            if state == "expired"
            else None
        ),
        "lifecycle_note": (
            "A later close above the rejection candle's high directionally confirmed the setup."
            if state == "directionally_confirmed"
            else None
        ),
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
            # detected_index (when the second pivot became confirmable, pivot_right_bars after
            # its own candle) must anchor the state timestamp, not setup_completion_index (the
            # pivot candle itself). Using setup_completion_index here made state_updated_at land
            # before detected_at, which is incoherent: the state cannot have "changed" before the
            # system was even able to recognize the pattern existed.
            state_reference_index = (
                int(primary["detected_index"])
                if primary.get("detected_index") is not None
                else int(primary.get("setup_completion_index"))
                if primary.get("setup_completion_index") is not None
                else _completion_reference_index(primary)
            )
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
            updated["directional_confirmation_index"] = lifecycle.get("confirmation_index")
            updated["confirmed_at"] = lifecycle.get("confirmed_at") if family_state == "directionally_confirmed" else None
            updated["expired_at"] = lifecycle["expired_at"] if family_state == "expired" else None
            updated["last_completed_candle_index"] = last_completed_index
            updated["last_completed_candle_at"] = _transition_timestamp(df, last_completed_index, interval)
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
    if pattern.get("cluster_suppressed"):
        return "clustered correlated evidence"
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


def _pattern_candle_summary(
    event: PatternEvent,
    df: pd.DataFrame,
    display_timezone,
) -> dict[str, Any] | None:
    """Original OHLC (and derived geometry ratios) for the candle this event is anchored to,
    read directly from the source DataFrame by index -- not from relevant_prices, so this is
    an independent cross-check of which candle a pattern actually refers to.

    Uses the last of ``relevant_indices`` (the candle whose close completed/confirmed the
    pattern), which for single-candle patterns like Shooting Star is that candle itself.
    """
    if not event.relevant_indices:
        return None
    anchor_index = int(event.relevant_indices[-1])
    if anchor_index < 0 or anchor_index >= len(df):
        return None
    row = df.iloc[anchor_index]
    open_price = float(row["Open"])
    high_price = float(row["High"])
    low_price = float(row["Low"])
    close_price = float(row["Close"])
    candle_range = high_price - low_price
    if candle_range > 1e-9:
        body_ratio = abs(close_price - open_price) / candle_range
        upper_wick_ratio = (high_price - max(open_price, close_price)) / candle_range
        lower_wick_ratio = (min(open_price, close_price) - low_price) / candle_range
        close_location = (close_price - low_price) / candle_range
    else:
        body_ratio = upper_wick_ratio = lower_wick_ratio = 0.0
        close_location = 0.5
    return {
        "index": anchor_index,
        "timestamp": format_display_datetime(pd.Timestamp(row["Datetime"]), display_timezone),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "body_ratio": round(body_ratio, 4),
        "upper_wick_ratio": round(upper_wick_ratio, 4),
        "lower_wick_ratio": round(lower_wick_ratio, 4),
        "close_location": round(close_location, 4),
    }


def _build_canonical_event_groups(
    patterns: list[dict[str, Any]],
    *,
    display_timezone,
    interval: str,
    df: pd.DataFrame,
    pattern_config: PatternConfig,
    scoring_config: ScoringConfig,
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
                str(pattern.get("detector_label") or pattern["pattern_name"])
                for pattern in sorted(
                    group_patterns,
                    key=lambda item: str(item.get("detector_label") or item["pattern_name"]),
                )
            )
        )
        raw_geometry_labels = list(
            dict.fromkeys(
                event.geometry_label
                for event in (pattern["event"] for pattern in group_patterns)
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
        pattern_score_contribution = round(float(primary.get("pattern_score_contribution", 0.0)), 2)
        volume_score_contribution = round(float(primary.get("volume_score_contribution", 0.0)), 2)
        combined_event_contribution = (
            round(pattern_score_contribution + volume_score_contribution, 2)
            if included_in_current_score
            else 0.0
        )
        invalidation_condition = _invalidation_condition_text(
            primary,
            df=df,
            config=pattern_config,
        )
        overlap_label_count = max(0, len(labels) - 1)
        state_reference_index = primary.get("state_reference_index")
        state_age_bars = (
            max(0, int(primary["last_completed_candle_index"]) - int(state_reference_index))
            if state_reference_index is not None
            else int(primary.get("score_anchor_candles_ago", primary["candles_ago"]))
        )
        canonical_events.append(
            {
                "event_id": f"canonical:{primary['evidence_group']}",
                "resolved_pattern_name": primary["pattern_name"],
                "resolved_bias": primary["bias"],
                "resolved_status": primary["status"],
                "resolved_context_quality": event.context_quality,
                "resolved_context": event.context_bias,
                "primary_pattern_name": primary["pattern_name"],
                "pattern_labels": labels,
                "matched_detector_labels": labels,
                "raw_geometry_labels": raw_geometry_labels,
                "family": primary["pattern_family"],
                "bias": primary["bias"],
                "status": primary["status"],
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
                "state_reference_index": int(state_reference_index) if state_reference_index is not None else None,
                "state_age_bars": state_age_bars,
                "score_age_bars": int(primary.get("score_anchor_candles_ago", primary["candles_ago"])),
                "score_max_age_bars": int(primary.get("score_max_age_bars", scoring_config.pattern_max_age_bars)),
                "state_expiration_bars": int(scoring_config.state_expiration_bars),
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
                "pattern_score_contribution": pattern_score_contribution,
                "volume_score_contribution": volume_score_contribution,
                "raw_volume_score_contribution": (
                    round(float(primary["raw_volume_score_contribution"]), 2)
                    if primary.get("raw_volume_score_contribution") is not None
                    else volume_score_contribution
                ),
                "volume_evidence_id": primary.get("volume_evidence_id"),
                "volume_deduplication_reason": primary.get("volume_deduplication_reason"),
                "combined_event_contribution": combined_event_contribution,
                "current_weighted_score": combined_event_contribution,
                "recency_weight": round(float(primary.get("recency_weight", 0.0)), 4),
                "evidence_group": primary["evidence_group"],
                "analytical_family": analytical_family(primary),
                "analytical_dependency_group": f"{analytical_family(primary)}:{primary['bias']}",
                "included_in_current_score": included_in_current_score,
                "score_eligible": bool(primary.get("score_eligible", False)),
                "exclusion_reason": inclusion_reason,
                "pattern_candle": _pattern_candle_summary(event, df, display_timezone),
                "geometry_label": event.geometry_label,
                "context_tags": list(event.context_tags),
                "context_bias": event.context_bias,
                "context_quality": event.context_quality,
                "detector_version": event.detector_version,
                "pattern_entry_trend": event.pattern_entry_trend,
                "pattern_entry_trend_score": float(event.pattern_entry_trend_score),
                "pattern_entry_trend_lookback_bars": int(event.pattern_entry_trend_lookback_bars),
                "cluster_suppressed": bool(primary.get("cluster_suppressed", False)),
                "cluster_id": primary.get("cluster_id"),
                "cluster_type": primary.get("cluster_type"),
                "cluster_member_ids": primary.get("cluster_member_ids") or [],
                "cluster_size": primary.get("cluster_size", 1),
                "cluster_price_zone": primary.get("cluster_price_zone"),
                "cluster_strongest_score": primary.get("cluster_strongest_score"),
                "cluster_repetition_bonus": primary.get("cluster_repetition_bonus"),
                "cluster_penalties_applied": primary.get("cluster_penalties_applied") or [],
                "cluster_bounded_contribution": primary.get("cluster_bounded_contribution"),
                "raw_pattern_score_contribution": primary.get("raw_pattern_score_contribution"),
                "rejection_confirmation_state": event.rejection_confirmation_state,
                "dampener_eligible": bool(event.dampener_eligible),
                "confirmed_at": (
                    format_iso_timestamp(primary["confirmed_at"])
                    if primary.get("confirmed_at") is not None
                    else None
                ),
                "geometry_status": primary.get("geometry_status", "Validated"),
                "context_status": primary.get("context_status", "Not Applicable"),
                "directional_confirmation": primary.get("directional_confirmation", "Not Required"),
                "follow_through": primary.get("follow_through", "Not Applicable"),
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
) -> list[dict[str, Any]]:
    session_history = [event for event in canonical_events if event["session_date"] == session_date]
    return session_history


def _current_display_sort_key(event: dict[str, Any]) -> tuple[object, ...]:
    return (
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
    )


def _is_current_display_relevant(event: dict[str, Any]) -> bool:
    state = str(event["state"])
    if state in {"expired", "invalidated", "failed", "failed_breakout", "failed_breakdown"}:
        return False
    if event["included_in_current_score"]:
        return event["score_age_bars"] <= event["score_max_age_bars"]
    return event["state_age_bars"] <= max(1, event["state_expiration_bars"])


def _build_evidence_collections(
    canonical_events: list[dict[str, Any]],
    *,
    overall_bias: str,
    session_date: str,
) -> dict[str, Any]:
    """Classify canonical events for display, keeping four distinct questions separate:

    1. Score eligibility (`score_eligible`) -- could this event participate in scoring at all.
    2. Inclusion in the current score (`included_in_current_score`) -- did it actually contribute
       a nonzero amount to the score just now.
    3. Direction relative to the final bias (`bias_aligned_evidence_count` /
       per-event `bias_aligned_with_overall`) -- only meaningful once a directional bias exists.
    4. Directional conflict between bullish and bearish evidence (`directional_conflict_present` /
       `directionally_conflicting_scored_evidence`) -- whether opposing scored evidence exists.

    Earlier, question 3/4 were conflated with question 2: any event on the "wrong" side of a
    conflict was moved out of `current_contributing_evidence` entirely, which for a Neutral overall
    bias (where *nothing* is bias-aligned) emptied that bucket even though every event was
    genuinely `included_in_current_score`. `current_contributing_evidence` now means exactly
    "score-eligible, included in the current score, nonzero contribution" -- full stop -- and
    conflict/alignment are reported as separate, additive facts about that same set.
    """
    contributing_directional = [
        event
        for event in canonical_events
        if event["included_in_current_score"]
        and abs(float(event["combined_event_contribution"])) > 0
        and event["bias"] in {"Bullish", "Bearish"}
        and event["score_eligible"]
        and _is_current_display_relevant(event)
    ]
    score_contributing_bullish_count = sum(
        1 for event in contributing_directional if event["bias"] == "Bullish"
    )
    score_contributing_bearish_count = sum(
        1 for event in contributing_directional if event["bias"] == "Bearish"
    )
    directional_conflict_present = (
        score_contributing_bullish_count > 0 and score_contributing_bearish_count > 0
    )
    bias_aligned_evidence_count = (
        sum(1 for event in contributing_directional if event["bias"] == overall_bias)
        if overall_bias in {"Bullish", "Bearish"}
        else None
    )
    if directional_conflict_present:
        if overall_bias in {"Bullish", "Bearish"}:
            # The bias that actually won is still bias-aligned and score-contributing; only the
            # minority side that pushed the other way is flagged as directionally conflicting.
            conflicting_event_ids = {
                event["event_id"] for event in contributing_directional if event["bias"] != overall_bias
            }
        else:
            # A Neutral overall bias has no aligned side at all, so every directional
            # score-contributing event is part of the conflict -- but they remain
            # score-contributing regardless; this only flags them as also being in conflict.
            conflicting_event_ids = {event["event_id"] for event in contributing_directional}
    else:
        conflicting_event_ids = set()

    categories: dict[str, list[dict[str, Any]]] = {
        "current_contributing_evidence": [],
        "awaiting_confirmation_evidence": [],
        "current_neutral_evidence": [],
        "recent_non_contributing_tracked_events": [],
        "historical_lifecycle_events": [],
    }
    category_by_event_id: dict[str, str] = {}

    for event in sorted(canonical_events, key=_current_display_sort_key):
        if not _is_current_display_relevant(event):
            category = "historical_lifecycle_events"
        elif (
            event["included_in_current_score"]
            and abs(float(event["combined_event_contribution"])) > 0
            and event["score_eligible"]
        ):
            category = "current_contributing_evidence"
        elif event["state"] == "awaiting_confirmation":
            category = "awaiting_confirmation_evidence"
        elif (
            event["bias"] == "Neutral"
            or event.get("context_quality") == "geometry_only"
            or event.get("status") == "candidate"
        ):
            category = "current_neutral_evidence"
        elif event["state"] in {"new", "active", "retested", "retest_pending", "retest_rejected", "reclaimed"}:
            category = "recent_non_contributing_tracked_events"
        else:
            category = "historical_lifecycle_events"

        is_conflicted = event["event_id"] in conflicting_event_ids
        bias_aligned_with_overall = (
            (event["bias"] == overall_bias) if overall_bias in {"Bullish", "Bearish"} else None
        )
        categories[category].append(
            dict(
                event,
                current_display_category=category,
                directionally_conflicted=is_conflicted,
                bias_aligned_with_overall=bias_aligned_with_overall,
            )
        )
        category_by_event_id[event["event_id"]] = category

    # A view over current_contributing_evidence, not a separate partition: every event here
    # also appears in current_contributing_evidence, tagged with directionally_conflicted=True.
    directionally_conflicting_scored_evidence = [
        event for event in categories["current_contributing_evidence"] if event["event_id"] in conflicting_event_ids
    ]

    historical_lifecycle = categories["historical_lifecycle_events"]
    relevant_session_detections = [event for event in canonical_events if event["session_date"] == session_date]
    return {
        **categories,
        "directionally_conflicting_scored_evidence": directionally_conflicting_scored_evidence,
        "score_contributing_bullish_count": score_contributing_bullish_count,
        "score_contributing_bearish_count": score_contributing_bearish_count,
        "bias_aligned_evidence_count": bias_aligned_evidence_count,
        "directional_conflict_present": directional_conflict_present,
        "relevant_session_detections": relevant_session_detections,
        "conflict_event_ids": sorted(conflicting_event_ids),
        "conflict_count": len(conflicting_event_ids),
        "current_relevant_patterns": sorted(
            (
                categories["current_contributing_evidence"]
                + categories["awaiting_confirmation_evidence"]
                + categories["current_neutral_evidence"]
                + categories["recent_non_contributing_tracked_events"]
            ),
            key=_current_display_sort_key,
        ),
        "current_relevant_patterns_deprecated": True,
        "historical_lifecycle_summary": {
            "count": len(historical_lifecycle),
            "by_state": {
                state: sum(1 for event in historical_lifecycle if event["state"] == state)
                for state in sorted({event["state"] for event in historical_lifecycle})
            },
            "by_family": {
                family: sum(1 for event in historical_lifecycle if event["family"] == family)
                for family in sorted({event["family"] for event in historical_lifecycle})
            },
        },
        "event_category_by_id": category_by_event_id,
        "overall_bias": overall_bias,
    }


def _build_explanation_sections(
    structured_explanation: dict[str, Any],
    *,
    evidence_collections: dict[str, Any],
    session_pattern_history: list[dict[str, Any]],
    latest_canonical_labels: list[str],
) -> dict[str, Any]:
    current_relevant_patterns = list(evidence_collections.get("current_relevant_patterns", []))
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
        session_context_lines.append(
            f"Total session lifecycle summary (including current events): {state_summary}."
        )
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

    conflicting_scored_evidence = evidence_collections.get("directionally_conflicting_scored_evidence") or []
    conflicts = []
    if evidence_collections.get("directional_conflict_present"):
        bullish_count = evidence_collections.get("score_contributing_bullish_count", 0)
        bearish_count = evidence_collections.get("score_contributing_bearish_count", 0)
        conflicts.append(
            f"Directional conflict present: {bullish_count} bullish and {bearish_count} bearish "
            "score-contributing event(s) disagree on direction."
        )
    conflict_count = int(evidence_collections.get("conflict_count", 0))
    if conflict_count and conflict_count != len(conflicting_scored_evidence):
        conflicts.append(
            f"Conflict accounting identified {conflict_count} conflicting canonical event(s)."
        )

    enriched = dict(structured_explanation)
    enriched["current_pattern_evidence"] = current_pattern_lines
    enriched["session_context"] = session_context_lines
    enriched["lifecycle_note"] = lifecycle_note
    enriched["conflicts"] = conflicts + list(structured_explanation.get("conflicts", []))
    enriched["current_display_summary"] = {
        "current_contributing_count": len(evidence_collections.get("current_contributing_evidence", [])),
        "score_contributing_bullish_count": evidence_collections.get("score_contributing_bullish_count", 0),
        "score_contributing_bearish_count": evidence_collections.get("score_contributing_bearish_count", 0),
        "bias_aligned_evidence_count": evidence_collections.get("bias_aligned_evidence_count"),
        "directional_conflict_present": evidence_collections.get("directional_conflict_present", False),
        "directionally_conflicting_count": len(conflicting_scored_evidence),
        "awaiting_confirmation_count": len(evidence_collections.get("awaiting_confirmation_evidence", [])),
        "current_neutral_count": len(evidence_collections.get("current_neutral_evidence", [])),
        "recent_non_contributing_count": len(evidence_collections.get("recent_non_contributing_tracked_events", [])),
    }
    return enriched


def _build_final_assessment(
    *,
    overall_bias: str,
    rule_confidence: float,
    trend: str,
    trend_structure_score: float | None,
    local_trend: str | None,
    local_trend_score: float | None,
    net_signal_score: float,
    structured_explanation: dict[str, Any],
) -> dict[str, Any]:
    """Combine the already-computed trend, pattern, and bias signals into one final
    technical-analysis label.

    This adds no new indicators -- it only weighs and summarizes signals the rest of the
    analysis already produced (overall_bias, rule_confidence, trend, local_trend,
    net_signal_score, and the bullish/bearish evidence lines). overall_bias is the base
    signal (it already combines confirmed-pattern evidence with recency/volume/trend
    context and is already gated on rule_confidence -- see ScoringService._derive_overall_bias
    and the minimum_bias_confidence gate around it), so it is not re-derived here. Only two
    additional, genuinely independent cross-checks are applied on top of it: whether bullish
    and bearish evidence are both currently contributing to the score, and whether the broad
    trend disagrees with the bias/local trend -- neither of which factors into overall_bias
    itself (see ScoringService._derive_overall_bias, which never reads `trend`).
    """
    bullish_pattern_lines = list(structured_explanation.get("bullish_evidence", []))
    bearish_pattern_lines = list(structured_explanation.get("bearish_evidence", []))
    display_summary = structured_explanation.get("current_display_summary", {})
    directional_conflict_present = bool(display_summary.get("directional_conflict_present"))
    trend_diverges = bool(
        local_trend
        and trend in {"Uptrend", "Downtrend"}
        and local_trend in {"Uptrend", "Downtrend"}
        and local_trend != trend
    )

    bullish_signals: list[str] = []
    bearish_signals: list[str] = []

    if trend == "Uptrend":
        suffix = f" (score {trend_structure_score:.2f})" if trend_structure_score is not None else ""
        bullish_signals.append(f"Broad trend is Uptrend{suffix}.")
    elif trend == "Downtrend":
        suffix = f" (score {trend_structure_score:.2f})" if trend_structure_score is not None else ""
        bearish_signals.append(f"Broad trend is Downtrend{suffix}.")

    if local_trend == "Uptrend":
        suffix = f" (score {local_trend_score:.2f})" if local_trend_score is not None else ""
        bullish_signals.append(f"Local session trend is Uptrend{suffix}.")
    elif local_trend == "Downtrend":
        suffix = f" (score {local_trend_score:.2f})" if local_trend_score is not None else ""
        bearish_signals.append(f"Local session trend is Downtrend{suffix}.")

    bullish_signals.extend(bullish_pattern_lines)
    bearish_signals.extend(bearish_pattern_lines)

    if net_signal_score > 0:
        bullish_signals.append(f"Net signal score is positive ({net_signal_score:.2f}).")
    elif net_signal_score < 0:
        bearish_signals.append(f"Net signal score is negative ({net_signal_score:.2f}).")

    if overall_bias == "Bullish":
        recommendation = "RECOMMEND TO BUY"
    elif overall_bias == "Bearish":
        recommendation = "NOT RECOMMENDED"
    else:
        recommendation = "NEUTRAL"

    downgrade_reasons: list[str] = []
    if recommendation != "NEUTRAL":
        if directional_conflict_present:
            downgrade_reasons.append("bullish and bearish evidence are both currently contributing to the score")
        if recommendation == "RECOMMEND TO BUY" and trend == "Downtrend":
            downgrade_reasons.append("the broader trend is Downtrend, opposing the bias")
        if recommendation == "NOT RECOMMENDED" and trend == "Uptrend":
            downgrade_reasons.append("the broader trend is Uptrend, opposing the bias")
        if trend_diverges:
            downgrade_reasons.append(
                f"the broad trend ({trend}) and local session trend ({local_trend}) disagree"
            )
        if downgrade_reasons:
            recommendation = "NEUTRAL"

    if rule_confidence >= 60.0:
        confidence_level = "HIGH"
    elif rule_confidence >= 30.0:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    if downgrade_reasons and confidence_level == "HIGH":
        # A downgraded, mixed read is inherently less certain than the raw rule-confidence
        # number alone would suggest.
        confidence_level = "MEDIUM"

    if recommendation == "RECOMMEND TO BUY":
        reasoning = (
            f"Overall bias is Bullish with {rule_confidence:.1f}/100 rule confidence, broad "
            f"trend is {trend}, and {len(bullish_pattern_lines)} confirmed bullish pattern(s) "
            f"outweighed {len(bearish_pattern_lines)} bearish."
        )
    elif recommendation == "NOT RECOMMENDED":
        reasoning = (
            f"Overall bias is Bearish with {rule_confidence:.1f}/100 rule confidence, broad "
            f"trend is {trend}, and {len(bearish_pattern_lines)} confirmed bearish pattern(s) "
            f"outweighed {len(bullish_pattern_lines)} bullish."
        )
    elif downgrade_reasons:
        reasoning = (
            f"Overall bias was {overall_bias}, but the read was downgraded to Neutral because "
            + "; ".join(downgrade_reasons) + "."
        )
    else:
        reasoning = (
            f"Signals were mixed or insufficient for a directional call: overall bias is "
            f"{overall_bias}, broad trend is {trend}, with {len(bullish_pattern_lines)} bullish "
            f"and {len(bearish_pattern_lines)} bearish confirmed pattern(s)."
        )

    return {
        "recommendation": recommendation,
        "confidence_level": confidence_level,
        "reasoning": reasoning,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "disclaimer": (
            "This assessment is based on technical analysis of the available data and is not "
            "financial advice."
        ),
    }


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
        "raw_volume_score_contribution": (
            pattern["raw_volume_score_contribution"]
            if pattern.get("raw_volume_score_contribution") is not None
            else pattern["volume_score_contribution"]
        ),
        "volume_evidence_id": pattern.get("volume_evidence_id"),
        "volume_deduplication_reason": pattern.get("volume_deduplication_reason"),
        "combined_score_contribution": pattern.get("combined_score_contribution", 0.0),
        "detection_reason": pattern["detection_reason"],
        "signal_strength": pattern["signal_strength"],
        "strength_label": pattern["strength_label"],
        "volume_baseline_source": pattern["volume_baseline_source"],
        "detector_label": pattern.get("detector_label", event.pattern_name),
        "pattern_candle": _pattern_candle_summary(event, df, display_timezone),
        "geometry_label": event.geometry_label,
        "context_tags": list(event.context_tags),
        "context_bias": event.context_bias,
        "context_quality": event.context_quality,
        "detector_version": event.detector_version,
        "pattern_entry_trend": event.pattern_entry_trend,
        "pattern_entry_trend_score": float(event.pattern_entry_trend_score),
        "pattern_entry_trend_lookback_bars": int(event.pattern_entry_trend_lookback_bars),
        "cluster_suppressed": bool(pattern.get("cluster_suppressed", False)),
        "cluster_id": pattern.get("cluster_id"),
        "cluster_type": pattern.get("cluster_type"),
        "cluster_member_ids": pattern.get("cluster_member_ids") or [],
        "cluster_size": pattern.get("cluster_size", 1),
        "cluster_price_zone": pattern.get("cluster_price_zone"),
        "cluster_strongest_score": pattern.get("cluster_strongest_score"),
        "cluster_repetition_bonus": pattern.get("cluster_repetition_bonus"),
        "cluster_penalties_applied": pattern.get("cluster_penalties_applied") or [],
        "cluster_bounded_contribution": pattern.get("cluster_bounded_contribution"),
        "raw_pattern_score_contribution": pattern.get("raw_pattern_score_contribution"),
        "rejection_confirmation_state": event.rejection_confirmation_state,
        "dampener_eligible": bool(event.dampener_eligible),
        "confirmed_at": (
            format_iso_timestamp(pattern["confirmed_at"])
            if pattern.get("confirmed_at") is not None
            else None
        ),
        "geometry_status": pattern.get("geometry_status", "Validated"),
        "context_status": pattern.get("context_status", "Not Applicable"),
        "directional_confirmation": pattern.get("directional_confirmation", "Not Required"),
        "follow_through": pattern.get("follow_through", "Not Applicable"),
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
    as_of: pd.Timestamp | None = None,
    display_timezone: str = "Asia/Jerusalem",
    lookback_bars: int = 12,
    top_pattern_count: int = 3,
    instrument: ResolvedInstrument | None = None,
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
    trend_evidence_structured = list(latest_row.get("Trend_Evidence_Structured", []))
    trend_horizon = str(latest_row.get("Trend_Horizon", "Short-to-medium term"))
    local_trend = str(latest_row.get("Local_Trend", "Neutral"))
    local_trend_score = round(float(latest_row.get("Local_Trend_Score", 0.0)), 2)
    local_trend_evidence = list(latest_row.get("Local_Trend_Evidence", []))
    local_trend_lookback_bars = int(
        latest_row.get("Local_Trend_Lookback_Bars", config.pattern.local_trend_lookback_bars)
    )
    latest_candle_direction, latest_candle_direction_score = classify_latest_candle_direction(latest_row)
    session_info = _latest_completed_session_info(pattern_df, interval, context=context)
    resolved_events, ignored_patterns_count = resolve_pattern_conflicts(raw_events)
    resolved_events, duplicate_structural_count = deduplicate_structural_events(
        resolved_events,
        price_tolerance_ratio=config.pattern.structural_duplicate_price_tolerance,
    )
    ignored_patterns_count += duplicate_structural_count
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
        trend_evidence_structured=trend_evidence_structured,
        trend_horizon=trend_horizon,
        local_trend=local_trend,
        local_trend_score=local_trend_score,
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
        scoring_config=config.scoring,
    )
    session_pattern_history = _build_session_context(
        canonical_events,
        session_info["session_date"],
    )
    evidence_collections = _build_evidence_collections(
        canonical_events,
        overall_bias=overall_bias,
        session_date=session_info["session_date"],
    )
    current_relevant_patterns = evidence_collections["current_relevant_patterns"]
    latest_canonical_labels = current_relevant_patterns[0]["pattern_labels"] if current_relevant_patterns else []
    structured_explanation = _build_explanation_sections(
        scoring_result["structured_explanation"],
        evidence_collections=evidence_collections,
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
    final_assessment = _build_final_assessment(
        overall_bias=overall_bias,
        rule_confidence=rule_confidence,
        trend=trend,
        trend_structure_score=trend_structure_score,
        local_trend=local_trend,
        local_trend_score=local_trend_score,
        net_signal_score=score["net_signal_score"],
        structured_explanation=structured_explanation,
    )
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
        "broad_trend": trend,
        "broad_trend_score": trend_structure_score,
        "local_trend": local_trend,
        "local_trend_score": local_trend_score,
        "local_trend_evidence": local_trend_evidence,
        "local_trend_lookback_bars": local_trend_lookback_bars,
        "latest_candle_direction": latest_candle_direction,
        "latest_candle_direction_score": latest_candle_direction_score,
        "short_term_trend": str(latest_row.get("Short_Term_Trend", trend)),
        "medium_term_trend": str(latest_row.get("Medium_Term_Trend", trend)),
        "long_term_trend": str(latest_row.get("Long_Term_Trend", trend)),
        "short_term_trend_score": round(float(latest_row.get("Short_Term_Trend_Score", trend_structure_score)), 2),
        "medium_term_trend_score": round(float(latest_row.get("Medium_Term_Trend_Score", trend_structure_score)), 2),
        "long_term_trend_score": round(float(latest_row.get("Long_Term_Trend_Score", trend_structure_score)), 2),
        "trend_evidence": trend_evidence,
        "trend_evidence_structured": trend_evidence_structured,
        "pattern_score": score["pattern_score"],
        "volume_score": score["volume_score"],
        "bullish_pattern_score": score["bullish_pattern_score"],
        "bearish_pattern_score": score["bearish_pattern_score"],
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
        "current_relevant_patterns_deprecated": evidence_collections["current_relevant_patterns_deprecated"],
        "session_pattern_history": session_pattern_history,
        **evidence_collections,
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
        "final_assessment": final_assessment,
    }


def analyze_stock(
    symbol: str,
    period: str = "1mo",
    interval: str = "15m",
    as_of: pd.Timestamp | None = None,
    lookback_bars: int = 12,
    top_pattern_count: int = 3,
    display_timezone: str = "Asia/Jerusalem",
    instrument: ResolvedInstrument | None = None,
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
