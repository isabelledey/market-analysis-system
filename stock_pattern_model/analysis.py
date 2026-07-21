"""Core analysis entry points for stock pattern detection."""

from __future__ import annotations

from typing import Any
from typing import Optional

import pandas as pd

from stock_pattern_model.config import AnalysisConfig
from stock_pattern_model.config import MarketDataConfig
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
from stock_pattern_model.scoring import ScoringService


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
) -> tuple[pd.DataFrame, list[PatternEvent]]:
    pattern_df = classify_intraday_trend(
        add_features(df),
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
        detected_index = index_lookup.get(detected_bar_key)
        final_index = index_lookup.get(final_bar_key)
        if detected_index is None or final_index is None:
            continue

        metadata = details.get(event.pattern_id, {})
        detected_row = df.iloc[detected_index]
        candles_ago = latest_index - detected_index
        recency_weight = _get_recency_weight(candles_ago)
        weighted_score = round(float(event.base_score) * recency_weight, 2)
        strong_signal = bool(
            event.strength_label == "strong"
            or detected_row.get("Strong_Volume", False)
            or detected_row.get("Strong_Range", False)
        )
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
                "score_eligible": event.status is PatternStatus.CONFIRMED
                or (
                    score_tentative_patterns
                    and event.status is PatternStatus.TENTATIVE
                ),
            }
        )

    return prepared


def _serialize_pattern_event(
    pattern: dict[str, Any],
    display_timezone,
) -> dict[str, Any]:
    event: PatternEvent = pattern["event"]
    pattern_start_exchange = event.pattern_start_at
    pattern_end_exchange = event.pattern_end_at
    bar_start_exchange = event.bar_start_at
    bar_end_exchange = event.bar_end_at
    detected_at_exchange = event.detected_at

    pattern_start_display = convert_to_timezone(pattern_start_exchange, display_timezone)
    pattern_end_display = convert_to_timezone(pattern_end_exchange, display_timezone)
    bar_start_display = convert_to_timezone(bar_start_exchange, display_timezone)
    bar_end_display = convert_to_timezone(bar_end_exchange, display_timezone)
    detected_at_display = convert_to_timezone(detected_at_exchange, display_timezone)

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
        "pattern_start_at_utc": format_iso_timestamp(pattern_start_exchange, timezone="UTC"),
        "pattern_end_at_utc": format_iso_timestamp(pattern_end_exchange, timezone="UTC"),
        "bar_start_at_utc": format_iso_timestamp(bar_start_exchange, timezone="UTC"),
        "bar_end_at_utc": format_iso_timestamp(bar_end_exchange, timezone="UTC"),
        "detected_at_utc": format_iso_timestamp(detected_at_exchange, timezone="UTC"),
        "exchange_timezone": event.exchange_timezone,
        "display_timezone": str(display_timezone),
        "pattern_start_exchange": format_display_datetime(pattern_start_exchange, event.exchange_timezone),
        "pattern_end_exchange": format_display_datetime(pattern_end_exchange, event.exchange_timezone),
        "bar_start_exchange": format_display_datetime(bar_start_exchange, event.exchange_timezone),
        "bar_end_exchange": format_display_datetime(bar_end_exchange, event.exchange_timezone),
        "detected_at_exchange": format_display_datetime(detected_at_exchange, event.exchange_timezone),
        "pattern_start_display": format_display_datetime(pattern_start_display, display_timezone),
        "pattern_end_display": format_display_datetime(pattern_end_display, display_timezone),
        "bar_start_display": format_display_datetime(bar_start_display, display_timezone),
        "bar_end_display": format_display_datetime(bar_end_display, display_timezone),
        "detected_at_display": format_display_datetime(detected_at_display, display_timezone),
        "candles_ago": pattern["candles_ago"],
        "base_score": pattern["base_score"],
        "weighted_score": pattern["weighted_score"],
        "pattern_score_contribution": pattern["pattern_score_contribution"],
        "volume_score_contribution": pattern["volume_score_contribution"],
        "detection_reason": pattern["detection_reason"],
        "signal_strength": pattern["signal_strength"],
        "strength_label": pattern["strength_label"],
        "volume_baseline_source": pattern["volume_baseline_source"],
        "group_primary": pattern["group_primary"],
        "group_suppressed": pattern["group_suppressed"],
        "relevant_prices": event.relevant_prices,
        "relevant_indices": event.relevant_indices,
    }


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

    if validate_data:
        validated_df, quality_report = validate_market_data(
            df,
            interval=interval,
            exchange_timezone=exchange_timezone,
            as_of=as_of,
            strict_data=strict_data,
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
    pattern_df, raw_events = _run_pattern_pipeline(completed_df, config, active_registry)

    latest_row = pattern_df.iloc[-1]
    trend = str(latest_row["Trend"])
    trend_structure_score = round(float(latest_row.get("Trend_Score", 0.0)), 2)
    trend_evidence = list(latest_row.get("Trend_Evidence", []))
    trend_horizon = str(latest_row.get("Trend_Horizon", "Short-to-medium term"))
    resolved_events, ignored_patterns_count = resolve_pattern_conflicts(raw_events)
    prepared_patterns = _prepare_pattern_records(
        pattern_df,
        resolved_events,
        interval,
        active_registry,
        score_tentative_patterns=config.pattern.score_tentative_patterns,
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
    ranked_patterns = scoring_result["patterns"]
    top_patterns_internal = ranked_patterns[: config.scoring.top_pattern_count]
    all_detected_patterns = [
        _serialize_pattern_event(pattern, display_zone) for pattern in ranked_patterns
    ]
    top_patterns = [
        _serialize_pattern_event(pattern, display_zone) for pattern in top_patterns_internal
    ]

    resolved_instrument = instrument or ResolvedInstrument(
        input_identifier=symbol,
        symbol=symbol.upper(),
        name=symbol.upper(),
        exchange="Unknown",
        currency="Unknown",
        exchange_timezone=_get_exchange_timezone(pattern_df),
    )

    warnings = list(quality_report.warnings)
    if latest_volume_baseline_source == "rolling_20":
        warnings.append(
            "Rolling 20-bar volume baseline used because time-of-day history was insufficient."
        )

    analysis_time_display = format_display_datetime(normalized_as_of, display_zone)
    analysis_time_exchange = format_display_datetime(normalized_as_of, _get_exchange_timezone(pattern_df))

    return {
        "instrument": resolved_instrument.to_dict(),
        "symbol": resolved_instrument.symbol.upper(),
        "as_of": format_iso_timestamp(normalized_as_of, timezone="UTC"),
        "analysis_time": analysis_time_display,
        "analysis_time_display": analysis_time_display,
        "analysis_time_exchange": analysis_time_exchange,
        "exchange_timezone": _get_exchange_timezone(pattern_df),
        "display_timezone": display_timezone,
        "latest_datetime": format_iso_timestamp(latest_bar_start_exchange),
        "latest_bar_start": format_display_datetime(latest_bar_start_display, display_zone),
        "latest_bar_end": format_display_datetime(latest_bar_end_display, display_zone),
        "latest_bar_start_exchange": format_display_datetime(latest_bar_start_exchange, _get_exchange_timezone(pattern_df)),
        "latest_bar_end_exchange": format_display_datetime(latest_bar_end_exchange, _get_exchange_timezone(pattern_df)),
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
        "warnings": warnings,
        "data_quality_report": quality_report.to_dict(),
        "latest_volume_baseline_source": latest_volume_baseline_source,
        "market_data_metadata": metadata or {},
        "structured_explanation": scoring_result["structured_explanation"],
        "explanation": scoring_result["explanation"],
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
) -> dict[str, Any]:
    """Analyze one symbol using completed intraday candles only."""
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
                    exchange_timezone=exchange_timezone,
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
    )
