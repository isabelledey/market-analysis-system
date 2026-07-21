"""Leakage-free historical signal evaluation and backtesting helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from statistics import mean
from statistics import median
from typing import Any

import pandas as pd

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.config import HistoricalEvaluationConfig
from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.context import build_analysis_context
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import HistoricalEvaluationResult
from stock_pattern_model.domain import HistoricalPerformanceSummary
from stock_pattern_model.domain import HistoricalSignalOutcome
from stock_pattern_model.domain import HistoricalSignalRecord
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.market_data import FileDataProvider
from stock_pattern_model.market_data import MarketDataProvider
from stock_pattern_model.market_data import YFinanceProvider
from stock_pattern_model.market_data import validate_market_data
from stock_pattern_model.pattern_detector import DEFAULT_PATTERN_REGISTRY
from stock_pattern_model.pattern_detector import PatternRegistry
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_END
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_START
from stock_pattern_model.session_utils import DEFAULT_SESSION_MODE
from stock_pattern_model.session_utils import session_segment_series


def _get_bar_timedelta(interval: str) -> pd.Timedelta:
    return pd.to_timedelta(interval)


def _get_bar_end(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    return timestamp + _get_bar_timedelta(interval)


def _normalize_as_of(as_of: pd.Timestamp | None) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz="UTC")
    normalized = pd.Timestamp(as_of)
    if normalized.tzinfo is None:
        raise DataValidationError("Historical evaluation as_of must be timezone-aware.")
    return normalized


def _get_exchange_timezone(df: pd.DataFrame) -> str:
    datetimes = pd.to_datetime(df["Datetime"])
    timezone = datetimes.dt.tz
    if timezone is None:
        raise DataValidationError("Historical evaluation requires timezone-aware Datetime values.")
    return str(timezone)


def _empty_quality_report(row_count: int) -> DataQualityReport:
    return DataQualityReport(
        row_count=row_count,
        completed_row_count=row_count,
        duplicate_count=0,
        missing_value_count=0,
        invalid_ohlc_count=0,
        irregular_gap_count=0,
        warnings=[],
        cleaning_actions=[],
    )


def _filter_completed_candles(
    df: pd.DataFrame,
    *,
    interval: str,
    as_of: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    normalized_as_of = _normalize_as_of(as_of)
    filtered_df = df.copy()
    filtered_df["Datetime"] = pd.to_datetime(filtered_df["Datetime"])
    filtered_df["Bar_End"] = filtered_df["Datetime"] + _get_bar_timedelta(interval)
    filtered_df = filtered_df.loc[filtered_df["Bar_End"] <= normalized_as_of].copy()
    filtered_df = filtered_df.drop(columns=["Bar_End"]).reset_index(drop=True)
    if filtered_df.empty:
        raise DataValidationError(
            f"No completed {interval} candles are available for historical evaluation as of "
            f"{normalized_as_of.isoformat(timespec='minutes')}."
        )
    return filtered_df, normalized_as_of


def _confidence_bucket(confidence: float, edges: tuple[float, ...]) -> str:
    lower_bound = 0
    for edge in edges:
        if confidence < edge:
            return f"{int(lower_bound)}-{int(edge) - 1}"
        lower_bound = int(edge)
    return f"{int(lower_bound)}-100"


def _session_segment_for_timestamp(
    timestamp: pd.Timestamp,
    *,
    exchange_timezone: str,
    regular_session_start: str,
    regular_session_end: str,
) -> str:
    segment = session_segment_series(
        pd.Series([timestamp]),
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    ).iloc[0]
    return str(segment)


def _target_stop_outcome(
    future_slice: pd.DataFrame,
    *,
    entry_price: float,
    bias: str,
    target_return: float,
    stop_return: float,
    same_bar_touch_policy: str,
    start_index: int,
) -> tuple[str, int | None, bool, bool, float | None]:
    if future_slice.empty:
        return "insufficient_future_bars", None, False, False, None

    if bias == "Bullish":
        target_price = entry_price * (1 + target_return)
        stop_price = entry_price * (1 - stop_return)
    else:
        target_price = entry_price * (1 - target_return)
        stop_price = entry_price * (1 + stop_return)

    for offset, (_, row) in enumerate(future_slice.iterrows(), start=1):
        high_price = float(row["High"])
        low_price = float(row["Low"])
        global_index = start_index + offset
        if bias == "Bullish":
            target_hit = high_price >= target_price
            stop_hit = low_price <= stop_price
        else:
            target_hit = low_price <= target_price
            stop_hit = high_price >= stop_price

        if target_hit and stop_hit:
            if same_bar_touch_policy == "target_first":
                return "target_first", global_index, True, True, target_return
            if same_bar_touch_policy == "stop_first":
                return "stop_first", global_index, True, True, -stop_return
            return "both_same_bar", global_index, True, True, None
        if target_hit:
            return "target_first", global_index, True, False, target_return
        if stop_hit:
            return "stop_first", global_index, False, True, -stop_return

    return "neither_hit", None, False, False, None


def _compute_outcomes_for_signal(
    history: pd.DataFrame,
    *,
    signal: HistoricalSignalRecord,
    interval: str,
    config: HistoricalEvaluationConfig,
) -> list[HistoricalSignalOutcome]:
    outcomes: list[HistoricalSignalOutcome] = []
    entry_price = float(signal.entry_price)

    for horizon in config.horizons_bars:
        exit_index = signal.detected_index + horizon
        if exit_index >= len(history):
            outcomes.append(
                HistoricalSignalOutcome(
                    horizon_bars=int(horizon),
                    future_bar_count=max(0, len(history) - signal.detected_index - 1),
                    available=False,
                    exit_index=None,
                    exit_at=None,
                    exit_close=None,
                    raw_forward_return=None,
                    directional_forward_return=None,
                    mfe_return=None,
                    mae_return=None,
                    direction_correct=None,
                    target_price=(
                        entry_price * (1 + config.target_return)
                        if signal.bias == "Bullish"
                        else entry_price * (1 - config.target_return)
                    ),
                    stop_price=(
                        entry_price * (1 - config.stop_return)
                        if signal.bias == "Bullish"
                        else entry_price * (1 + config.stop_return)
                    ),
                    target_hit=False,
                    stop_hit=False,
                    first_touch="insufficient_future_bars",
                    first_touch_index=None,
                    simulated_trade_return=None,
                )
            )
            continue

        future_slice = history.iloc[signal.detected_index + 1 : exit_index + 1].copy()
        exit_row = history.iloc[exit_index]
        exit_close = float(exit_row["Close"])
        raw_forward_return = (exit_close / entry_price) - 1
        directional_forward_return = (
            raw_forward_return if signal.bias == "Bullish" else -raw_forward_return
        )

        highs = future_slice["High"].astype(float)
        lows = future_slice["Low"].astype(float)
        if signal.bias == "Bullish":
            mfe_return = max(((highs / entry_price) - 1).tolist())
            mae_return = min(((lows / entry_price) - 1).tolist())
            target_price = entry_price * (1 + config.target_return)
            stop_price = entry_price * (1 - config.stop_return)
        else:
            mfe_return = max(((entry_price - lows) / entry_price).tolist())
            mae_return = min(((entry_price - highs) / entry_price).tolist())
            target_price = entry_price * (1 - config.target_return)
            stop_price = entry_price * (1 + config.stop_return)

        first_touch, first_touch_index, target_hit, stop_hit, trade_return = _target_stop_outcome(
            future_slice,
            entry_price=entry_price,
            bias=signal.bias,
            target_return=config.target_return,
            stop_return=config.stop_return,
            same_bar_touch_policy=config.same_bar_touch_policy,
            start_index=signal.detected_index,
        )
        simulated_trade_return = (
            directional_forward_return if trade_return is None else trade_return
        )

        outcomes.append(
            HistoricalSignalOutcome(
                horizon_bars=int(horizon),
                future_bar_count=int(horizon),
                available=True,
                exit_index=exit_index,
                exit_at=_get_bar_end(pd.Timestamp(exit_row["Datetime"]), interval),
                exit_close=round(exit_close, 6),
                raw_forward_return=round(raw_forward_return, 6),
                directional_forward_return=round(directional_forward_return, 6),
                mfe_return=round(float(mfe_return), 6),
                mae_return=round(float(mae_return), 6),
                direction_correct=directional_forward_return > 0,
                target_price=round(float(target_price), 6),
                stop_price=round(float(stop_price), 6),
                target_hit=target_hit,
                stop_hit=stop_hit,
                first_touch=first_touch,
                first_touch_index=first_touch_index,
                simulated_trade_return=round(float(simulated_trade_return), 6),
            )
        )

    return outcomes


def _summary_from_observations(
    observations: list[dict[str, Any]],
    *,
    horizon_bars: int,
) -> HistoricalPerformanceSummary:
    if not observations:
        return HistoricalPerformanceSummary(
            horizon_bars=horizon_bars,
            evaluated_signals=0,
            wins=0,
            losses=0,
            flat=0,
            direction_correct_rate=None,
            precision=None,
            false_positive_rate=None,
            win_rate=None,
            average_forward_return=None,
            median_forward_return=None,
            average_raw_forward_return=None,
            median_raw_forward_return=None,
            average_mfe_return=None,
            median_mfe_return=None,
            average_mae_return=None,
            median_mae_return=None,
            expectancy=None,
            target_first_rate=None,
            stop_first_rate=None,
            neither_hit_rate=None,
            ambiguous_same_bar_rate=None,
        )

    directional_returns = [float(item["directional_forward_return"]) for item in observations]
    raw_returns = [float(item["raw_forward_return"]) for item in observations]
    mfe_returns = [float(item["mfe_return"]) for item in observations]
    mae_returns = [float(item["mae_return"]) for item in observations]
    expectancy_returns = [float(item["simulated_trade_return"]) for item in observations]
    wins = sum(1 for value in directional_returns if value > 0)
    losses = sum(1 for value in directional_returns if value < 0)
    flat = len(observations) - wins - losses
    trade_wins = sum(1 for value in expectancy_returns if value > 0)

    def _rate(count: int) -> float:
        return round(count / len(observations), 6)

    return HistoricalPerformanceSummary(
        horizon_bars=horizon_bars,
        evaluated_signals=len(observations),
        wins=wins,
        losses=losses,
        flat=flat,
        direction_correct_rate=round(mean(1.0 if item["direction_correct"] else 0.0 for item in observations), 6),
        precision=_rate(wins),
        false_positive_rate=_rate(losses),
        win_rate=_rate(trade_wins),
        average_forward_return=round(mean(directional_returns), 6),
        median_forward_return=round(median(directional_returns), 6),
        average_raw_forward_return=round(mean(raw_returns), 6),
        median_raw_forward_return=round(median(raw_returns), 6),
        average_mfe_return=round(mean(mfe_returns), 6),
        median_mfe_return=round(median(mfe_returns), 6),
        average_mae_return=round(mean(mae_returns), 6),
        median_mae_return=round(median(mae_returns), 6),
        expectancy=round(mean(expectancy_returns), 6),
        target_first_rate=_rate(sum(1 for item in observations if item["first_touch"] == "target_first")),
        stop_first_rate=_rate(sum(1 for item in observations if item["first_touch"] == "stop_first")),
        neither_hit_rate=_rate(sum(1 for item in observations if item["first_touch"] == "neither_hit")),
        ambiguous_same_bar_rate=_rate(sum(1 for item in observations if item["first_touch"] == "both_same_bar")),
    )


def _flatten_outcomes(signals: list[HistoricalSignalRecord]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for signal in signals:
        for outcome in signal.outcomes:
            if not outcome.available:
                continue
            observations.append(
                {
                    "signal_id": signal.signal_id,
                    "pattern_name": signal.pattern_name,
                    "market_context": f"{signal.trend} | {signal.market_state} | {signal.overall_bias}",
                    "session_time": f"{signal.session_segment} | {signal.session_time_exchange}",
                    "confidence_bucket": signal.signal_confidence_bucket,
                    "horizon_bars": outcome.horizon_bars,
                    "direction_correct": bool(outcome.direction_correct),
                    "raw_forward_return": float(outcome.raw_forward_return),
                    "directional_forward_return": float(outcome.directional_forward_return),
                    "mfe_return": float(outcome.mfe_return),
                    "mae_return": float(outcome.mae_return),
                    "first_touch": outcome.first_touch,
                    "simulated_trade_return": float(outcome.simulated_trade_return),
                }
            )
    return observations


def _build_grouped_summaries(
    observations: list[dict[str, Any]],
    *,
    group_field: str,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, HistoricalPerformanceSummary]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        grouped[str(observation[group_field])][int(observation["horizon_bars"])].append(observation)

    summaries: dict[str, dict[str, HistoricalPerformanceSummary]] = {}
    for group_value, horizon_groups in grouped.items():
        summaries[group_value] = {
            str(horizon): _summary_from_observations(
                horizon_groups.get(horizon, []),
                horizon_bars=horizon,
            )
            for horizon in horizons
        }
    return summaries


def _validate_and_prepare_history(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    exchange_timezone: str | None,
    as_of: pd.Timestamp | None,
    strict_data: bool,
    validate_data: bool,
    data_quality_report: DataQualityReport | None,
    include_extended_hours: bool,
    session_mode: str | None,
    regular_session_start: str,
    regular_session_end: str,
) -> tuple[pd.DataFrame, DataQualityReport, pd.Timestamp]:
    effective_session_mode = session_mode or ("extended" if include_extended_hours else "regular")
    context = build_analysis_context(
        symbol=symbol,
        interval=interval,
        display_timezone="Asia/Jerusalem",
        session_mode=effective_session_mode,
        provider="historical-evaluation",
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
        quality_report = data_quality_report or _empty_quality_report(len(validated_df))

    completed_df, normalized_as_of = _filter_completed_candles(
        validated_df,
        interval=interval,
        as_of=as_of,
    )
    quality_report = replace(quality_report, completed_row_count=len(completed_df))
    return completed_df, quality_report, normalized_as_of


def _collect_signal_records(
    history: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    display_timezone: str,
    lookback_bars: int,
    exchange_timezone: str,
    strict_data: bool,
    registry: PatternRegistry,
    evaluation_config: HistoricalEvaluationConfig,
    include_extended_hours: bool,
    session_mode: str | None,
    regular_session_start: str,
    regular_session_end: str,
) -> list[HistoricalSignalRecord]:
    signals: list[HistoricalSignalRecord] = []
    seen_event_ids: set[str] = set()

    for index in range(len(history)):
        if index + 1 < evaluation_config.minimum_history_bars:
            continue

        prefix_df = history.iloc[: index + 1].copy()
        prefix_as_of = _get_bar_end(pd.Timestamp(prefix_df.iloc[-1]["Datetime"]), interval)
        result = analyze_dataframe(
            df=prefix_df,
            symbol=symbol,
            interval=interval,
            as_of=prefix_as_of,
            display_timezone=display_timezone,
            lookback_bars=lookback_bars,
            top_pattern_count=10,
            exchange_timezone=exchange_timezone,
            strict_data=strict_data,
            data_quality_report=_empty_quality_report(len(prefix_df)),
            validate_data=False,
            registry=registry,
            include_extended_hours=include_extended_hours,
            session_mode=session_mode,
            regular_session_start=regular_session_start,
            regular_session_end=regular_session_end,
        )

        for pattern in result["all_detected_patterns"]:
            if pattern["bias"] not in {"Bullish", "Bearish"}:
                continue
            if int(pattern["detected_index"]) != index:
                continue
            if not bool(pattern.get("group_primary")):
                continue
            if evaluation_config.only_score_eligible_signals and not bool(pattern.get("included_in_current_score")):
                continue
            event_id = str(pattern["event_id"])
            if event_id in seen_event_ids:
                continue

            bar_start = pd.Timestamp(pattern["bar_start_at"])
            signal_row = history.iloc[index]
            session_time_exchange = bar_start.tz_convert(exchange_timezone).strftime("%H:%M")
            session_segment = _session_segment_for_timestamp(
                bar_start,
                exchange_timezone=exchange_timezone,
                regular_session_start=regular_session_start,
                regular_session_end=regular_session_end,
            )
            confidence = float(result["rule_confidence"])
            signals.append(
                HistoricalSignalRecord(
                    signal_id=f"{event_id}:{index}",
                    event_id=event_id,
                    setup_id=str(pattern["setup_id"]),
                    evidence_group=str(pattern["evidence_group"]),
                    symbol=symbol,
                    interval=interval,
                    pattern_id=str(pattern["pattern_id"]),
                    pattern_name=str(pattern["pattern_name"]),
                    pattern_family=str(pattern["pattern_family"]),
                    bias=str(pattern["bias"]),
                    status=str(pattern["status"]),
                    detected_at=pd.Timestamp(pattern["detected_at"]),
                    bar_start_at=bar_start,
                    detected_index=int(pattern["detected_index"]),
                    entry_price=float(signal_row["Close"]),
                    trend=str(result["trend"]),
                    market_state=str(result["market_state"]),
                    overall_bias=str(result["overall_bias"]),
                    rule_confidence=round(confidence, 6),
                    signal_confidence_bucket=_confidence_bucket(
                        confidence,
                        evaluation_config.confidence_bucket_edges,
                    ),
                    exchange_timezone=exchange_timezone,
                    display_timezone=display_timezone,
                    session_segment=session_segment,
                    session_time_exchange=session_time_exchange,
                    signal_strength=float(pattern["signal_strength"]),
                    strength_label=str(pattern["strength_label"]),
                    volume_baseline_source=str(pattern["volume_baseline_source"]),
                    outcomes=[],
                )
            )
            seen_event_ids.add(event_id)

    return signals


def collect_historical_signals_from_dataframe(
    df: pd.DataFrame,
    *,
    symbol: str = "DATAFRAME",
    interval: str = "15m",
    as_of: pd.Timestamp | None = None,
    display_timezone: str = "Asia/Jerusalem",
    lookback_bars: int = 12,
    exchange_timezone: str | None = None,
    strict_data: bool = True,
    validate_data: bool = True,
    data_quality_report: DataQualityReport | None = None,
    registry: PatternRegistry | None = None,
    evaluation_config: HistoricalEvaluationConfig | None = None,
    include_extended_hours: bool = True,
    session_mode: str | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> list[HistoricalSignalRecord]:
    """Collect historically knowable signals without using future data."""

    config = evaluation_config or HistoricalEvaluationConfig()
    config.validate()
    active_registry = registry or DEFAULT_PATTERN_REGISTRY
    completed_df, _, _ = _validate_and_prepare_history(
        df,
        symbol=symbol,
        interval=interval,
        exchange_timezone=exchange_timezone,
        as_of=as_of,
        strict_data=strict_data,
        validate_data=validate_data,
        data_quality_report=data_quality_report,
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    active_exchange_timezone = exchange_timezone or _get_exchange_timezone(completed_df)
    return _collect_signal_records(
        completed_df,
        symbol=symbol,
        interval=interval,
        display_timezone=display_timezone,
        lookback_bars=lookback_bars,
        exchange_timezone=active_exchange_timezone,
        strict_data=strict_data,
        registry=active_registry,
        evaluation_config=config,
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )


def evaluate_historical_dataframe(
    df: pd.DataFrame,
    *,
    symbol: str = "DATAFRAME",
    interval: str = "15m",
    as_of: pd.Timestamp | None = None,
    display_timezone: str = "Asia/Jerusalem",
    lookback_bars: int = 12,
    exchange_timezone: str | None = None,
    strict_data: bool = True,
    validate_data: bool = True,
    data_quality_report: DataQualityReport | None = None,
    registry: PatternRegistry | None = None,
    evaluation_config: HistoricalEvaluationConfig | None = None,
    include_extended_hours: bool = True,
    session_mode: str | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> HistoricalEvaluationResult:
    """Evaluate historically collected signals against future market outcomes."""

    config = evaluation_config or HistoricalEvaluationConfig()
    config.validate()
    active_registry = registry or DEFAULT_PATTERN_REGISTRY
    completed_df, quality_report, normalized_as_of = _validate_and_prepare_history(
        df,
        symbol=symbol,
        interval=interval,
        exchange_timezone=exchange_timezone,
        as_of=as_of,
        strict_data=strict_data,
        validate_data=validate_data,
        data_quality_report=data_quality_report,
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    active_exchange_timezone = exchange_timezone or _get_exchange_timezone(completed_df)
    signals = _collect_signal_records(
        completed_df,
        symbol=symbol,
        interval=interval,
        display_timezone=display_timezone,
        lookback_bars=lookback_bars,
        exchange_timezone=active_exchange_timezone,
        strict_data=strict_data,
        registry=active_registry,
        evaluation_config=config,
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    enriched_signals = [
        replace(
            signal,
            outcomes=_compute_outcomes_for_signal(
                completed_df,
                signal=signal,
                interval=interval,
                config=config,
            ),
        )
        for signal in signals
    ]
    observations = _flatten_outcomes(enriched_signals)
    overall_groups = defaultdict(list)
    for observation in observations:
        overall_groups[int(observation["horizon_bars"])].append(observation)
    overall_by_horizon = {
        str(horizon): _summary_from_observations(
            overall_groups.get(horizon, []),
            horizon_bars=horizon,
        )
        for horizon in config.horizons_bars
    }

    return HistoricalEvaluationResult(
        symbol=symbol,
        interval=interval,
        evaluation_as_of=normalized_as_of,
        exchange_timezone=active_exchange_timezone,
        display_timezone=display_timezone,
        target_return=float(config.target_return),
        stop_return=float(config.stop_return),
        horizons_bars=tuple(int(horizon) for horizon in config.horizons_bars),
        signal_count=len(enriched_signals),
        signals=enriched_signals,
        overall_by_horizon=overall_by_horizon,
        by_pattern=_build_grouped_summaries(
            observations,
            group_field="pattern_name",
            horizons=config.horizons_bars,
        ),
        by_market_context=_build_grouped_summaries(
            observations,
            group_field="market_context",
            horizons=config.horizons_bars,
        ),
        by_session_time=_build_grouped_summaries(
            observations,
            group_field="session_time",
            horizons=config.horizons_bars,
        ),
        by_confidence_bucket=_build_grouped_summaries(
            observations,
            group_field="confidence_bucket",
            horizons=config.horizons_bars,
        ),
        notes=[
            "Historical performance is measured separately from heuristic rule confidence.",
            "Signal Confidence buckets in this report are grouping labels only and are not calibrated probabilities.",
            "Precision is defined here as directionally correct signals divided by evaluated signals.",
            "False-positive rate is defined here as directionally incorrect signals divided by evaluated signals.",
            *list(quality_report.warnings),
        ],
    )


def evaluate_historical_stock(
    symbol: str,
    *,
    period: str = "1mo",
    interval: str = "15m",
    as_of: pd.Timestamp | None = None,
    display_timezone: str = "Asia/Jerusalem",
    lookback_bars: int = 12,
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
    evaluation_config: HistoricalEvaluationConfig | None = None,
    include_extended_hours: bool = True,
    session_mode: str | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> HistoricalEvaluationResult:
    """Load market data through a provider and evaluate historical signals."""

    del instrument
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
                    include_extended_hours=include_extended_hours,
                    session_mode=session_mode,
                    regular_session_start=regular_session_start,
                    regular_session_end=regular_session_end,
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
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
    )
    return evaluate_historical_dataframe(
        payload.dataframe,
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        display_timezone=display_timezone,
        lookback_bars=lookback_bars,
        exchange_timezone=payload.exchange_timezone,
        strict_data=strict_data,
        validate_data=False,
        data_quality_report=payload.quality_report,
        registry=registry,
        evaluation_config=evaluation_config,
        include_extended_hours=include_extended_hours,
        session_mode=session_mode,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
