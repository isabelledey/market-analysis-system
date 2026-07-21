"""Typed domain objects used by the CLI and analysis package."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class PatternFamily(str, Enum):
    """Supported pattern families for registry-based detection."""

    BREAKOUT = "breakout"
    ENGULFING = "engulfing"
    PIN_BAR = "pin_bar"
    INSIDE_BAR = "inside_bar"
    INSIDE_BAR_FAILURE = "inside_bar_failure"
    DOJI = "doji"
    STAR = "star"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    EXTENSION = "extension"


class PatternStatus(str, Enum):
    """Lifecycle states for pattern events."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ResolvedInstrument:
    """Normalized instrument identity used by the analyzer and CLI."""

    input_identifier: str
    symbol: str
    security_number: str | None = None
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    exchange_timezone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    """Structured summary of validation results for loaded market data."""

    row_count: int
    completed_row_count: int
    duplicate_count: int
    missing_value_count: int
    invalid_ohlc_count: int
    irregular_gap_count: int
    warnings: list[str]
    cleaning_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketDataPayload:
    """Container for validated market data plus metadata."""

    dataframe: Any
    quality_report: DataQualityReport
    exchange_timezone: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_report": self.quality_report.to_dict(),
            "exchange_timezone": self.exchange_timezone,
            "metadata": self.metadata,
        }


def _serialize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat(timespec="minutes")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PatternEvent:
    """Structured event emitted by a pattern detector."""

    pattern_id: str
    pattern_name: str
    pattern_family: PatternFamily
    bias: str
    status: PatternStatus
    pattern_start_at: pd.Timestamp
    pattern_end_at: pd.Timestamp
    bar_start_at: pd.Timestamp
    bar_end_at: pd.Timestamp
    detected_at: pd.Timestamp
    relevant_prices: dict[str, float]
    relevant_indices: list[int]
    detection_reason: str
    signal_strength: float
    base_score: float
    exchange_timezone: str
    setup_completion_at: pd.Timestamp | None = None
    confirmation_at: pd.Timestamp | None = None
    pattern_start_index: int | None = None
    pattern_completion_index: int | None = None
    detected_index: int | None = None
    event_id: str | None = None
    setup_id: str | None = None
    evidence_group: str | None = None
    parent_pattern_id: str | None = None
    confirms_pattern_id: str | None = None
    related_event_ids: list[str] | None = None
    relationship_type: str | None = None
    strength_label: str = "regular"
    volume_baseline_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _serialize_value(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class PatternScoreEligibility:
    """Single source of truth for whether a pattern can affect the current signal."""

    eligible: bool
    reason: str | None
    anchor_type: str
    anchor_index: int | None
    age_bars: int
    max_age_bars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalSignalOutcome:
    """Forward-looking outcome metrics for one signal at one forecast horizon."""

    horizon_bars: int
    future_bar_count: int
    available: bool
    exit_index: int | None
    exit_at: pd.Timestamp | None
    exit_close: float | None
    raw_forward_return: float | None
    directional_forward_return: float | None
    mfe_return: float | None
    mae_return: float | None
    direction_correct: bool | None
    target_price: float | None
    stop_price: float | None
    target_hit: bool
    stop_hit: bool
    first_touch: str
    first_touch_index: int | None
    simulated_trade_return: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _serialize_value(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class HistoricalSignalRecord:
    """One historically collected signal plus its evaluation outcomes."""

    signal_id: str
    event_id: str
    setup_id: str
    evidence_group: str
    symbol: str
    interval: str
    pattern_id: str
    pattern_name: str
    pattern_family: str
    bias: str
    status: str
    detected_at: pd.Timestamp
    bar_start_at: pd.Timestamp
    detected_index: int
    entry_price: float
    trend: str
    market_state: str
    overall_bias: str
    rule_confidence: float
    signal_confidence_bucket: str
    exchange_timezone: str
    display_timezone: str
    session_segment: str
    session_time_exchange: str
    signal_strength: float
    strength_label: str
    volume_baseline_source: str
    outcomes: list[HistoricalSignalOutcome]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: _serialize_value(value)
            for key, value in asdict(self).items()
            if key != "outcomes"
        }
        payload["outcomes"] = [outcome.to_dict() for outcome in self.outcomes]
        return payload


@dataclass(frozen=True)
class HistoricalPerformanceSummary:
    """Aggregate performance metrics for a horizon and grouping bucket."""

    horizon_bars: int
    evaluated_signals: int
    wins: int
    losses: int
    flat: int
    direction_correct_rate: float | None
    precision: float | None
    false_positive_rate: float | None
    win_rate: float | None
    average_forward_return: float | None
    median_forward_return: float | None
    average_raw_forward_return: float | None
    median_raw_forward_return: float | None
    average_mfe_return: float | None
    median_mfe_return: float | None
    average_mae_return: float | None
    median_mae_return: float | None
    expectancy: float | None
    target_first_rate: float | None
    stop_first_rate: float | None
    neither_hit_rate: float | None
    ambiguous_same_bar_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _serialize_value(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class HistoricalEvaluationResult:
    """Structured result for historical evaluation and backtesting."""

    symbol: str
    interval: str
    evaluation_as_of: pd.Timestamp
    exchange_timezone: str
    display_timezone: str
    target_return: float
    stop_return: float
    horizons_bars: tuple[int, ...]
    signal_count: int
    signals: list[HistoricalSignalRecord]
    overall_by_horizon: dict[str, HistoricalPerformanceSummary]
    by_pattern: dict[str, dict[str, HistoricalPerformanceSummary]]
    by_market_context: dict[str, dict[str, HistoricalPerformanceSummary]]
    by_session_time: dict[str, dict[str, HistoricalPerformanceSummary]]
    by_confidence_bucket: dict[str, dict[str, HistoricalPerformanceSummary]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "evaluation_as_of": _serialize_value(self.evaluation_as_of),
            "exchange_timezone": self.exchange_timezone,
            "display_timezone": self.display_timezone,
            "target_return": self.target_return,
            "stop_return": self.stop_return,
            "horizons_bars": list(self.horizons_bars),
            "signal_count": self.signal_count,
            "signals": [signal.to_dict() for signal in self.signals],
            "overall_by_horizon": {
                key: summary.to_dict() for key, summary in self.overall_by_horizon.items()
            },
            "by_pattern": {
                key: {horizon: summary.to_dict() for horizon, summary in value.items()}
                for key, value in self.by_pattern.items()
            },
            "by_market_context": {
                key: {horizon: summary.to_dict() for horizon, summary in value.items()}
                for key, value in self.by_market_context.items()
            },
            "by_session_time": {
                key: {horizon: summary.to_dict() for horizon, summary in value.items()}
                for key, value in self.by_session_time.items()
            },
            "by_confidence_bucket": {
                key: {horizon: summary.to_dict() for horizon, summary in value.items()}
                for key, value in self.by_confidence_bucket.items()
            },
            "notes": list(self.notes),
        }
