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
    event_id: str | None = None
    setup_id: str | None = None
    evidence_group: str | None = None
    strength_label: str = "regular"
    volume_baseline_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _serialize_value(value)
            for key, value in asdict(self).items()
        }
