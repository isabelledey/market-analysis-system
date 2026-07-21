"""Configuration objects and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from stock_pattern_model.exceptions import ConfigurationError
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_END
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_START
from stock_pattern_model.session_utils import DEFAULT_SESSION_MODE
from stock_pattern_model.session_utils import normalize_session_mode


SUPPORTED_INTERVALS = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
)


@dataclass(frozen=True)
class PatternConfig:
    """Tunable values for rule-based pattern detection."""

    breakout_lookback: int = 20
    breakout_cooldown_bars: int = 3
    atr_tolerance_multiplier: float = 0.15
    percentage_tolerance: float = 0.001
    reclaim_confirmation_bars: int = 2
    minimum_volume_strength: float = 1.0
    doji_body_ratio_max: float = 0.08
    star_body_ratio_max: float = 0.35
    gap_tolerance_ratio: float = 0.10
    pivot_left_bars: int = 2
    pivot_right_bars: int = 2
    double_pattern_price_tolerance_ratio: float = 0.006
    double_pattern_min_separation_bars: int = 4
    double_pattern_max_separation_bars: int = 24
    double_pattern_min_valley_depth_ratio: float = 0.008
    score_tentative_patterns: bool = False

    def validate(self) -> None:
        if self.breakout_lookback < 2:
            raise ConfigurationError("breakout_lookback must be at least 2.")
        if self.breakout_cooldown_bars < 0:
            raise ConfigurationError("breakout_cooldown_bars must be >= 0.")
        if self.atr_tolerance_multiplier < 0:
            raise ConfigurationError("atr_tolerance_multiplier must be >= 0.")
        if self.percentage_tolerance < 0:
            raise ConfigurationError("percentage_tolerance must be >= 0.")
        if self.reclaim_confirmation_bars < 1:
            raise ConfigurationError("reclaim_confirmation_bars must be at least 1.")
        if self.minimum_volume_strength <= 0:
            raise ConfigurationError("minimum_volume_strength must be positive.")
        if not 0 < self.doji_body_ratio_max < 1:
            raise ConfigurationError("doji_body_ratio_max must be between 0 and 1.")
        if not 0 < self.star_body_ratio_max < 1:
            raise ConfigurationError("star_body_ratio_max must be between 0 and 1.")
        if self.gap_tolerance_ratio < 0:
            raise ConfigurationError("gap_tolerance_ratio must be >= 0.")
        if self.pivot_left_bars < 1 or self.pivot_right_bars < 1:
            raise ConfigurationError("pivot_left_bars and pivot_right_bars must be at least 1.")
        if self.double_pattern_min_separation_bars < 1:
            raise ConfigurationError("double_pattern_min_separation_bars must be at least 1.")
        if self.double_pattern_max_separation_bars < self.double_pattern_min_separation_bars:
            raise ConfigurationError(
                "double_pattern_max_separation_bars must be >= double_pattern_min_separation_bars."
            )
        if self.double_pattern_price_tolerance_ratio < 0:
            raise ConfigurationError("double_pattern_price_tolerance_ratio must be >= 0.")
        if self.double_pattern_min_valley_depth_ratio < 0:
            raise ConfigurationError("double_pattern_min_valley_depth_ratio must be >= 0.")


@dataclass(frozen=True)
class ScoringConfig:
    """Tunable values for analysis scoring and ranking."""

    lookback_bars: int = 12
    top_pattern_count: int = 3
    pattern_max_age_bars: int = 12
    reversal_pattern_max_age_bars: int = 6
    breakout_pattern_max_age_bars: int = 12
    consolidation_pattern_max_age_bars: int = 8
    structural_pattern_max_age_bars: int = 16
    tentative_pattern_max_age_bars: int = 6
    recency_decay: float = 0.85
    state_expiration_bars: int = 4
    breakout_state_max_age_bars: int = 3
    trend_score_weight: float = 12.0
    strong_signal_multiplier: float = 1.15
    tentative_signal_multiplier: float = 0.50
    volume_confirmation_bonus: float = 2.5
    bias_threshold: float = 12.0
    minimum_bias_confidence: float = 28.0
    conflict_neutrality_ratio: float = 0.65
    duplicate_group_confidence_penalty: float = 4.0
    data_warning_confidence_penalty: float = 5.0

    def validate(self) -> None:
        if self.lookback_bars < 1:
            raise ConfigurationError("lookback_bars must be at least 1.")
        if self.top_pattern_count < 1:
            raise ConfigurationError("top_pattern_count must be at least 1.")
        if self.pattern_max_age_bars < 1:
            raise ConfigurationError("pattern_max_age_bars must be at least 1.")
        if self.reversal_pattern_max_age_bars < 1:
            raise ConfigurationError("reversal_pattern_max_age_bars must be at least 1.")
        if self.breakout_pattern_max_age_bars < 1:
            raise ConfigurationError("breakout_pattern_max_age_bars must be at least 1.")
        if self.consolidation_pattern_max_age_bars < 1:
            raise ConfigurationError("consolidation_pattern_max_age_bars must be at least 1.")
        if self.structural_pattern_max_age_bars < 1:
            raise ConfigurationError("structural_pattern_max_age_bars must be at least 1.")
        if self.tentative_pattern_max_age_bars < 1:
            raise ConfigurationError("tentative_pattern_max_age_bars must be at least 1.")
        if not 0 < self.recency_decay <= 1:
            raise ConfigurationError("recency_decay must be between 0 and 1.")
        if self.state_expiration_bars < 0:
            raise ConfigurationError("state_expiration_bars must be >= 0.")
        if self.breakout_state_max_age_bars < 0:
            raise ConfigurationError("breakout_state_max_age_bars must be >= 0.")
        if self.trend_score_weight < 0:
            raise ConfigurationError("trend_score_weight must be >= 0.")
        if self.strong_signal_multiplier < 1:
            raise ConfigurationError("strong_signal_multiplier must be >= 1.")
        if not 0 <= self.tentative_signal_multiplier <= 1:
            raise ConfigurationError("tentative_signal_multiplier must be between 0 and 1.")
        if self.volume_confirmation_bonus < 0:
            raise ConfigurationError("volume_confirmation_bonus must be >= 0.")
        if self.bias_threshold < 0:
            raise ConfigurationError("bias_threshold must be >= 0.")
        if self.minimum_bias_confidence < 0:
            raise ConfigurationError("minimum_bias_confidence must be >= 0.")
        if not 0 <= self.conflict_neutrality_ratio <= 1:
            raise ConfigurationError("conflict_neutrality_ratio must be between 0 and 1.")
        if self.duplicate_group_confidence_penalty < 0:
            raise ConfigurationError("duplicate_group_confidence_penalty must be >= 0.")
        if self.data_warning_confidence_penalty < 0:
            raise ConfigurationError("data_warning_confidence_penalty must be >= 0.")


@dataclass(frozen=True)
class TimezoneConfig:
    """Display-time configuration for result formatting."""

    display_timezone: str = "Asia/Jerusalem"

    def validate(self) -> None:
        try:
            ZoneInfo(self.display_timezone)
        except ZoneInfoNotFoundError as error:
            raise ConfigurationError(
                f"Unknown display timezone: {self.display_timezone}"
            ) from error

    def to_zoneinfo(self) -> ZoneInfo:
        self.validate()
        return ZoneInfo(self.display_timezone)


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level configuration for one analysis run."""

    interval: str = "15m"
    period: str = "1mo"
    pattern: PatternConfig = PatternConfig()
    scoring: ScoringConfig = ScoringConfig()
    timezone: TimezoneConfig = TimezoneConfig()

    def validate(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ConfigurationError(
                f"Unsupported interval '{self.interval}'. Supported intervals: "
                f"{', '.join(SUPPORTED_INTERVALS)}"
            )
        if not self.period:
            raise ConfigurationError("period must be a non-empty string.")
        self.pattern.validate()
        self.scoring.validate()
        self.timezone.validate()


@dataclass(frozen=True)
class MarketDataConfig:
    """Provider-level configuration for loading and validating market data."""

    timeout_seconds: float = 10.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    cache_dir: str | None = None
    cache_ttl_seconds: int = 3600
    use_cache: bool = True
    strict_data: bool = True
    exchange_timezone: str | None = None
    include_extended_hours: bool = True
    session_mode: str = DEFAULT_SESSION_MODE
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive.")
        if self.retry_attempts < 1:
            raise ConfigurationError("retry_attempts must be at least 1.")
        if self.retry_backoff_seconds < 0:
            raise ConfigurationError("retry_backoff_seconds must be >= 0.")
        if self.cache_ttl_seconds < 0:
            raise ConfigurationError("cache_ttl_seconds must be >= 0.")
        if self.cache_dir is not None and not str(self.cache_dir).strip():
            raise ConfigurationError("cache_dir must not be blank when provided.")
        if self.exchange_timezone is not None:
            try:
                ZoneInfo(self.exchange_timezone)
            except ZoneInfoNotFoundError as error:
                raise ConfigurationError(
                    f"Unknown exchange timezone: {self.exchange_timezone}"
                ) from error
        try:
            normalize_session_mode(self.session_mode)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        try:
            start_clock = time.fromisoformat(self.regular_session_start)
            end_clock = time.fromisoformat(self.regular_session_end)
        except ValueError as error:
            raise ConfigurationError("regular_session_start and regular_session_end must use HH:MM format.") from error
        if start_clock >= end_clock:
            raise ConfigurationError("regular_session_start must be earlier than regular_session_end.")

    def cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir)


@dataclass(frozen=True)
class HistoricalEvaluationConfig:
    """Configuration for leakage-free historical signal evaluation."""

    horizons_bars: tuple[int, ...] = (1, 3, 6)
    target_return: float = 0.01
    stop_return: float = 0.005
    minimum_history_bars: int = 20
    only_score_eligible_signals: bool = True
    same_bar_touch_policy: str = "ambiguous"
    confidence_bucket_edges: tuple[float, ...] = (25.0, 50.0, 75.0)

    def validate(self) -> None:
        if not self.horizons_bars:
            raise ConfigurationError("horizons_bars must contain at least one positive horizon.")
        if any(int(horizon) < 1 for horizon in self.horizons_bars):
            raise ConfigurationError("Every evaluation horizon must be at least 1 bar.")
        if tuple(sorted(int(horizon) for horizon in self.horizons_bars)) != tuple(int(horizon) for horizon in self.horizons_bars):
            raise ConfigurationError("horizons_bars must be sorted in ascending order.")
        if self.target_return <= 0:
            raise ConfigurationError("target_return must be positive.")
        if self.stop_return <= 0:
            raise ConfigurationError("stop_return must be positive.")
        if self.minimum_history_bars < 1:
            raise ConfigurationError("minimum_history_bars must be at least 1.")
        if self.same_bar_touch_policy not in {"ambiguous", "target_first", "stop_first"}:
            raise ConfigurationError(
                "same_bar_touch_policy must be one of: ambiguous, target_first, stop_first."
            )
        if any(edge <= 0 or edge >= 100 for edge in self.confidence_bucket_edges):
            raise ConfigurationError("confidence_bucket_edges must fall strictly between 0 and 100.")
        if tuple(sorted(float(edge) for edge in self.confidence_bucket_edges)) != tuple(
            float(edge) for edge in self.confidence_bucket_edges
        ):
            raise ConfigurationError("confidence_bucket_edges must be sorted in ascending order.")
