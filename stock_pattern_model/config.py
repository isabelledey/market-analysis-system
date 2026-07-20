"""Configuration objects and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from stock_pattern_model.exceptions import ConfigurationError


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

    def cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir)
