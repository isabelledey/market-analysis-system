"""Registry-based rule detectors for candlestick and chart patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from stock_pattern_model.config import PatternConfig
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.session_utils import pattern_session_key_series


def _get_bar_timedelta(interval: str) -> pd.Timedelta:
    return pd.to_timedelta(interval)


def _get_bar_end(timestamp: pd.Timestamp, interval: str) -> pd.Timestamp:
    return timestamp + _get_bar_timedelta(interval)


def _get_exchange_timezone(data: pd.DataFrame) -> str:
    timezone = pd.to_datetime(data["Datetime"]).dt.tz
    if timezone is None:
        raise ValueError("Pattern detection requires timezone-aware Datetime values.")
    return str(timezone)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    if pd.isna(value):
        return fallback
    return float(value)


def _signal_strength(row: pd.Series) -> float:
    return round(
        max(
            _safe_float(row.get("Range_Strength")),
            _safe_float(row.get("Volume_Strength")),
        ),
        2,
    )


def _gap_tolerance(*ranges: float, ratio: float) -> float:
    valid_ranges = [item for item in ranges if pd.notna(item)]
    if not valid_ranges:
        return 0.0
    return float(np.mean(valid_ranges)) * ratio


def _price_tolerance_from_row(
    row: pd.Series,
    reference_level: float,
    config: PatternConfig,
) -> float:
    avg_range = row.get("Avg_Range_20_Bars")
    candle_range = _safe_float(row.get("Candle_Range"))
    baseline = _safe_float(avg_range, candle_range)
    if baseline <= 0:
        baseline = max(
            _safe_float(row.get("High")) - _safe_float(row.get("Low")),
            abs(reference_level) * config.percentage_tolerance,
            0.01,
        )
    return max(
        baseline * config.atr_tolerance_multiplier,
        abs(reference_level) * config.percentage_tolerance,
        0.01,
    )


def _rolling_break_levels(
    data: pd.DataFrame,
    *,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    rolling_high = data["High"].rolling(
        window=lookback,
        min_periods=lookback,
    ).max().shift(1)
    rolling_low = data["Low"].rolling(
        window=lookback,
        min_periods=lookback,
    ).min().shift(1)
    return rolling_high, rolling_low


def _volume_confirmed(row: pd.Series, config: PatternConfig) -> bool:
    volume_strength = row.get("Volume_Strength")
    return pd.notna(volume_strength) and float(volume_strength) >= config.minimum_volume_strength


def _break_distance_strength(
    close_price: float,
    reference_level: float,
    tolerance: float,
) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, abs(close_price - reference_level) / tolerance)


def _is_strong_break_event(
    row: pd.Series,
    *,
    close_price: float,
    reference_level: float,
    tolerance: float,
) -> bool:
    distance_strength = _break_distance_strength(close_price, reference_level, tolerance)
    return bool(
        row.get("Strong_Volume", False)
        or row.get("Strong_Range", False)
        or distance_strength >= 2.5
    )


def _same_pattern_session(data: pd.DataFrame, *indices: int) -> bool:
    if "Pattern_Session_Key" in data.columns:
        keys = {str(data.iloc[index]["Pattern_Session_Key"]) for index in indices}
        return len(keys) == 1
    fallback_keys = pattern_session_key_series(data.iloc[list(indices)]["Datetime"])
    return fallback_keys.nunique() == 1


class PatternDetector(Protocol):
    """Interface for independently testable pattern detectors."""

    pattern_id: str
    pattern_name: str
    family: PatternFamily
    minimum_required_history: int
    default_bias: str
    default_base_score: float
    default_priority: int

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        ...


@dataclass(frozen=True)
class BasePatternDetector:
    """Shared metadata and event-building helpers."""

    pattern_id: str
    pattern_name: str
    family: PatternFamily
    minimum_required_history: int
    default_bias: str
    default_base_score: float
    default_priority: int

    def _build_event(
        self,
        data: pd.DataFrame,
        interval: str,
        final_index: int,
        *,
        pattern_start_index: int,
        relevant_indices: list[int],
        relevant_prices: dict[str, float],
        detection_reason: str,
        status: PatternStatus = PatternStatus.CONFIRMED,
        bias: str | None = None,
        detected_at_index: int | None = None,
        pattern_end_index: int | None = None,
        setup_completion_index: int | None = None,
        confirmation_index: int | None = None,
        signal_strength: float | None = None,
        base_score: float | None = None,
        strength_label: str = "regular",
        volume_baseline_source: str | None = None,
        pattern_name: str | None = None,
    ) -> PatternEvent:
        final_bar_index = final_index
        detected_bar_index = final_index if detected_at_index is None else detected_at_index
        pattern_end_bar_index = final_index if pattern_end_index is None else pattern_end_index
        bar_start_at = pd.Timestamp(data.iloc[final_bar_index]["Datetime"])
        bar_end_at = _get_bar_end(bar_start_at, interval)
        detected_bar_start = pd.Timestamp(data.iloc[detected_bar_index]["Datetime"])
        detected_at = _get_bar_end(detected_bar_start, interval)
        pattern_start_at = pd.Timestamp(data.iloc[pattern_start_index]["Datetime"])
        pattern_end_start = pd.Timestamp(data.iloc[pattern_end_bar_index]["Datetime"])
        pattern_end_at = _get_bar_end(pattern_end_start, interval)
        setup_completion_bar_index = (
            pattern_end_bar_index if setup_completion_index is None else setup_completion_index
        )
        setup_completion_at = _get_bar_end(
            pd.Timestamp(data.iloc[setup_completion_bar_index]["Datetime"]),
            interval,
        )
        confirmation_at = None
        if confirmation_index is not None:
            confirmation_at = _get_bar_end(
                pd.Timestamp(data.iloc[confirmation_index]["Datetime"]),
                interval,
            )
        exchange_timezone = _get_exchange_timezone(data)
        row = data.iloc[final_bar_index]
        return PatternEvent(
            pattern_id=self.pattern_id,
            pattern_name=pattern_name or self.pattern_name,
            pattern_family=self.family,
            bias=bias or self.default_bias,
            status=status,
            pattern_start_at=pattern_start_at,
            pattern_end_at=pattern_end_at,
            bar_start_at=bar_start_at,
            bar_end_at=bar_end_at,
            detected_at=detected_at,
            setup_completion_at=setup_completion_at,
            confirmation_at=confirmation_at,
            relevant_prices=relevant_prices,
            relevant_indices=relevant_indices,
            detection_reason=detection_reason,
            signal_strength=signal_strength if signal_strength is not None else _signal_strength(row),
            base_score=base_score if base_score is not None else self.default_base_score,
            exchange_timezone=exchange_timezone,
            pattern_start_index=pattern_start_index,
            pattern_completion_index=pattern_end_bar_index,
            detected_index=detected_bar_index,
            strength_label=strength_label,
            volume_baseline_source=(
                volume_baseline_source
                if volume_baseline_source is not None
                else str(row.get("Volume_Baseline_Source", "unknown"))
            ),
        )


class BullishEngulfingDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="bullish_engulfing",
            pattern_name="Bullish Engulfing",
            family=PatternFamily.ENGULFING,
            minimum_required_history=2,
            default_bias="Bullish",
            default_base_score=15,
            default_priority=3,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(1, len(data)):
            previous_row = data.iloc[index - 1]
            row = data.iloc[index]
            if not _same_pattern_session(data, index - 1, index):
                continue
            if not (
                bool(previous_row["Is_Bearish"])
                and bool(row["Is_Bullish"])
                and row["Open"] <= previous_row["Close"]
                and row["Close"] >= previous_row["Open"]
                and (
                    bool(row["Is_Significant_Candle"])
                    or bool(previous_row["Is_Significant_Candle"])
                )
            ):
                continue

            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 1,
                    relevant_indices=[index - 1, index],
                    relevant_prices={
                        "first_open": float(previous_row["Open"]),
                        "first_close": float(previous_row["Close"]),
                        "second_open": float(row["Open"]),
                        "second_close": float(row["Close"]),
                    },
                    detection_reason="The second candle fully engulfed the prior bearish candle body on the close.",
                )
            )
        return events


class BearishEngulfingDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="bearish_engulfing",
            pattern_name="Bearish Engulfing",
            family=PatternFamily.ENGULFING,
            minimum_required_history=2,
            default_bias="Bearish",
            default_base_score=15,
            default_priority=3,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(1, len(data)):
            previous_row = data.iloc[index - 1]
            row = data.iloc[index]
            if not _same_pattern_session(data, index - 1, index):
                continue
            if not (
                bool(previous_row["Is_Bullish"])
                and bool(row["Is_Bearish"])
                and row["Open"] >= previous_row["Close"]
                and row["Close"] <= previous_row["Open"]
                and (
                    bool(row["Is_Significant_Candle"])
                    or bool(previous_row["Is_Significant_Candle"])
                )
            ):
                continue

            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 1,
                    relevant_indices=[index - 1, index],
                    relevant_prices={
                        "first_open": float(previous_row["Open"]),
                        "first_close": float(previous_row["Close"]),
                        "second_open": float(row["Open"]),
                        "second_close": float(row["Close"]),
                    },
                    detection_reason="The second candle fully engulfed the prior bullish candle body on the close.",
                )
            )
        return events


class BullishPinBarDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="bullish_pin_bar",
            pattern_name="Bullish Pin Bar",
            family=PatternFamily.PIN_BAR,
            minimum_required_history=1,
            default_bias="Bullish",
            default_base_score=10,
            default_priority=5,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index, row in data.iterrows():
            candle_range = row["Candle_Range"]
            if candle_range == 0 or pd.isna(candle_range):
                continue
            close_location = (row["Close"] - row["Low"]) / candle_range
            if not (
                row["Lower_Wick_Ratio"] >= 0.55
                and row["Body_Ratio"] <= 0.35
                and close_location >= 0.60
                and bool(row["Is_Significant_Candle"])
            ):
                continue

            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
            events.append(
                self._build_event(
                    data,
                    interval,
                    int(index),
                    pattern_start_index=int(index),
                    relevant_indices=[int(index)],
                    relevant_prices={
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        "A long lower wick and strong close marked a bullish pin bar on candle close "
                        f"with the {volume_source} volume baseline."
                    ),
                )
            )
        return events


class HammerDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="hammer",
            pattern_name="Hammer",
            family=PatternFamily.PIN_BAR,
            minimum_required_history=1,
            default_bias="Bullish",
            default_base_score=10,
            default_priority=6,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index, row in data.iterrows():
            candle_range = row["Candle_Range"]
            if candle_range == 0 or pd.isna(candle_range):
                continue
            close_location = (row["Close"] - row["Low"]) / candle_range
            trend_context = str(row.get("Trend", "Neutral")) == "Downtrend"
            if not (
                trend_context
                and row["Lower_Wick_Ratio"] >= 0.55
                and row["Body_Ratio"] <= 0.35
                and close_location >= 0.60
                and bool(row["Is_Significant_Candle"])
            ):
                continue

            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
            events.append(
                self._build_event(
                    data,
                    interval,
                    int(index),
                    pattern_start_index=int(index),
                    relevant_indices=[int(index)],
                    relevant_prices={
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        "A hammer formed after a downtrend: the small body met the configured "
                        "tolerance, the lower wick showed bullish rejection, and the candle only "
                        f"became knowable on the close with the {volume_source} volume baseline."
                    ),
                )
            )
        return events


class ShootingStarDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="shooting_star",
            pattern_name="Shooting Star",
            family=PatternFamily.PIN_BAR,
            minimum_required_history=1,
            default_bias="Bearish",
            default_base_score=10,
            default_priority=5,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index, row in data.iterrows():
            candle_range = row["Candle_Range"]
            if candle_range == 0 or pd.isna(candle_range):
                continue
            close_location = (row["Close"] - row["Low"]) / candle_range
            if not (
                row["Upper_Wick_Ratio"] >= 0.55
                and row["Body_Ratio"] <= 0.35
                and close_location <= 0.40
                and bool(row["Is_Significant_Candle"])
            ):
                continue

            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
            events.append(
                self._build_event(
                    data,
                    interval,
                    int(index),
                    pattern_start_index=int(index),
                    relevant_indices=[int(index)],
                    relevant_prices={
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        "A long upper wick and weak close marked a shooting star on candle close "
                        f"with the {volume_source} volume baseline."
                    ),
                )
            )
        return events


class InsideBarDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="inside_bar",
            pattern_name="Inside Bar",
            family=PatternFamily.INSIDE_BAR,
            minimum_required_history=2,
            default_bias="Neutral",
            default_base_score=0,
            default_priority=6,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(1, len(data)):
            previous_row = data.iloc[index - 1]
            row = data.iloc[index]
            if not _same_pattern_session(data, index - 1, index):
                continue
            if not (
                row["High"] < previous_row["High"]
                and row["Low"] > previous_row["Low"]
                and (
                    bool(row["Is_Significant_Candle"])
                    or bool(previous_row["Is_Significant_Candle"])
                )
            ):
                continue

            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 1,
                    relevant_indices=[index - 1, index],
                    relevant_prices={
                        "mother_high": float(previous_row["High"]),
                        "mother_low": float(previous_row["Low"]),
                        "inside_high": float(row["High"]),
                        "inside_low": float(row["Low"]),
                    },
                    detection_reason="The inside bar closed within the full range of the mother bar.",
                )
            )
        return events


class InsideBarFailureDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="inside_bar_failure",
            pattern_name="Inside Bar Failure",
            family=PatternFamily.INSIDE_BAR_FAILURE,
            minimum_required_history=3,
            default_bias="Neutral",
            default_base_score=11,
            default_priority=4,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(2, len(data)):
            if not _same_pattern_session(data, index - 2, index - 1, index):
                continue
            mother_bar = data.iloc[index - 2]
            inside_bar = data.iloc[index - 1]
            failure_bar = data.iloc[index]
            valid_inside = (
                inside_bar["High"] < mother_bar["High"]
                and inside_bar["Low"] > mother_bar["Low"]
            )
            if not valid_inside or not bool(failure_bar["Is_Significant_Candle"]):
                continue

            bearish_failure = (
                failure_bar["High"] > mother_bar["High"]
                and failure_bar["Low"] >= mother_bar["Low"]
                and failure_bar["Close"] < mother_bar["High"]
                and failure_bar["Close"] > mother_bar["Low"]
                and bool(failure_bar["Is_Bearish"])
            )
            bullish_failure = (
                failure_bar["Low"] < mother_bar["Low"]
                and failure_bar["High"] <= mother_bar["High"]
                and failure_bar["Close"] > mother_bar["Low"]
                and failure_bar["Close"] < mother_bar["High"]
                and bool(failure_bar["Is_Bullish"])
            )
            if not bearish_failure and not bullish_failure:
                continue

            is_bearish = bearish_failure
            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 2,
                    relevant_indices=[index - 2, index - 1, index],
                    relevant_prices={
                        "mother_high": float(mother_bar["High"]),
                        "mother_low": float(mother_bar["Low"]),
                        "inside_high": float(inside_bar["High"]),
                        "inside_low": float(inside_bar["Low"]),
                        "failure_close": float(failure_bar["Close"]),
                    },
                    detection_reason=(
                        "The failure candle swept an inside-bar boundary and closed back inside the "
                        "mother-bar range."
                    ),
                    bias="Bearish" if is_bearish else "Bullish",
                    pattern_name=(
                        "Inside Bar Failure Bearish Reversal"
                        if is_bearish
                        else "Inside Bar Failure Bullish Reversal"
                    ),
                )
            )
        return events


class BreakoutDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="breakout",
            pattern_name="20-Bar Breakout",
            family=PatternFamily.BREAKOUT,
            minimum_required_history=21,
            default_bias="Bullish",
            default_base_score=18,
            default_priority=2,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        if len(data) < config.breakout_lookback + 1:
            return []

        rolling_high, _ = _rolling_break_levels(data, lookback=config.breakout_lookback)
        previous_reference_high = rolling_high.shift(1).fillna(rolling_high)
        events: list[PatternEvent] = []
        next_allowed_index = 0
        active_reference: float | None = None
        active_tolerance: float | None = None

        for index in range(1, len(data)):
            if index < next_allowed_index:
                continue
            row = data.iloc[index]
            previous_row = data.iloc[index - 1]
            current_reference = rolling_high.iloc[index]
            previous_reference = previous_reference_high.iloc[index]
            if pd.isna(current_reference) or pd.isna(previous_reference):
                continue
            if not _same_pattern_session(data, index - 1, index):
                active_reference = None
                active_tolerance = None
                continue
            if (
                active_reference is not None
                and active_tolerance is not None
                and float(previous_row["Close"]) >= active_reference + active_tolerance
            ):
                continue
            current_tolerance = _price_tolerance_from_row(row, float(current_reference), config)
            previous_tolerance = _price_tolerance_from_row(previous_row, float(previous_reference), config)
            crossed = (
                float(row["Close"]) > float(current_reference)
                and float(previous_row["Close"]) <= float(previous_reference)
                and _volume_confirmed(row, config)
            )
            if not crossed:
                continue

            is_strong = _is_strong_break_event(
                row,
                close_price=float(row["Close"]),
                reference_level=float(current_reference),
                tolerance=current_tolerance,
            )
            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index,
                    relevant_indices=[index],
                    relevant_prices={
                        "breakout_level": float(current_reference),
                        "tolerance": float(current_tolerance),
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        f"Close crossed above the prior {config.breakout_lookback}-bar high at "
                        f"{current_reference:.2f}, passed volume confirmation, and cleared the "
                        f"event-reset tolerance state. "
                        f"Baseline tolerance for follow-through and reset logic was "
                        f"{current_tolerance:.2f} using the {volume_source} volume baseline."
                    ),
                    base_score=26 if is_strong else 18,
                    strength_label="strong" if is_strong else "regular",
                    pattern_name=(
                        f"Strong {config.breakout_lookback}-Bar Breakout"
                        if is_strong
                        else f"{config.breakout_lookback}-Bar Breakout"
                    ),
                )
            )
            active_reference = float(current_reference)
            active_tolerance = float(current_tolerance)
            next_allowed_index = index + config.breakout_cooldown_bars + 1
        return events


class BreakdownDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="breakdown",
            pattern_name="20-Bar Breakdown",
            family=PatternFamily.BREAKOUT,
            minimum_required_history=21,
            default_bias="Bearish",
            default_base_score=18,
            default_priority=2,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        if len(data) < config.breakout_lookback + 1:
            return []

        _, rolling_low = _rolling_break_levels(data, lookback=config.breakout_lookback)
        previous_reference_low = rolling_low.shift(1).fillna(rolling_low)
        events: list[PatternEvent] = []
        next_allowed_index = 0
        active_reference: float | None = None
        active_tolerance: float | None = None

        for index in range(1, len(data)):
            if index < next_allowed_index:
                continue
            row = data.iloc[index]
            previous_row = data.iloc[index - 1]
            current_reference = rolling_low.iloc[index]
            previous_reference = previous_reference_low.iloc[index]
            if pd.isna(current_reference) or pd.isna(previous_reference):
                continue
            if not _same_pattern_session(data, index - 1, index):
                active_reference = None
                active_tolerance = None
                continue
            if (
                active_reference is not None
                and active_tolerance is not None
                and float(previous_row["Close"]) <= active_reference - active_tolerance
            ):
                continue
            current_tolerance = _price_tolerance_from_row(row, float(current_reference), config)
            previous_tolerance = _price_tolerance_from_row(previous_row, float(previous_reference), config)
            crossed = (
                float(row["Close"]) < float(current_reference)
                and float(previous_row["Close"]) >= float(previous_reference)
                and _volume_confirmed(row, config)
            )
            if not crossed:
                continue

            is_strong = _is_strong_break_event(
                row,
                close_price=float(row["Close"]),
                reference_level=float(current_reference),
                tolerance=current_tolerance,
            )
            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index,
                    relevant_indices=[index],
                    relevant_prices={
                        "breakdown_level": float(current_reference),
                        "tolerance": float(current_tolerance),
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        f"Close crossed below the prior {config.breakout_lookback}-bar low at "
                        f"{current_reference:.2f}, passed volume confirmation, and cleared the "
                        f"event-reset tolerance state. "
                        f"Baseline tolerance for follow-through and reset logic was "
                        f"{current_tolerance:.2f} using the {volume_source} volume baseline."
                    ),
                    base_score=26 if is_strong else 18,
                    strength_label="strong" if is_strong else "regular",
                    pattern_name=(
                        f"Strong {config.breakout_lookback}-Bar Breakdown"
                        if is_strong
                        else f"{config.breakout_lookback}-Bar Breakdown"
                    ),
                )
            )
            active_reference = float(current_reference)
            active_tolerance = float(current_tolerance)
            next_allowed_index = index + config.breakout_cooldown_bars + 1
        return events


class DojiDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="doji",
            pattern_name="Doji",
            family=PatternFamily.DOJI,
            minimum_required_history=1,
            default_bias="Neutral",
            default_base_score=4,
            default_priority=7,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index, row in data.iterrows():
            candle_range = row["Candle_Range"]
            if candle_range == 0 or pd.isna(candle_range):
                continue
            if row["Body_Ratio"] > config.doji_body_ratio_max:
                continue
            events.append(
                self._build_event(
                    data,
                    interval,
                    int(index),
                    pattern_start_index=int(index),
                    relevant_indices=[int(index)],
                    relevant_prices={
                        "open": float(row["Open"]),
                        "close": float(row["Close"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                    },
                    detection_reason=(
                        "The candle body stayed within the configured doji tolerance and became "
                        "knowable only after the bar closed."
                    ),
                    signal_strength=round(1.0 - min(float(row["Body_Ratio"]), 1.0), 2),
                )
            )
        return events


class MorningStarDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="morning_star",
            pattern_name="Morning Star",
            family=PatternFamily.STAR,
            minimum_required_history=3,
            default_bias="Bullish",
            default_base_score=16,
            default_priority=3,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(2, len(data)):
            if not _same_pattern_session(data, index - 2, index - 1, index):
                continue
            first = data.iloc[index - 2]
            second = data.iloc[index - 1]
            third = data.iloc[index]
            tolerance = _gap_tolerance(
                float(first["Candle_Range"]),
                float(second["Candle_Range"]),
                float(third["Candle_Range"]),
                ratio=config.gap_tolerance_ratio,
            )
            midpoint = (float(first["Open"]) + float(first["Close"])) / 2.0
            second_high_body = max(float(second["Open"]), float(second["Close"]))
            second_low_body = min(float(second["Open"]), float(second["Close"]))
            if not (
                bool(first["Is_Bearish"])
                and float(first["Body_Ratio"]) >= 0.45
                and float(second["Body_Ratio"]) <= config.star_body_ratio_max
                and second_high_body <= float(first["Close"]) + tolerance
                and bool(third["Is_Bullish"])
                and float(third["Close"]) >= midpoint
                and second_low_body <= float(first["Close"]) + tolerance
            ):
                continue

            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 2,
                    relevant_indices=[index - 2, index - 1, index],
                    relevant_prices={
                        "first_close": float(first["Close"]),
                        "star_open": float(second["Open"]),
                        "star_close": float(second["Close"]),
                        "recovery_close": float(third["Close"]),
                        "midpoint": midpoint,
                    },
                    detection_reason=(
                        "A bearish impulse, small-bodied star, and bullish recovery above the "
                        "first candle midpoint completed a morning star."
                    ),
                )
            )
        return events


class EveningStarDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="evening_star",
            pattern_name="Evening Star",
            family=PatternFamily.STAR,
            minimum_required_history=3,
            default_bias="Bearish",
            default_base_score=16,
            default_priority=3,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for index in range(2, len(data)):
            if not _same_pattern_session(data, index - 2, index - 1, index):
                continue
            first = data.iloc[index - 2]
            second = data.iloc[index - 1]
            third = data.iloc[index]
            tolerance = _gap_tolerance(
                float(first["Candle_Range"]),
                float(second["Candle_Range"]),
                float(third["Candle_Range"]),
                ratio=config.gap_tolerance_ratio,
            )
            midpoint = (float(first["Open"]) + float(first["Close"])) / 2.0
            second_low_body = min(float(second["Open"]), float(second["Close"]))
            second_high_body = max(float(second["Open"]), float(second["Close"]))
            if not (
                bool(first["Is_Bullish"])
                and float(first["Body_Ratio"]) >= 0.45
                and float(second["Body_Ratio"]) <= config.star_body_ratio_max
                and second_low_body >= float(first["Close"]) - tolerance
                and bool(third["Is_Bearish"])
                and float(third["Close"]) <= midpoint
                and second_high_body >= float(first["Close"]) - tolerance
            ):
                continue

            events.append(
                self._build_event(
                    data,
                    interval,
                    index,
                    pattern_start_index=index - 2,
                    relevant_indices=[index - 2, index - 1, index],
                    relevant_prices={
                        "first_close": float(first["Close"]),
                        "star_open": float(second["Open"]),
                        "star_close": float(second["Close"]),
                        "reversal_close": float(third["Close"]),
                        "midpoint": midpoint,
                    },
                    detection_reason=(
                        "A bullish impulse, small-bodied star, and bearish reversal below the "
                        "first candle midpoint completed an evening star."
                    ),
                )
            )
        return events


def _confirmed_pivot_highs(
    data: pd.DataFrame,
    left: int,
    right: int,
) -> list[dict[str, int | float]]:
    pivots: list[dict[str, int | float]] = []
    highs = data["High"].tolist()
    for center in range(left, len(data) - right):
        center_high = highs[center]
        left_highs = highs[center - left:center]
        right_highs = highs[center + 1:center + right + 1]
        if center_high > max(left_highs) and center_high >= max(right_highs):
            pivots.append(
                {
                    "index": center,
                    "price": float(center_high),
                    "detected_index": center + right,
                }
            )
    return pivots


def _confirmed_pivot_lows(
    data: pd.DataFrame,
    left: int,
    right: int,
) -> list[dict[str, int | float]]:
    pivots: list[dict[str, int | float]] = []
    lows = data["Low"].tolist()
    for center in range(left, len(data) - right):
        center_low = lows[center]
        left_lows = lows[center - left:center]
        right_lows = lows[center + 1:center + right + 1]
        if center_low < min(left_lows) and center_low <= min(right_lows):
            pivots.append(
                {
                    "index": center,
                    "price": float(center_low),
                    "detected_index": center + right,
                }
            )
    return pivots


class DoubleTopDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="double_top",
            pattern_name="Double Top",
            family=PatternFamily.DOUBLE_TOP,
            minimum_required_history=7,
            default_bias="Bearish",
            default_base_score=20,
            default_priority=2,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        pivots = _confirmed_pivot_highs(data, config.pivot_left_bars, config.pivot_right_bars)
        valley_pivots = _confirmed_pivot_lows(data, config.pivot_left_bars, config.pivot_right_bars)
        events: list[PatternEvent] = []

        for first_position in range(len(pivots) - 1):
            first = pivots[first_position]
            for second_position in range(first_position + 1, len(pivots)):
                second = pivots[second_position]
                separation = int(second["index"]) - int(first["index"])
                if separation < config.double_pattern_min_separation_bars:
                    continue
                if separation > config.double_pattern_max_separation_bars:
                    break

                first_peak = float(first["price"])
                second_peak = float(second["price"])
                peak_reference = max(first_peak, second_peak)
                if abs(first_peak - second_peak) / peak_reference > config.double_pattern_price_tolerance_ratio:
                    continue

                candidate_valleys = [
                    pivot
                    for pivot in valley_pivots
                    if int(first["index"]) < int(pivot["index"]) < int(second["index"])
                ]
                if not candidate_valleys:
                    continue
                valley = min(candidate_valleys, key=lambda item: float(item["price"]))
                valley_index = int(valley["index"])
                neckline = float(valley["price"])
                valley_depth = (min(first_peak, second_peak) - neckline) / min(first_peak, second_peak)
                if valley_depth < config.double_pattern_min_valley_depth_ratio:
                    continue

                tentative_detected_index = int(second["detected_index"])
                events.append(
                    self._build_event(
                        data,
                        interval,
                        int(second["index"]),
                        pattern_start_index=int(first["index"]),
                        relevant_indices=[int(first["index"]), valley_index, int(second["index"])],
                        relevant_prices={
                            "first_peak": first_peak,
                            "second_peak": second_peak,
                            "neckline": neckline,
                            "confirmation_price": neckline,
                        },
                        detection_reason=(
                            "Two confirmed swing highs matched within tolerance and formed a meaningful "
                            "valley, but neckline confirmation had not yet occurred."
                        ),
                        status=PatternStatus.TENTATIVE,
                        detected_at_index=tentative_detected_index,
                        setup_completion_index=int(second["index"]),
                        signal_strength=round(valley_depth * 100.0, 2),
                    )
                )

                expiry_index = tentative_detected_index + config.double_pattern_max_separation_bars
                second_row = data.iloc[int(second["index"])]
                neckline_tolerance = _price_tolerance_from_row(second_row, neckline, config)
                invalidation_tolerance = _price_tolerance_from_row(second_row, peak_reference, config)
                invalidation_level = peak_reference + invalidation_tolerance
                confirmed_or_failed = False
                for scan_index in range(tentative_detected_index + 1, min(len(data), expiry_index + 1)):
                    close_price = float(data.iloc[scan_index]["Close"])
                    if close_price < neckline - neckline_tolerance:
                        events.append(
                            self._build_event(
                                data,
                                interval,
                                scan_index,
                                pattern_start_index=int(first["index"]),
                                relevant_indices=[
                                    int(first["index"]),
                                    valley_index,
                                    int(second["index"]),
                                    scan_index,
                                ],
                                relevant_prices={
                                    "first_peak": first_peak,
                                    "second_peak": second_peak,
                                    "neckline": neckline,
                                    "confirmation_price": close_price,
                                },
                                detection_reason=(
                                    "After both swing highs were confirmed, price broke below the neckline "
                                    "and confirmed the double top."
                                ),
                                status=PatternStatus.CONFIRMED,
                                setup_completion_index=int(second["index"]),
                                confirmation_index=scan_index,
                                signal_strength=round(valley_depth * 100.0, 2),
                            )
                        )
                        confirmed_or_failed = True
                        break
                    if close_price > invalidation_level:
                        events.append(
                            self._build_event(
                                data,
                                interval,
                                scan_index,
                                pattern_start_index=int(first["index"]),
                                relevant_indices=[
                                    int(first["index"]),
                                    valley_index,
                                    int(second["index"]),
                                    scan_index,
                                ],
                                relevant_prices={
                                    "first_peak": first_peak,
                                    "second_peak": second_peak,
                                    "neckline": neckline,
                                    "confirmation_price": close_price,
                                },
                                detection_reason=(
                                    "The tentative double top failed because price closed back above the "
                                    "peak-tolerance ceiling before the neckline broke."
                                ),
                                status=PatternStatus.FAILED,
                                setup_completion_index=int(second["index"]),
                                signal_strength=round(valley_depth * 100.0, 2),
                            )
                        )
                        confirmed_or_failed = True
                        break

                if not confirmed_or_failed and expiry_index < len(data):
                    events.append(
                        self._build_event(
                            data,
                            interval,
                            expiry_index,
                            pattern_start_index=int(first["index"]),
                            relevant_indices=[int(first["index"]), valley_index, int(second["index"]), expiry_index],
                            relevant_prices={
                                "first_peak": first_peak,
                                "second_peak": second_peak,
                                "neckline": neckline,
                                "confirmation_price": neckline,
                            },
                            detection_reason=(
                                "The tentative double top expired because neckline confirmation did not "
                                "arrive within the configured follow-through window."
                            ),
                            status=PatternStatus.EXPIRED,
                            setup_completion_index=int(second["index"]),
                            signal_strength=round(valley_depth * 100.0, 2),
                        )
                    )
                break
        return events


class DoubleBottomDetector(BasePatternDetector):
    def __init__(self) -> None:
        super().__init__(
            pattern_id="double_bottom",
            pattern_name="Double Bottom",
            family=PatternFamily.DOUBLE_BOTTOM,
            minimum_required_history=7,
            default_bias="Bullish",
            default_base_score=20,
            default_priority=2,
        )

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        pivots = _confirmed_pivot_lows(data, config.pivot_left_bars, config.pivot_right_bars)
        peak_pivots = _confirmed_pivot_highs(data, config.pivot_left_bars, config.pivot_right_bars)
        events: list[PatternEvent] = []

        for first_position in range(len(pivots) - 1):
            first = pivots[first_position]
            for second_position in range(first_position + 1, len(pivots)):
                second = pivots[second_position]
                separation = int(second["index"]) - int(first["index"])
                if separation < config.double_pattern_min_separation_bars:
                    continue
                if separation > config.double_pattern_max_separation_bars:
                    break

                first_low = float(first["price"])
                second_low = float(second["price"])
                price_reference = max(first_low, second_low)
                if abs(first_low - second_low) / price_reference > config.double_pattern_price_tolerance_ratio:
                    continue

                candidate_peaks = [
                    pivot
                    for pivot in peak_pivots
                    if int(first["index"]) < int(pivot["index"]) < int(second["index"])
                ]
                if not candidate_peaks:
                    continue
                peak = max(candidate_peaks, key=lambda item: float(item["price"]))
                peak_index = int(peak["index"])
                neckline = float(peak["price"])
                peak_height = (neckline - max(first_low, second_low)) / neckline
                if peak_height < config.double_pattern_min_valley_depth_ratio:
                    continue

                tentative_detected_index = int(second["detected_index"])
                events.append(
                    self._build_event(
                        data,
                        interval,
                        int(second["index"]),
                        pattern_start_index=int(first["index"]),
                        relevant_indices=[int(first["index"]), peak_index, int(second["index"])],
                        relevant_prices={
                            "first_bottom": first_low,
                            "second_bottom": second_low,
                            "neckline": neckline,
                            "confirmation_price": neckline,
                        },
                        detection_reason=(
                            "Two confirmed swing lows matched within tolerance and formed a meaningful "
                            "intervening rally, but neckline confirmation had not yet occurred."
                        ),
                        status=PatternStatus.TENTATIVE,
                        detected_at_index=tentative_detected_index,
                        setup_completion_index=int(second["index"]),
                        signal_strength=round(peak_height * 100.0, 2),
                    )
                )

                expiry_index = tentative_detected_index + config.double_pattern_max_separation_bars
                second_row = data.iloc[int(second["index"])]
                neckline_tolerance = _price_tolerance_from_row(second_row, neckline, config)
                invalidation_tolerance = _price_tolerance_from_row(second_row, min(first_low, second_low), config)
                invalidation_level = min(first_low, second_low) - invalidation_tolerance
                confirmed_or_failed = False
                for scan_index in range(tentative_detected_index + 1, min(len(data), expiry_index + 1)):
                    close_price = float(data.iloc[scan_index]["Close"])
                    if close_price > neckline + neckline_tolerance:
                        events.append(
                            self._build_event(
                                data,
                                interval,
                                scan_index,
                                pattern_start_index=int(first["index"]),
                                relevant_indices=[
                                    int(first["index"]),
                                    peak_index,
                                    int(second["index"]),
                                    scan_index,
                                ],
                                relevant_prices={
                                    "first_bottom": first_low,
                                    "second_bottom": second_low,
                                    "neckline": neckline,
                                    "confirmation_price": close_price,
                                },
                                detection_reason=(
                                    "After both swing lows were confirmed, price broke above the neckline "
                                    "and confirmed the double bottom."
                                ),
                                status=PatternStatus.CONFIRMED,
                                setup_completion_index=int(second["index"]),
                                confirmation_index=scan_index,
                                signal_strength=round(peak_height * 100.0, 2),
                            )
                        )
                        confirmed_or_failed = True
                        break
                    if close_price < invalidation_level:
                        events.append(
                            self._build_event(
                                data,
                                interval,
                                scan_index,
                                pattern_start_index=int(first["index"]),
                                relevant_indices=[
                                    int(first["index"]),
                                    peak_index,
                                    int(second["index"]),
                                    scan_index,
                                ],
                                relevant_prices={
                                    "first_bottom": first_low,
                                    "second_bottom": second_low,
                                    "neckline": neckline,
                                    "confirmation_price": close_price,
                                },
                                detection_reason=(
                                    "The tentative double bottom failed because price closed below the "
                                    "low-tolerance floor before the neckline broke."
                                ),
                                status=PatternStatus.FAILED,
                                setup_completion_index=int(second["index"]),
                                signal_strength=round(peak_height * 100.0, 2),
                            )
                        )
                        confirmed_or_failed = True
                        break

                if not confirmed_or_failed and expiry_index < len(data):
                    events.append(
                        self._build_event(
                            data,
                            interval,
                            expiry_index,
                            pattern_start_index=int(first["index"]),
                            relevant_indices=[int(first["index"]), peak_index, int(second["index"]), expiry_index],
                            relevant_prices={
                                "first_bottom": first_low,
                                "second_bottom": second_low,
                                "neckline": neckline,
                                "confirmation_price": neckline,
                            },
                            detection_reason=(
                                "The tentative double bottom expired because neckline confirmation did not "
                                "arrive within the configured follow-through window."
                            ),
                            status=PatternStatus.EXPIRED,
                            setup_completion_index=int(second["index"]),
                            signal_strength=round(peak_height * 100.0, 2),
                        )
                    )
                break
        return events


@dataclass(frozen=True)
class PatternRegistry:
    """Extensible registry of pattern detectors."""

    detectors: tuple[PatternDetector, ...]

    def register(self, detector: PatternDetector) -> "PatternRegistry":
        return PatternRegistry(detectors=self.detectors + (detector,))

    def detect(self, data: pd.DataFrame, config: PatternConfig, interval: str) -> list[PatternEvent]:
        events: list[PatternEvent] = []
        for detector in self.detectors:
            if len(data) < detector.minimum_required_history:
                continue
            events.extend(detector.detect(data.copy(deep=False), config, interval))
        return events

    def details(self) -> dict[str, dict[str, object]]:
        return {
            detector.pattern_id: {
                "label": detector.pattern_name,
                "bias": detector.default_bias,
                "priority": detector.default_priority,
                "base_score": detector.default_base_score,
                "family": detector.family.value,
                "minimum_required_history": detector.minimum_required_history,
            }
            for detector in self.detectors
        }


def build_default_pattern_registry() -> PatternRegistry:
    return PatternRegistry(
        detectors=(
            BreakoutDetector(),
            BreakdownDetector(),
            BullishEngulfingDetector(),
            BearishEngulfingDetector(),
            MorningStarDetector(),
            EveningStarDetector(),
            DoubleTopDetector(),
            DoubleBottomDetector(),
            InsideBarFailureDetector(),
            BullishPinBarDetector(),
            HammerDetector(),
            ShootingStarDetector(),
            InsideBarDetector(),
            DojiDetector(),
        )
    )


DEFAULT_PATTERN_REGISTRY = build_default_pattern_registry()
PATTERN_DETAILS = DEFAULT_PATTERN_REGISTRY.details()


def detect_patterns(
    data: pd.DataFrame,
    config: PatternConfig,
    interval: str,
    registry: PatternRegistry | None = None,
) -> list[PatternEvent]:
    active_registry = registry or DEFAULT_PATTERN_REGISTRY
    return active_registry.detect(data, config, interval)


@dataclass(frozen=True)
class _TrendSnapshot:
    score: float
    label: str
    evidence: list[str]


@dataclass(frozen=True)
class _BreakStructureEvent:
    index: int
    direction: int
    base_score: float


def _trend_horizons(lookback_bars: int) -> tuple[int, int, int]:
    short_horizon = max(12, min(20, lookback_bars))
    medium_horizon = max(40, min(60, short_horizon * 4))
    long_horizon = max(100, min(200, short_horizon * 10))
    return short_horizon, medium_horizon, long_horizon


def _trend_label(score: float) -> str:
    if score >= 18.0:
        return "Uptrend"
    if score <= -18.0:
        return "Downtrend"
    return "Neutral"


def _relative_move(reference: float, comparison: float, tolerance: float) -> int:
    if comparison > reference * (1.0 + tolerance):
        return 1
    if comparison < reference * (1.0 - tolerance):
        return -1
    return 0


def _pattern_session_key_values(data: pd.DataFrame) -> np.ndarray:
    if "Pattern_Session_Key" in data.columns:
        return data["Pattern_Session_Key"].astype("string").fillna("").astype(str).to_numpy(dtype=object)
    return pattern_session_key_series(data["Datetime"]).astype(str).to_numpy(dtype=object)


def _confirmed_swings_from_arrays(
    highs_array: np.ndarray,
    lows_array: np.ndarray,
    *,
    pivot_left_bars: int,
    pivot_right_bars: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    window_length = len(highs_array)
    if window_length < pivot_left_bars + pivot_right_bars + 1:
        return highs, lows

    for index in range(pivot_left_bars, window_length - pivot_right_bars):
        high_slice = highs_array[index - pivot_left_bars : index + pivot_right_bars + 1]
        low_slice = lows_array[index - pivot_left_bars : index + pivot_right_bars + 1]
        high_value = float(highs_array[index])
        low_value = float(lows_array[index])

        if high_value == float(np.max(high_slice)) and int(np.count_nonzero(high_slice == high_value)) == 1:
            highs.append((index, high_value))
        if low_value == float(np.min(low_slice)) and int(np.count_nonzero(low_slice == low_value)) == 1:
            lows.append((index, low_value))

    return highs, lows


def _break_structure_events(
    data: pd.DataFrame,
    *,
    breakout_lookback: int,
) -> list[_BreakStructureEvent]:
    if len(data) < max(breakout_lookback + 1, 3):
        return []

    rolling_high, rolling_low = _rolling_break_levels(data, lookback=breakout_lookback)
    previous_reference_high = rolling_high.shift(1).fillna(rolling_high)
    previous_reference_low = rolling_low.shift(1).fillna(rolling_low)
    close_values = data["Close"].astype(float).to_numpy(copy=False)
    strong_volume = data.get("Strong_Volume", pd.Series(False, index=data.index)).fillna(False).astype(bool).to_numpy()
    strong_range = data.get("Strong_Range", pd.Series(False, index=data.index)).fillna(False).astype(bool).to_numpy()
    session_keys = _pattern_session_key_values(data)
    rolling_high_values = rolling_high.to_numpy(dtype=float)
    rolling_low_values = rolling_low.to_numpy(dtype=float)
    previous_high_values = previous_reference_high.to_numpy(dtype=float)
    previous_low_values = previous_reference_low.to_numpy(dtype=float)

    events: list[_BreakStructureEvent] = []
    for index in range(1, len(data)):
        if session_keys[index] != session_keys[index - 1]:
            continue

        current_prev_high = rolling_high_values[index]
        previous_prev_high = previous_high_values[index]
        if (
            not np.isnan(current_prev_high)
            and not np.isnan(previous_prev_high)
            and float(close_values[index]) > float(current_prev_high)
            and float(close_values[index - 1]) <= float(previous_prev_high)
        ):
            base_score = 11.0
            if bool(strong_volume[index]):
                base_score += 2.0
            if bool(strong_range[index]):
                base_score += 1.5
            events.append(_BreakStructureEvent(index=index, direction=1, base_score=base_score))

        current_prev_low = rolling_low_values[index]
        previous_prev_low = previous_low_values[index]
        if (
            not np.isnan(current_prev_low)
            and not np.isnan(previous_prev_low)
            and float(close_values[index]) < float(current_prev_low)
            and float(close_values[index - 1]) >= float(previous_prev_low)
        ):
            base_score = 11.0
            if bool(strong_volume[index]):
                base_score += 2.0
            if bool(strong_range[index]):
                base_score += 1.5
            events.append(_BreakStructureEvent(index=index, direction=-1, base_score=base_score))
    return events


def _prefix_break_structure_scores(
    data: pd.DataFrame,
    *,
    breakout_lookback: int,
) -> np.ndarray:
    scores = np.zeros(len(data), dtype=float)
    events = _break_structure_events(data, breakout_lookback=breakout_lookback)
    if not events:
        return scores

    event_position = 0
    for end_index in range(len(data)):
        while event_position < len(events) and events[event_position].index <= end_index:
            event_position += 1

        score = 0.0
        for event in events[:event_position]:
            bars_ago = end_index - event.index
            candidate = float(event.base_score) - (bars_ago * 2.0)
            if event.direction > 0:
                score = max(score, candidate)
            else:
                score = min(score, -max(4.0, candidate))
        scores[end_index] = score
    return scores


def _break_structure_evidence(
    break_score: float,
    *,
    breakout_lookback: int,
) -> list[str]:
    evidence: list[str] = []
    if break_score >= 4.0:
        evidence.append(
            f"Price confirmed an upside break above the prior {breakout_lookback}-bar high."
        )
    elif break_score <= -4.0:
        evidence.append(
            f"Price confirmed a downside break below the prior {breakout_lookback}-bar low."
        )
    return evidence


def _trend_snapshot_from_arrays(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    returns: np.ndarray,
    *,
    horizon: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    breakout_lookback: int,
    raw_break_score: float,
) -> _TrendSnapshot:
    if len(closes) < 8:
        return _TrendSnapshot(score=0.0, label="Neutral", evidence=[])

    atr_scale = float(np.nanmean(highs - lows))
    atr_scale = atr_scale if atr_scale > 1e-9 else max(abs(float(closes[-1])) * 0.001, 1e-6)

    regression_x = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(regression_x, closes, 1)[0])
    slope_score = float(np.clip((slope / atr_scale) * 80.0, -24.0, 24.0))

    fast_period = min(20, max(5, len(closes) // 3))
    slow_period = min(50, max(fast_period + 4, len(closes)))
    fast_ma = float(np.mean(closes[-fast_period:]))
    slow_ma = float(np.mean(closes[-slow_period:]))
    price = float(closes[-1])

    ma_score = 0.0
    ma_separation = abs(fast_ma - slow_ma) / atr_scale
    if ma_separation >= 0.12:
        if fast_ma > slow_ma:
            ma_score += 10.0
        elif fast_ma < slow_ma:
            ma_score -= 10.0
    if price > fast_ma and price > slow_ma:
        ma_score += 5.0
    elif price < fast_ma and price < slow_ma:
        ma_score -= 5.0
    if len(closes) >= slow_period + 3:
        previous_fast = float(np.mean(closes[-fast_period - 3 : -3]))
        previous_slow = float(np.mean(closes[-slow_period - 3 : -3]))
        if fast_ma > previous_fast and slow_ma > previous_slow:
            ma_score += 5.0
        elif fast_ma < previous_fast and slow_ma < previous_slow:
            ma_score -= 5.0

    persistence_window = returns[-min(len(returns), max(8, horizon // 2)) :]
    bullish_persistence = float(np.mean(persistence_window > 0))
    bearish_persistence = float(np.mean(persistence_window < 0))
    persistence_score = (bullish_persistence - bearish_persistence) * 18.0

    swing_highs, swing_lows = _confirmed_swings_from_arrays(
        highs,
        lows,
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
    )
    swing_tolerance = max((atr_scale / max(abs(price), 1.0)) * 0.35, 0.0015)
    swing_score = 0.0
    if len(swing_highs) >= 2:
        high_direction = _relative_move(swing_highs[-2][1], swing_highs[-1][1], swing_tolerance)
        swing_score += 7.0 * high_direction
    if len(swing_lows) >= 2:
        low_direction = _relative_move(swing_lows[-2][1], swing_lows[-1][1], swing_tolerance)
        swing_score += 7.0 * low_direction

    break_evidence = _break_structure_evidence(raw_break_score, breakout_lookback=breakout_lookback)
    break_score = raw_break_score

    raw_score = slope_score + ma_score + persistence_score + swing_score + break_score
    score = float(np.clip(raw_score, -100.0, 100.0))

    evidence: list[str] = []
    if slope_score >= 6.0:
        evidence.append("Recent close regression slope was positive after range normalization.")
    elif slope_score <= -6.0:
        evidence.append("Recent close regression slope was negative after range normalization.")

    if ma_score >= 8.0:
        evidence.append("Price and moving-average alignment stayed bullish across the recent horizon.")
    elif ma_score <= -8.0:
        evidence.append("Price and moving-average alignment stayed bearish across the recent horizon.")

    if swing_score >= 7.0:
        evidence.append("Confirmed swing highs and lows were progressing upward.")
    elif swing_score <= -7.0:
        evidence.append("Confirmed swing highs and lows were progressing downward.")

    if persistence_score >= 4.0:
        evidence.append("Directional persistence favored bullish closes over the recent bars.")
    elif persistence_score <= -4.0:
        evidence.append("Directional persistence favored bearish closes over the recent bars.")

    evidence.extend(break_evidence)
    return _TrendSnapshot(score=round(score, 2), label=_trend_label(score), evidence=evidence)


def _confirmed_swings(
    window: pd.DataFrame,
    *,
    pivot_left_bars: int,
    pivot_right_bars: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    return _confirmed_swings_from_arrays(
        window["High"].astype(float).to_numpy(copy=False),
        window["Low"].astype(float).to_numpy(copy=False),
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
    )


def _break_structure_score(
    history: pd.DataFrame,
    *,
    breakout_lookback: int,
) -> tuple[float, list[str]]:
    if history.empty:
        return 0.0, []
    break_scores = _prefix_break_structure_scores(history, breakout_lookback=breakout_lookback)
    breakout_score = float(break_scores[-1]) if len(break_scores) else 0.0
    return breakout_score, _break_structure_evidence(
        breakout_score,
        breakout_lookback=breakout_lookback,
    )


def _trend_snapshot(
    history: pd.DataFrame,
    *,
    horizon: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    breakout_lookback: int,
) -> _TrendSnapshot:
    window = history.tail(min(len(history), horizon)).reset_index(drop=True)
    raw_break_score, _ = _break_structure_score(history, breakout_lookback=breakout_lookback)
    scaled_break_score = raw_break_score * (1.0 if horizon <= 60 else 0.5)
    snapshot = _trend_snapshot_from_arrays(
        window["Close"].astype(float).to_numpy(copy=False),
        window["High"].astype(float).to_numpy(copy=False),
        window["Low"].astype(float).to_numpy(copy=False),
        window["Bar_Return"].fillna(0.0).astype(float).to_numpy(),
        horizon=horizon,
        pivot_left_bars=pivot_left_bars,
        pivot_right_bars=pivot_right_bars,
        breakout_lookback=breakout_lookback,
        raw_break_score=scaled_break_score,
    )
    return snapshot


def classify_intraday_trend(
    df: pd.DataFrame,
    *,
    lookback_bars: int = 12,
    pivot_left_bars: int = 2,
    pivot_right_bars: int = 2,
    breakout_lookback: int = 20,
) -> pd.DataFrame:
    """Classify the intraday trend using recency-aware structural components."""
    pattern_df = df.copy()
    short_horizon, medium_horizon, long_horizon = _trend_horizons(lookback_bars)
    close_values = pattern_df["Close"].astype(float).to_numpy(copy=False)
    high_values = pattern_df["High"].astype(float).to_numpy(copy=False)
    low_values = pattern_df["Low"].astype(float).to_numpy(copy=False)
    return_values = pattern_df["Bar_Return"].fillna(0.0).astype(float).to_numpy()
    bullish_values = pattern_df.get("Is_Bullish", pd.Series(False, index=pattern_df.index)).fillna(False).astype(bool).to_numpy()
    bearish_values = pattern_df.get("Is_Bearish", pd.Series(False, index=pattern_df.index)).fillna(False).astype(bool).to_numpy()
    break_scores = _prefix_break_structure_scores(
        pattern_df,
        breakout_lookback=breakout_lookback,
    )
    short_scores: list[float] = []
    medium_scores: list[float] = []
    long_scores: list[float] = []
    short_labels: list[str] = []
    medium_labels: list[str] = []
    long_labels: list[str] = []
    trend_scores: list[float] = []
    trend_labels: list[str] = []
    trend_evidence: list[list[str]] = []
    trend_horizons: list[str] = []
    trend_lookbacks: list[int] = []

    for index in range(len(pattern_df)):
        def build_snapshot(horizon: int) -> _TrendSnapshot:
            start_index = max(0, index - horizon + 1)
            break_score = float(break_scores[index]) * (1.0 if horizon <= 60 else 0.5)
            return _trend_snapshot_from_arrays(
                close_values[start_index : index + 1],
                high_values[start_index : index + 1],
                low_values[start_index : index + 1],
                return_values[start_index : index + 1],
                horizon=horizon,
                pivot_left_bars=pivot_left_bars,
                pivot_right_bars=pivot_right_bars,
                breakout_lookback=breakout_lookback,
                raw_break_score=break_score,
            )

        short_snapshot = build_snapshot(short_horizon)
        medium_snapshot = build_snapshot(medium_horizon)
        long_snapshot = build_snapshot(long_horizon)

        weights: list[float] = []
        weighted_scores: list[float] = []
        available_snapshots = (
            (short_snapshot, 0.50, short_horizon),
            (medium_snapshot, 0.35, medium_horizon),
            (long_snapshot, 0.15, long_horizon),
        )
        for snapshot, weight, horizon in available_snapshots:
            if (index + 1) >= min(8, horizon):
                weights.append(weight)
                weighted_scores.append(snapshot.score * weight)

        composite_score = round(sum(weighted_scores) / sum(weights), 2) if weights else 0.0
        composite_label = _trend_label(composite_score)
        evidence = list(dict.fromkeys(short_snapshot.evidence + medium_snapshot.evidence))
        if composite_label == "Downtrend" and bool(bullish_values[index]):
            evidence.append(
                "A recent bullish candle was treated as a counter-trend reaction, not a confirmed reversal."
            )
        elif composite_label == "Uptrend" and bool(bearish_values[index]):
            evidence.append(
                "A recent bearish candle was treated as a counter-trend reaction, not a confirmed reversal."
            )
        if not evidence:
            evidence.append(
                "Slope, moving averages, swing structure, and recent breaks were too mixed to confirm a trend."
            )

        short_scores.append(short_snapshot.score)
        medium_scores.append(medium_snapshot.score)
        long_scores.append(long_snapshot.score)
        short_labels.append(short_snapshot.label)
        medium_labels.append(medium_snapshot.label)
        long_labels.append(long_snapshot.label)
        trend_scores.append(composite_score)
        trend_labels.append(composite_label)
        trend_evidence.append(evidence)
        trend_horizons.append("Short-to-medium term")
        trend_lookbacks.append(medium_horizon)

    pattern_df["Short_Term_Trend"] = short_labels
    pattern_df["Medium_Term_Trend"] = medium_labels
    pattern_df["Long_Term_Trend"] = long_labels
    pattern_df["Short_Term_Trend_Score"] = short_scores
    pattern_df["Medium_Term_Trend_Score"] = medium_scores
    pattern_df["Long_Term_Trend_Score"] = long_scores
    pattern_df["Trend"] = trend_labels
    pattern_df["Trend_Score"] = trend_scores
    pattern_df["Trend_Evidence"] = trend_evidence
    pattern_df["Trend_Horizon"] = trend_horizons
    pattern_df["Trend_Lookback_Bars"] = trend_lookbacks
    return pattern_df


def resolve_pattern_conflicts(
    raw_patterns: list[PatternEvent],
) -> tuple[list[PatternEvent], int]:
    """Deduplicate exact duplicates while preserving distinct same-bar events."""
    deduplicated: dict[tuple[object, ...], PatternEvent] = {}
    ignored_patterns_count = 0

    for pattern in raw_patterns:
        key = (
            pattern.pattern_id,
            pattern.status.value,
            pattern.detected_at.isoformat(),
            tuple(pattern.relevant_indices),
            tuple(sorted(pattern.relevant_prices.items())),
        )
        if key in deduplicated:
            ignored_patterns_count += 1
            continue
        deduplicated[key] = pattern

    resolved_patterns = sorted(
        deduplicated.values(),
        key=lambda item: (
            item.detected_at,
            item.pattern_name,
            item.status.value,
        ),
    )
    return resolved_patterns, ignored_patterns_count
