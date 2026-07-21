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
            relevant_prices=relevant_prices,
            relevant_indices=relevant_indices,
            detection_reason=detection_reason,
            signal_strength=signal_strength if signal_strength is not None else _signal_strength(row),
            base_score=base_score if base_score is not None else self.default_base_score,
            exchange_timezone=exchange_timezone,
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

        rolling_high = data["High"].rolling(
            window=config.breakout_lookback,
            min_periods=config.breakout_lookback,
        ).max().shift(1)
        previous_reference_high = rolling_high.shift(1).fillna(rolling_high)
        events: list[PatternEvent] = []
        next_allowed_index = 0

        for index in range(1, len(data)):
            if index < next_allowed_index:
                continue
            row = data.iloc[index]
            previous_row = data.iloc[index - 1]
            current_reference = rolling_high.iloc[index]
            previous_reference = previous_reference_high.iloc[index]
            if pd.isna(current_reference) or pd.isna(previous_reference):
                continue
            crossed = (
                row["Close"] > current_reference
                and previous_row["Close"] <= previous_reference
                and _safe_float(row.get("Volume_Strength")) >= config.minimum_volume_strength
            )
            if not crossed:
                continue

            is_strong = bool(row.get("Strong_Volume", False))
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
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        f"Close crossed above the prior {config.breakout_lookback}-bar high at "
                        f"{current_reference:.2f} with volume confirmation using the "
                        f"{volume_source} baseline."
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

        rolling_low = data["Low"].rolling(
            window=config.breakout_lookback,
            min_periods=config.breakout_lookback,
        ).min().shift(1)
        previous_reference_low = rolling_low.shift(1).fillna(rolling_low)
        events: list[PatternEvent] = []
        next_allowed_index = 0

        for index in range(1, len(data)):
            if index < next_allowed_index:
                continue
            row = data.iloc[index]
            previous_row = data.iloc[index - 1]
            current_reference = rolling_low.iloc[index]
            previous_reference = previous_reference_low.iloc[index]
            if pd.isna(current_reference) or pd.isna(previous_reference):
                continue
            crossed = (
                row["Close"] < current_reference
                and previous_row["Close"] >= previous_reference
                and _safe_float(row.get("Volume_Strength")) >= config.minimum_volume_strength
            )
            if not crossed:
                continue

            is_strong = bool(row.get("Strong_Volume", False))
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
                        "close": float(row["Close"]),
                    },
                    detection_reason=(
                        f"Close crossed below the prior {config.breakout_lookback}-bar low at "
                        f"{current_reference:.2f} with volume confirmation using the "
                        f"{volume_source} baseline."
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
                        signal_strength=round(valley_depth * 100.0, 2),
                    )
                )

                expiry_index = tentative_detected_index + config.double_pattern_max_separation_bars
                invalidation_level = peak_reference * (1 + config.double_pattern_price_tolerance_ratio)
                confirmed_or_failed = False
                for scan_index in range(tentative_detected_index + 1, min(len(data), expiry_index + 1)):
                    close_price = float(data.iloc[scan_index]["Close"])
                    if close_price < neckline:
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
                        signal_strength=round(peak_height * 100.0, 2),
                    )
                )

                expiry_index = tentative_detected_index + config.double_pattern_max_separation_bars
                invalidation_level = min(first_low, second_low) * (1 - config.double_pattern_price_tolerance_ratio)
                confirmed_or_failed = False
                for scan_index in range(tentative_detected_index + 1, min(len(data), expiry_index + 1)):
                    close_price = float(data.iloc[scan_index]["Close"])
                    if close_price > neckline:
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


def _confirmed_swings(
    window: pd.DataFrame,
    *,
    pivot_left_bars: int,
    pivot_right_bars: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    if len(window) < pivot_left_bars + pivot_right_bars + 1:
        return highs, lows

    for index in range(pivot_left_bars, len(window) - pivot_right_bars):
        high_value = float(window.iloc[index]["High"])
        low_value = float(window.iloc[index]["Low"])

        high_slice = window["High"].iloc[index - pivot_left_bars : index + pivot_right_bars + 1]
        low_slice = window["Low"].iloc[index - pivot_left_bars : index + pivot_right_bars + 1]

        if high_value == float(high_slice.max()) and int((high_slice == high_value).sum()) == 1:
            highs.append((index, high_value))
        if low_value == float(low_slice.min()) and int((low_slice == low_value).sum()) == 1:
            lows.append((index, low_value))

    return highs, lows


def _break_structure_score(
    history: pd.DataFrame,
    *,
    breakout_lookback: int,
) -> tuple[float, list[str]]:
    evidence: list[str] = []
    if len(history) < max(breakout_lookback + 1, 3):
        return 0.0, evidence

    recent_history = history.tail(max(6, min(12, breakout_lookback // 2 + 6))).reset_index(drop=True)
    breakout_score = 0.0
    for index in range(1, len(recent_history)):
        previous_row = recent_history.iloc[index - 1]
        row = recent_history.iloc[index]

        current_prev_high = row.get("Rolling_High_20_Bars")
        previous_prev_high = previous_row.get("Rolling_High_20_Bars")
        current_prev_low = row.get("Rolling_Low_20_Bars")
        previous_prev_low = previous_row.get("Rolling_Low_20_Bars")

        if (
            pd.notna(current_prev_high)
            and pd.notna(previous_prev_high)
            and row["Close"] > current_prev_high
            and previous_row["Close"] <= previous_prev_high
        ):
            bars_ago = len(recent_history) - 1 - index
            breakout_score = max(
                breakout_score,
                11.0 - (bars_ago * 2.0) + (2.0 if bool(row.get("Strong_Volume", False)) else 0.0),
            )
        if (
            pd.notna(current_prev_low)
            and pd.notna(previous_prev_low)
            and row["Close"] < current_prev_low
            and previous_row["Close"] >= previous_prev_low
        ):
            bars_ago = len(recent_history) - 1 - index
            bearish_break_score = max(
                4.0,
                11.0 - (bars_ago * 2.0) + (2.0 if bool(row.get("Strong_Volume", False)) else 0.0),
            )
            breakout_score = min(
                breakout_score,
                -bearish_break_score,
            )

    if breakout_score >= 4.0:
        evidence.append(
            f"Price confirmed an upside break above the prior {breakout_lookback}-bar high."
        )
    elif breakout_score <= -4.0:
        evidence.append(
            f"Price confirmed a downside break below the prior {breakout_lookback}-bar low."
        )
    return breakout_score, evidence


def _trend_snapshot(
    history: pd.DataFrame,
    *,
    horizon: int,
    pivot_left_bars: int,
    pivot_right_bars: int,
    breakout_lookback: int,
) -> _TrendSnapshot:
    window = history.tail(min(len(history), horizon)).reset_index(drop=True)
    if len(window) < 8:
        return _TrendSnapshot(score=0.0, label="Neutral", evidence=[])

    closes = window["Close"].astype(float).to_numpy()
    highs = window["High"].astype(float).to_numpy()
    lows = window["Low"].astype(float).to_numpy()
    returns = window["Bar_Return"].fillna(0.0).astype(float).to_numpy()
    atr_scale = float(np.nanmean(highs - lows))
    atr_scale = atr_scale if atr_scale > 1e-9 else max(abs(float(closes[-1])) * 0.001, 1e-6)

    regression_x = np.arange(len(closes), dtype=float)
    slope = float(np.polyfit(regression_x, closes, 1)[0])
    slope_score = float(np.clip((slope / atr_scale) * 80.0, -24.0, 24.0))

    fast_period = min(20, max(5, len(window) // 3))
    slow_period = min(50, max(fast_period + 4, len(window)))
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

    swing_highs, swing_lows = _confirmed_swings(
        window,
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

    break_score, break_evidence = _break_structure_score(
        history,
        breakout_lookback=breakout_lookback,
    )
    break_score *= 1.0 if horizon <= 60 else 0.5

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
        history = pattern_df.iloc[: index + 1].copy(deep=False)
        short_snapshot = _trend_snapshot(
            history,
            horizon=short_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )
        medium_snapshot = _trend_snapshot(
            history,
            horizon=medium_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )
        long_snapshot = _trend_snapshot(
            history,
            horizon=long_horizon,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=breakout_lookback,
        )

        weights: list[float] = []
        weighted_scores: list[float] = []
        available_snapshots = (
            (short_snapshot, 0.50, short_horizon),
            (medium_snapshot, 0.35, medium_horizon),
            (long_snapshot, 0.15, long_horizon),
        )
        for snapshot, weight, horizon in available_snapshots:
            if len(history) >= min(8, horizon):
                weights.append(weight)
                weighted_scores.append(snapshot.score * weight)

        composite_score = round(sum(weighted_scores) / sum(weights), 2) if weights else 0.0
        composite_label = _trend_label(composite_score)
        evidence = list(dict.fromkeys(short_snapshot.evidence + medium_snapshot.evidence))
        latest_row = history.iloc[-1]
        if composite_label == "Downtrend" and bool(latest_row.get("Is_Bullish", False)):
            evidence.append(
                "A recent bullish candle was treated as a counter-trend reaction, not a confirmed reversal."
            )
        elif composite_label == "Uptrend" and bool(latest_row.get("Is_Bearish", False)):
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
