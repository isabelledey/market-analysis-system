"""Registry-based rule detectors for candlestick and chart patterns."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Protocol

import numpy as np
import pandas as pd

from stock_pattern_model.config import PatternConfig
from stock_pattern_model.domain import PatternEvent, PatternFamily, PatternStatus
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


def _long_lower_rejection_geometry(row: pd.Series) -> bool:
    candle_range = row["Candle_Range"]
    if candle_range == 0 or pd.isna(candle_range):
        return False
    close_location = (row["Close"] - row["Low"]) / candle_range
    return bool(
        row["Lower_Wick_Ratio"] >= 0.55
        and row["Body_Ratio"] <= 0.35
        and close_location >= 0.60
        and bool(row["Is_Significant_Candle"])
    )


def _long_upper_rejection_geometry(row: pd.Series) -> bool:
    candle_range = row["Candle_Range"]
    if candle_range == 0 or pd.isna(candle_range):
        return False
    close_location = (row["Close"] - row["Low"]) / candle_range
    return bool(
        row["Upper_Wick_Ratio"] >= 0.55
        and row["Body_Ratio"] <= 0.35
        and close_location <= 0.40
        and bool(row["Is_Significant_Candle"])
    )


def _doji_geometry(row: pd.Series, config: PatternConfig) -> bool:
    candle_range = row["Candle_Range"]
    if candle_range == 0 or pd.isna(candle_range):
        return False
    return bool(row["Body_Ratio"] <= config.doji_body_ratio_max)


@dataclass(frozen=True)
class _PriorPatternContext:
    direction: str
    strength: float
    lookback_bars: int
    regression_score: float
    atr_normalized_move: float
    swing_context: str
    quality_passed: bool
    reason: str
    pattern_entry_trend: str = "Ambiguous"
    pattern_entry_trend_score: float = 0.0


_PATTERN_ENTRY_TREND_LABELS: dict[str, str] = {
    "uptrend": "Uptrend",
    "downtrend": "Downtrend",
    "sideways": "Sideways",
    "ambiguous": "Ambiguous",
}


def _context_direction_matches(direction: str, expected_direction: str | None) -> bool:
    if expected_direction is None:
        return direction in {"uptrend", "downtrend"}
    return direction == expected_direction


def _recent_resistance_reference(
    data: pd.DataFrame,
    pattern_index: int,
    lookback_bars: int,
) -> float | None:
    """Highest High in the causal prefix window: a proxy for a recent swing high / resistance area.

    Excludes the pattern candle itself so a shooting star cannot "create" its own resistance.
    """
    start_index = max(0, pattern_index - lookback_bars)
    window = data.iloc[start_index:pattern_index]
    if window.empty:
        return None
    return float(window["High"].max())


def _recent_support_reference(
    data: pd.DataFrame,
    pattern_index: int,
    lookback_bars: int,
) -> float | None:
    """Lowest Low in the causal prefix window: a proxy for a recent swing low / support area.

    Excludes the pattern candle itself, mirroring `_recent_resistance_reference`.
    """
    start_index = max(0, pattern_index - lookback_bars)
    window = data.iloc[start_index:pattern_index]
    if window.empty:
        return None
    return float(window["Low"].min())


@dataclass(frozen=True)
class _LowerWickRejectionOutcome:
    context: _PriorPatternContext
    near_support: bool
    support_reference: float | None
    classic_gate_passed: bool
    context_validated_state: bool


def _evaluate_lower_wick_rejection(
    data: pd.DataFrame,
    pattern_index: int,
    row: pd.Series,
    config: PatternConfig,
) -> _LowerWickRejectionOutcome:
    """Determine whether a lower-wick rejection candle enters the staged confirmation lifecycle.

    `classic_gate_passed` mirrors the existing "genuine preceding downtrend" check (used for the
    classic Hammer/Bullish Pin Bar name). `context_validated_state` is the broader, looser gate
    described for Stage 4: near a recent low/support area OR a genuinely displaced preceding
    downtrend -- either alone is enough to enter the state machine, even when the classic name
    does not apply. The actual awaiting/directionally_confirmed/invalidated resolution is left to
    `analysis._apply_generic_pattern_lifecycle`, which already forward-scans every tracked event
    for retest/invalidation; Stage 4 extends that existing scan with a directional-confirmation
    check rather than duplicating a second, inconsistent forward scan here.
    """
    context = classify_prior_pattern_context(
        data,
        pattern_index,
        config,
        expected_direction="downtrend",
    )
    support_reference = _recent_support_reference(data, pattern_index, config.resistance_proximity_lookback_bars)
    candle_low = float(row["Low"])
    near_support = (
        support_reference is not None
        and candle_low <= support_reference * (1.0 + config.support_resistance_proximity_tolerance)
    )
    context_validated_state = context.quality_passed or near_support

    return _LowerWickRejectionOutcome(
        context=context,
        near_support=near_support,
        support_reference=support_reference,
        classic_gate_passed=context.quality_passed,
        context_validated_state=context_validated_state,
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
        geometry_label: str = "unknown",
        context_tags: tuple[str, ...] = (),
        context_bias: str = "Neutral",
        context_quality: str = "unknown",
        pattern_entry_trend: str = "Not Applicable",
        pattern_entry_trend_score: float = 0.0,
        pattern_entry_trend_lookback_bars: int = 0,
        rejection_confirmation_state: str = "not_applicable",
        dampener_eligible: bool = False,
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
            geometry_label=geometry_label,
            context_tags=context_tags,
            context_bias=context_bias,
            context_quality=context_quality,
            pattern_entry_trend=pattern_entry_trend,
            pattern_entry_trend_score=pattern_entry_trend_score,
            pattern_entry_trend_lookback_bars=pattern_entry_trend_lookback_bars,
            rejection_confirmation_state=rejection_confirmation_state,
            dampener_eligible=dampener_eligible,
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
        return _detect_lower_wick_rejection(
            self,
            data,
            config,
            interval,
            classic_pattern_name="Bullish Pin Bar",
            classic_geometry_phrase="bullish pin-bar geometry",
        )


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
        return _detect_lower_wick_rejection(
            self,
            data,
            config,
            interval,
            classic_pattern_name="Hammer",
            classic_geometry_phrase="hammer geometry",
        )


def _detect_lower_wick_rejection(
    detector: BasePatternDetector,
    data: pd.DataFrame,
    config: PatternConfig,
    interval: str,
    *,
    classic_pattern_name: str,
    classic_geometry_phrase: str,
) -> list[PatternEvent]:
    """Shared staged-confirmation detection for Hammer and Bullish Pin Bar.

    Both detectors check the same geometry and the same backward context; they only differ in
    what the classic (context-validated) name is. Detection only resolves the backward-looking
    half of the state machine (`detected` vs `context_validated`); the forward-looking half
    (awaiting directional confirmation, directionally confirmed, or invalidated) is resolved by
    `analysis._apply_generic_pattern_lifecycle`, which re-scans forward on every analysis run --
    unlike a one-shot detector, it can discover confirmation/invalidation that occurred well after
    the original detection.
    """
    events: list[PatternEvent] = []
    for index, row in data.iterrows():
        if not _long_lower_rejection_geometry(row):
            continue

        pattern_index = int(index)
        outcome = _evaluate_lower_wick_rejection(data, pattern_index, row, config)
        volume_source = str(row.get("Volume_Baseline_Source", "unknown"))
        context = outcome.context

        if not outcome.context_validated_state:
            bias = "Neutral"
            status = PatternStatus.CANDIDATE
            pattern_name = "Lower-Wick Rejection"
            context_quality = "geometry_only"
            dampener_eligible = False
            rejection_confirmation_state = "detected"
            detection_reason = (
                f"A long lower wick and strong close matched {classic_geometry_phrase}, but the "
                f"prefix-only prior context did not qualify as a downtrend and the candle was not "
                f"near a recent support area ({context.reason}), so it remains a neutral "
                f"geometry-only candidate with the {volume_source} volume baseline."
            )
        else:
            bias = "Bullish"
            status = PatternStatus.TENTATIVE
            dampener_eligible = True
            rejection_confirmation_state = "context_validated"
            pattern_name = classic_pattern_name if outcome.classic_gate_passed else "Lower-Wick Rejection"
            context_quality = "validated" if outcome.classic_gate_passed else "geometry_only"
            context_basis = (
                f"a qualifying prefix-only downtrend ({context.reason})"
                if outcome.classic_gate_passed
                else f"proximity to a recent support level ({outcome.support_reference:.2f})"
            )
            detection_reason = (
                f"A long lower wick and strong close matched {classic_geometry_phrase} after "
                f"{context_basis}. A later close above the rejection candle's high would "
                "directionally confirm the reversal; a close below its low would invalidate it. "
                "Until then this stays a bounded, unconfirmed bullish signal."
            )

        relevant_prices = {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
        if outcome.support_reference is not None:
            relevant_prices["support_reference"] = outcome.support_reference

        events.append(
            detector._build_event(
                data,
                interval,
                pattern_index,
                pattern_start_index=pattern_index,
                relevant_indices=[pattern_index],
                relevant_prices=relevant_prices,
                detection_reason=detection_reason,
                status=status,
                bias=bias,
                pattern_name=pattern_name,
                geometry_label="long_lower_rejection",
                context_tags=(
                    "single_candle",
                    "bullish_rejection",
                    "context_validated" if outcome.classic_gate_passed else "geometry_first",
                    f"prior_context_{context.direction}",
                    f"rejection_state_{rejection_confirmation_state}",
                ),
                context_bias="Bullish" if bias == "Bullish" else "Neutral",
                context_quality=context_quality,
                pattern_entry_trend=context.pattern_entry_trend,
                pattern_entry_trend_score=context.pattern_entry_trend_score,
                pattern_entry_trend_lookback_bars=context.lookback_bars,
                rejection_confirmation_state=rejection_confirmation_state,
                dampener_eligible=dampener_eligible,
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
            if not _long_upper_rejection_geometry(row):
                continue

            pattern_index = int(index)
            context = classify_prior_pattern_context(
                data,
                pattern_index,
                config,
                expected_direction="uptrend",
            )
            volume_source = str(row.get("Volume_Baseline_Source", "unknown"))

            resistance_reference = _recent_resistance_reference(
                data,
                pattern_index,
                config.resistance_proximity_lookback_bars,
            )
            candle_high = float(row["High"])
            near_resistance = (
                resistance_reference is not None
                and candle_high >= resistance_reference * (1.0 - config.support_resistance_proximity_tolerance)
            )
            local_trend_before_candle = (
                str(data.iloc[pattern_index - 1].get("Local_Trend", "Neutral"))
                if pattern_index > 0
                else "Neutral"
            )
            contradicted_by_local_trend = local_trend_before_candle == "Downtrend"

            # A classic Shooting Star requires all three: a genuine, sufficiently displaced
            # preceding uptrend (context.quality_passed); the rejection occurring at/above a
            # recent swing high or resistance level (near_resistance); and no contradictory
            # local-session downtrend already in force heading into the candle.
            classic_shooting_star = (
                context.quality_passed and near_resistance and not contradicted_by_local_trend
            )

            resistance_note = (
                f"price ({candle_high:.2f}) was near the recent {resistance_reference:.2f} "
                "swing-high/resistance level"
                if resistance_reference is not None
                else "no recent swing-high/resistance level could be established"
            )

            if classic_shooting_star:
                pattern_name = "Shooting Star"
                bias = "Bearish"
                status = PatternStatus.TENTATIVE
                context_quality = "validated"
                role_tag = "shooting_star"
                detection_reason = (
                    "A long upper wick and weak close matched shooting-star geometry, the "
                    f"prefix-only prior context qualified as a genuine uptrend ({context.reason}), "
                    f"{resistance_note}, and no contradictory local downtrend was present. A later "
                    "close below the rejection candle's low would directionally confirm the "
                    "reversal; a close above its high would invalidate it. Until then this stays a "
                    "bounded, unconfirmed bearish signal."
                )
            elif contradicted_by_local_trend:
                pattern_name = "Upper-Wick Rejection"
                bias = "Bearish"
                status = PatternStatus.TENTATIVE
                context_quality = "geometry_only"
                role_tag = "bearish_continuation_rejection"
                detection_reason = (
                    "A long upper wick and weak close matched shooting-star geometry, but the local "
                    "session trend was already a downtrend heading into the candle, so this is a "
                    "bearish upper-wick rejection during a short-term decline rather than a reversal "
                    f"from a genuine uptrend ({context.reason}). A later close below the rejection "
                    "candle's low would directionally confirm continued weakness; until then this "
                    f"stays a bounded, unconfirmed bearish signal, with the {volume_source} volume "
                    "baseline."
                )
            elif near_resistance:
                pattern_name = "Resistance Rejection"
                bias = "Neutral"
                status = PatternStatus.CANDIDATE
                context_quality = "geometry_only"
                role_tag = "resistance_rejection"
                detection_reason = (
                    f"A long upper wick and weak close matched shooting-star geometry and {resistance_note}, "
                    f"but the preceding move did not qualify as a genuine uptrend ({context.reason}), so this "
                    f"is a resistance-zone rejection rather than a classic shooting star, with the "
                    f"{volume_source} volume baseline."
                )
            else:
                pattern_name = "Upper-Wick Rejection"
                bias = "Neutral"
                status = PatternStatus.CANDIDATE
                context_quality = "geometry_only"
                role_tag = "geometry_first"
                detection_reason = (
                    "A long upper wick and weak close matched shooting-star geometry, but the "
                    f"prefix-only prior context did not qualify as an uptrend near resistance "
                    f"({context.reason}; {resistance_note}), so the candle remains a neutral "
                    f"geometry-only candidate with the {volume_source} volume baseline."
                )

            relevant_prices = {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
            if resistance_reference is not None:
                relevant_prices["resistance_reference"] = resistance_reference

            dampener_eligible = bias == "Bearish"
            rejection_confirmation_state = "context_validated" if dampener_eligible else "detected"

            events.append(
                self._build_event(
                    data,
                    interval,
                    pattern_index,
                    pattern_start_index=pattern_index,
                    relevant_indices=[pattern_index],
                    relevant_prices=relevant_prices,
                    detection_reason=detection_reason,
                    status=status,
                    bias=bias,
                    pattern_name=pattern_name,
                    geometry_label="long_upper_rejection",
                    context_tags=(
                        "single_candle",
                        "bearish_rejection",
                        f"prior_context_{context.direction}",
                        f"local_trend_{local_trend_before_candle.lower()}",
                        "near_resistance" if near_resistance else "away_from_resistance",
                        role_tag,
                    ),
                    context_bias="Bearish" if bias == "Bearish" else "Neutral",
                    context_quality=context_quality,
                    pattern_entry_trend=context.pattern_entry_trend,
                    pattern_entry_trend_score=context.pattern_entry_trend_score,
                    pattern_entry_trend_lookback_bars=context.lookback_bars,
                    rejection_confirmation_state=rejection_confirmation_state,
                    dampener_eligible=dampener_eligible,
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
            if not _doji_geometry(row, config):
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
                    status=PatternStatus.CANDIDATE,
                    bias="Neutral",
                    signal_strength=round(1.0 - min(float(row["Body_Ratio"]), 1.0), 2),
                    geometry_label="doji",
                    context_tags=("single_candle", "indecision", "geometry_first"),
                    context_bias="Neutral",
                    context_quality="geometry_only",
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
                        "star_low": float(second["Low"]),
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
                        "star_high": float(second["High"]),
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

    def register(self, detector: PatternDetector) -> PatternRegistry:
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
    evidence_items: list[dict[str, object]]


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


def classify_prior_pattern_context(
    data: pd.DataFrame,
    pattern_index: int,
    config: PatternConfig,
    *,
    expected_direction: str | None = None,
) -> _PriorPatternContext:
    prefix = data.iloc[:pattern_index].copy().reset_index(drop=True)
    if len(prefix) < 8:
        return _PriorPatternContext(
            direction="ambiguous",
            strength=0.0,
            lookback_bars=len(prefix),
            regression_score=0.0,
            atr_normalized_move=0.0,
            swing_context="insufficient_history",
            quality_passed=False,
            reason="fewer than 8 completed bars existed before the pattern candle",
            pattern_entry_trend="Ambiguous",
            pattern_entry_trend_score=0.0,
        )

    lookback_bars = min(len(prefix), max(8, config.pattern_entry_trend_lookback_bars))
    window = prefix.tail(lookback_bars).reset_index(drop=True)
    closes = window["Close"].astype(float).to_numpy(copy=False)
    highs = window["High"].astype(float).to_numpy(copy=False)
    lows = window["Low"].astype(float).to_numpy(copy=False)

    atr_scale = float(np.nanmean(highs - lows))
    atr_scale = atr_scale if atr_scale > 1e-9 else max(abs(float(closes[-1])) * 0.001, 1e-6)
    regression_x = np.arange(len(closes), dtype=float)
    regression_slope = float(np.polyfit(regression_x, closes, 1)[0])
    regression_score = float(np.clip((regression_slope / atr_scale) * 80.0, -24.0, 24.0))
    atr_normalized_move = round(float((closes[-1] - closes[0]) / atr_scale), 2)

    swing_highs, swing_lows = _confirmed_swings_from_arrays(
        highs,
        lows,
        pivot_left_bars=config.pivot_left_bars,
        pivot_right_bars=config.pivot_right_bars,
    )
    high_direction = 0
    low_direction = 0
    if len(swing_highs) >= 2:
        high_direction = _relative_move(swing_highs[-2][1], swing_highs[-1][1], 0.003)
    if len(swing_lows) >= 2:
        low_direction = _relative_move(swing_lows[-2][1], swing_lows[-1][1], 0.003)
    if high_direction > 0 and low_direction > 0:
        swing_context = "higher_highs_and_higher_lows"
    elif high_direction < 0 and low_direction < 0:
        swing_context = "lower_highs_and_lower_lows"
    elif high_direction == 0 and low_direction == 0:
        swing_context = "mixed_or_flat"
    else:
        swing_context = "mixed_swings"

    break_score, _ = _break_structure_score(
        prefix,
        breakout_lookback=config.breakout_lookback,
    )
    snapshot = _trend_snapshot(
        prefix,
        horizon=lookback_bars,
        pivot_left_bars=config.pivot_left_bars,
        pivot_right_bars=config.pivot_right_bars,
        breakout_lookback=config.breakout_lookback,
    )

    minimum_displacement = config.context_minimum_displacement_atr
    if snapshot.label == "Uptrend" and atr_normalized_move >= minimum_displacement:
        direction = "uptrend"
    elif snapshot.label == "Downtrend" and atr_normalized_move <= -minimum_displacement:
        direction = "downtrend"
    elif abs(snapshot.score) >= 16.0 and abs(atr_normalized_move) >= minimum_displacement * 0.8:
        direction = "uptrend" if snapshot.score > 0 else "downtrend"
    elif abs(snapshot.score) >= 10.0:
        direction = "sideways"
    else:
        direction = "ambiguous"

    quality_passed = (
        _context_direction_matches(direction, expected_direction)
        and abs(snapshot.score) >= 18.0
        and abs(regression_score) >= 6.0
        and abs(atr_normalized_move) >= minimum_displacement
    )
    reason = (
        f"{direction} context over {lookback_bars} prior bars "
        f"(trend score {snapshot.score:.2f}, regression {regression_score:.2f}, "
        f"ATR-normalized move {atr_normalized_move:.2f}, swing context {swing_context}, "
        f"break score {break_score:.2f})"
    )
    return _PriorPatternContext(
        direction=direction,
        strength=round(abs(snapshot.score), 2),
        lookback_bars=lookback_bars,
        regression_score=round(regression_score, 2),
        atr_normalized_move=atr_normalized_move,
        swing_context=swing_context,
        quality_passed=quality_passed,
        reason=reason,
        pattern_entry_trend=_PATTERN_ENTRY_TREND_LABELS.get(direction, "Ambiguous"),
        pattern_entry_trend_score=round(snapshot.score, 2),
    )


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
            decayed_magnitude = max(0.0, float(event.base_score) - (bars_ago * 2.0))
            signed_candidate = float(event.direction) * decayed_magnitude
            if signed_candidate > 0:
                score = max(score, signed_candidate)
            elif signed_candidate < 0:
                score = min(score, signed_candidate)
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
    horizon_label: str = "Composite",
) -> _TrendSnapshot:
    if len(closes) < 8:
        return _TrendSnapshot(score=0.0, label="Neutral", evidence=[], evidence_items=[])

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
    evidence_items: list[dict[str, object]] = []

    def append_evidence(
        *,
        evidence_type: str,
        direction: str,
        raw_feature_value: float,
        score_contribution: float,
        explanation: str,
    ) -> None:
        evidence.append(explanation)
        evidence_items.append(
            {
                "type": evidence_type,
                "direction": direction,
                "horizon": horizon_label,
                "raw_feature_value": round(raw_feature_value, 4),
                "score_contribution": round(score_contribution, 2),
                "evaluation_index": int(len(closes) - 1),
                "supports_composite_trend": False,
                "conflicts_with_composite_trend": False,
                "explanation": explanation,
            }
        )

    if slope_score >= 6.0:
        append_evidence(
            evidence_type="regression_slope",
            direction="Bullish",
            raw_feature_value=slope,
            score_contribution=slope_score,
            explanation="Recent close regression slope was positive after range normalization.",
        )
    elif slope_score <= -6.0:
        append_evidence(
            evidence_type="regression_slope",
            direction="Bearish",
            raw_feature_value=slope,
            score_contribution=slope_score,
            explanation="Recent close regression slope was negative after range normalization.",
        )

    if ma_score >= 8.0:
        append_evidence(
            evidence_type="moving_average_alignment",
            direction="Bullish",
            raw_feature_value=ma_separation,
            score_contribution=ma_score,
            explanation="Price and moving-average alignment stayed bullish across the recent horizon.",
        )
    elif ma_score <= -8.0:
        append_evidence(
            evidence_type="moving_average_alignment",
            direction="Bearish",
            raw_feature_value=ma_separation,
            score_contribution=ma_score,
            explanation="Price and moving-average alignment stayed bearish across the recent horizon.",
        )

    if swing_score >= 7.0:
        append_evidence(
            evidence_type="swing_structure",
            direction="Bullish",
            raw_feature_value=swing_score,
            score_contribution=swing_score,
            explanation="Confirmed swing highs and lows were progressing upward.",
        )
    elif swing_score <= -7.0:
        append_evidence(
            evidence_type="swing_structure",
            direction="Bearish",
            raw_feature_value=swing_score,
            score_contribution=swing_score,
            explanation="Confirmed swing highs and lows were progressing downward.",
        )

    if persistence_score >= 4.0:
        append_evidence(
            evidence_type="directional_persistence",
            direction="Bullish",
            raw_feature_value=bullish_persistence - bearish_persistence,
            score_contribution=persistence_score,
            explanation="Directional persistence favored bullish closes over the recent bars.",
        )
    elif persistence_score <= -4.0:
        append_evidence(
            evidence_type="directional_persistence",
            direction="Bearish",
            raw_feature_value=bullish_persistence - bearish_persistence,
            score_contribution=persistence_score,
            explanation="Directional persistence favored bearish closes over the recent bars.",
        )

    if break_score >= 4.0:
        append_evidence(
            evidence_type="break_structure",
            direction="Bullish",
            raw_feature_value=break_score,
            score_contribution=break_score,
            explanation=break_evidence[0],
        )
    elif break_score <= -4.0:
        append_evidence(
            evidence_type="break_structure",
            direction="Bearish",
            raw_feature_value=break_score,
            score_contribution=break_score,
            explanation=break_evidence[0],
        )

    return _TrendSnapshot(
        score=round(score, 2),
        label=_trend_label(score),
        evidence=evidence,
        evidence_items=evidence_items,
    )


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
    trend_evidence_structured: list[list[dict[str, object]]] = []
    trend_horizons: list[str] = []
    trend_lookbacks: list[int] = []

    for index in range(len(pattern_df)):
        def build_snapshot(horizon: int, index: int = index) -> _TrendSnapshot:
            # `index` is bound as a default argument (not read from the enclosing loop) so this
            # closure can never accidentally capture a later iteration's value if it were ever
            # stored and called outside its own loop iteration.
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
                horizon_label=(
                    "Short-term"
                    if horizon == short_horizon
                    else ("Medium-term" if horizon == medium_horizon else "Long-term")
                ),
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
        structured_evidence: list[dict[str, object]] = []
        for snapshot in (short_snapshot, medium_snapshot, long_snapshot):
            for item in snapshot.evidence_items:
                enriched_item = dict(item)
                if composite_label == "Uptrend":
                    enriched_item["supports_composite_trend"] = item["direction"] == "Bullish"
                    enriched_item["conflicts_with_composite_trend"] = item["direction"] == "Bearish"
                elif composite_label == "Downtrend":
                    enriched_item["supports_composite_trend"] = item["direction"] == "Bearish"
                    enriched_item["conflicts_with_composite_trend"] = item["direction"] == "Bullish"
                structured_evidence.append(enriched_item)

        evidence = [
            f"[{item['horizon']}, {item['direction']}"
            f"{' Conflict' if item['conflicts_with_composite_trend'] else ''}] {item['explanation']}"
            for item in structured_evidence
        ]
        if not structured_evidence:
            neutral_item = {
                "type": "composite_neutrality",
                "direction": "Neutral",
                "horizon": "Composite",
                "raw_feature_value": 0.0,
                "score_contribution": 0.0,
                "evaluation_index": int(index),
                "supports_composite_trend": False,
                "conflicts_with_composite_trend": False,
                "explanation": "Slope, moving averages, swing structure, and recent breaks were too mixed to confirm a trend.",
            }
            structured_evidence.append(neutral_item)
            evidence = [f"[Composite, Neutral] {neutral_item['explanation']}"]

        short_scores.append(short_snapshot.score)
        medium_scores.append(medium_snapshot.score)
        long_scores.append(long_snapshot.score)
        short_labels.append(short_snapshot.label)
        medium_labels.append(medium_snapshot.label)
        long_labels.append(long_snapshot.label)
        trend_scores.append(composite_score)
        trend_labels.append(composite_label)
        trend_evidence.append(evidence)
        trend_evidence_structured.append(structured_evidence)
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
    pattern_df["Trend_Evidence_Structured"] = trend_evidence_structured
    pattern_df["Trend_Horizon"] = trend_horizons
    pattern_df["Trend_Lookback_Bars"] = trend_lookbacks
    return pattern_df


def classify_local_session_trend(
    df: pd.DataFrame,
    *,
    lookback_bars: int = 20,
    pivot_left_bars: int = 2,
    pivot_right_bars: int = 2,
) -> pd.DataFrame:
    """Classify a session-anchored local trend, independent of the broad multi-horizon trend.

    Unlike ``classify_intraday_trend`` (a weighted blend of short/medium/long horizons that can
    stay "Uptrend" on the strength of older history), this evaluates only the current session,
    capped to ``lookback_bars``, so a sharp intraday reversal is never masked by a bullish broad
    trend. It combines: (1) the same slope/moving-average/persistence/swing-structure evidence used
    for the broad trend, scoped to the local window; (2) cumulative return from the session open,
    ATR-normalized; and (3) the candle's position within the session's high/low range so far.
    """
    pattern_df = df.copy()
    close_values = pattern_df["Close"].astype(float).to_numpy(copy=False)
    high_values = pattern_df["High"].astype(float).to_numpy(copy=False)
    low_values = pattern_df["Low"].astype(float).to_numpy(copy=False)
    return_values = pattern_df["Bar_Return"].fillna(0.0).astype(float).to_numpy()
    session_open_values = pattern_df["Session_Open"].astype(float).to_numpy(copy=False)
    session_high_values = pattern_df["Session_High"].astype(float).to_numpy(copy=False)
    session_low_values = pattern_df["Session_Low"].astype(float).to_numpy(copy=False)
    session_position = pattern_df.groupby("Session_Date", sort=False).cumcount().to_numpy()

    labels: list[str] = []
    scores: list[float] = []
    evidence_all: list[list[str]] = []
    lookbacks: list[int] = []

    for index in range(len(pattern_df)):
        window_len = min(lookback_bars, int(session_position[index]) + 1)
        start_index = index - window_len + 1

        structure_snapshot = _trend_snapshot_from_arrays(
            close_values[start_index : index + 1],
            high_values[start_index : index + 1],
            low_values[start_index : index + 1],
            return_values[start_index : index + 1],
            horizon=window_len,
            pivot_left_bars=pivot_left_bars,
            pivot_right_bars=pivot_right_bars,
            breakout_lookback=window_len,
            raw_break_score=0.0,
            horizon_label="Local Session",
        )

        close = float(close_values[index])
        session_open = float(session_open_values[index])
        session_high = float(session_high_values[index])
        session_low = float(session_low_values[index])
        window_highs = high_values[start_index : index + 1]
        window_lows = low_values[start_index : index + 1]
        atr_scale = float(np.nanmean(window_highs - window_lows))
        atr_scale = atr_scale if atr_scale > 1e-9 else max(abs(close) * 0.001, 1e-6)

        # Internal scoring weights (not user-facing config): scaled to sit on the same -100..100
        # composite scale as the structural snapshot above and the broad trend's own components
        # (_trend_snapshot_from_arrays), so the three additive terms below are comparable in
        # magnitude. Only the *lookback* is a tunable knob (local_trend_lookback_bars); these
        # weights are fixed, matching how the broad trend's own slope/MA/persistence weights are
        # fixed constants rather than configuration.
        cumulative_return_atr = (close - session_open) / atr_scale if session_open else 0.0
        cumulative_return_score = float(np.clip(cumulative_return_atr * 6.0, -22.0, 22.0))

        session_range = session_high - session_low
        range_position = (close - session_low) / session_range if session_range > 1e-9 else 0.5
        range_position_score = (range_position - 0.5) * 2.0 * 12.0

        evidence: list[str] = list(structure_snapshot.evidence)
        if abs(cumulative_return_score) >= 6.0:
            direction_word = "Bullish" if cumulative_return_score > 0 else "Bearish"
            evidence.append(
                f"[Local Session, {direction_word}] Cumulative return from the session open was "
                f"{'positive' if cumulative_return_score > 0 else 'negative'} "
                f"({cumulative_return_atr:.2f} ATR)."
            )
        if abs(range_position_score) >= 6.0:
            direction_word = "Bullish" if range_position_score > 0 else "Bearish"
            evidence.append(
                f"[Local Session, {direction_word}] Price was trading near the session "
                f"{'high' if range_position_score > 0 else 'low'} "
                f"({range_position * 100:.0f}% of the session range so far)."
            )

        raw_score = structure_snapshot.score + cumulative_return_score + range_position_score
        score = float(np.clip(raw_score, -100.0, 100.0))
        label = _trend_label(score)
        if not evidence:
            evidence = [
                "Slope, session position, and swing structure were too mixed to confirm a local trend."
            ]

        labels.append(label)
        scores.append(round(score, 2))
        evidence_all.append(evidence)
        lookbacks.append(window_len)

    pattern_df["Local_Trend"] = labels
    pattern_df["Local_Trend_Score"] = scores
    pattern_df["Local_Trend_Evidence"] = evidence_all
    pattern_df["Local_Trend_Lookback_Bars"] = lookbacks
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


_STRUCTURAL_PATTERN_IDS = {"double_top", "double_bottom"}


def _structural_canonical_key(event: PatternEvent) -> tuple[object, ...]:
    """Identity for "the same underlying structural setup at the same lifecycle stage".

    `setup_completion_at` is tied to a specific bar index (the second confirmed swing), so it is
    an exact, discrete anchor -- not something that needs float tolerance -- and is what a double
    top/bottom detector's own duplication bug shares across near-duplicate detections (multiple
    qualifying earlier swings all pairing with the same later swing and peak). `confirmation_at`
    and `status` distinguish genuinely different lifecycle moments (tentative vs. confirmed vs.
    failed vs. expired) of that same setup, which must never be collapsed into one another.
    """
    setup_completion = event.setup_completion_at or event.pattern_end_at
    confirmation = event.confirmation_at
    return (
        event.pattern_id,
        event.bias,
        setup_completion.isoformat(),
        confirmation.isoformat() if confirmation is not None else None,
        event.status.value,
    )


def _prices_within_tolerance(
    prices_a: dict[str, float],
    prices_b: dict[str, float],
    *,
    ratio: float,
) -> bool:
    shared_keys = set(prices_a) & set(prices_b)
    if not shared_keys:
        return True
    for key in shared_keys:
        value_a = float(prices_a[key])
        value_b = float(prices_b[key])
        reference = max(abs(value_a), abs(value_b), 1e-9)
        if abs(value_a - value_b) / reference > ratio:
            return False
    return True


def _cluster_events_by_price_tolerance(
    events: list[PatternEvent],
    *,
    ratio: float,
) -> list[list[PatternEvent]]:
    clusters: list[list[PatternEvent]] = []
    for event in events:
        placed = False
        for cluster in clusters:
            if _prices_within_tolerance(cluster[0].relevant_prices, event.relevant_prices, ratio=ratio):
                cluster.append(event)
                placed = True
                break
        if not placed:
            clusters.append([event])
    return clusters


def _select_structural_representative(cluster: list[PatternEvent]) -> PatternEvent:
    """Deterministic tie-break, independent of input order: prefer the strongest signal (deepest
    valley / tallest peak, already captured in `signal_strength`), then the earliest-forming
    (widest) pairing, then the lowest pattern_start_index as a final, fully deterministic tie-break.
    """
    ranked = sorted(
        cluster,
        key=lambda event: (
            -event.signal_strength,
            event.pattern_start_at,
            event.pattern_start_index if event.pattern_start_index is not None else min(event.relevant_indices),
        ),
    )
    representative = ranked[0]
    if len(cluster) == 1:
        return representative

    other_starts = sorted(
        {
            event.pattern_start_at.isoformat()
            for event in cluster
            if event is not representative
        }
    )
    merged_reason = (
        f"{representative.detection_reason} This setup was also independently detected from "
        f"{len(cluster) - 1} other qualifying earlier swing pairing(s) starting at "
        f"{', '.join(other_starts)}; these near-duplicate detections were merged into this one "
        "canonical event."
    )
    merged_indices = sorted({index for event in cluster for index in event.relevant_indices})
    return dataclass_replace(
        representative,
        detection_reason=merged_reason,
        relevant_indices=merged_indices,
    )


def deduplicate_structural_events(
    events: list[PatternEvent],
    *,
    price_tolerance_ratio: float = 0.005,
) -> tuple[list[PatternEvent], int]:
    """Merge near-duplicate structural (multi-pivot) pattern events.

    Double Top/Bottom detectors pair every qualifying earlier swing with the nearest valid later
    swing; when more than one earlier swing qualifies against the same later swing and peak, this
    produces multiple near-identical events for what is really one underlying setup (same
    timestamps, status, and price levels, modulo tiny float differences). This runs immediately
    after `resolve_pattern_conflicts`, before events are prepared, tracked, or scored, so downstream
    layers never see the duplicate.

    Deterministic and idempotent: grouping is keyed purely by each event's own canonical identity
    and price levels (never by input position), and the representative choice is a fixed, content-
    based tie-break -- shuffling the input never changes which events survive, and re-running this
    on an already-deduplicated list is a no-op. Distinct setups (different `setup_completion_at`,
    different lifecycle stage, or prices outside tolerance) are never merged.
    """
    grouped: dict[tuple[object, ...], list[PatternEvent]] = {}
    for event in events:
        if event.pattern_id not in _STRUCTURAL_PATTERN_IDS:
            continue
        grouped.setdefault(_structural_canonical_key(event), []).append(event)

    representative_by_id: dict[int, PatternEvent] = {}
    removed_count = 0
    for candidates in grouped.values():
        for cluster in _cluster_events_by_price_tolerance(candidates, ratio=price_tolerance_ratio):
            representative = _select_structural_representative(cluster)
            for event in cluster:
                representative_by_id[id(event)] = representative
            removed_count += len(cluster) - 1

    deduplicated: list[PatternEvent] = []
    seen_representative_ids: set[int] = set()
    for event in events:
        if event.pattern_id not in _STRUCTURAL_PATTERN_IDS:
            deduplicated.append(event)
            continue
        representative = representative_by_id[id(event)]
        representative_key = id(representative)
        if representative_key in seen_representative_ids:
            continue
        seen_representative_ids.add(representative_key)
        deduplicated.append(representative)

    deduplicated.sort(
        key=lambda item: (
            item.detected_at,
            item.pattern_start_at,
            item.pattern_name,
            item.status.value,
        )
    )
    return deduplicated, removed_count
