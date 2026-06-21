"""Rule-based intraday candlestick and breakout pattern detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PATTERN_DETAILS = {
    "Strong_Breakout": {
        "label": "Strong 20-Bar Breakout",
        "bias": "Bullish",
        "priority": 1,
        "base_score": 26,
        "family": "breakout",
    },
    "Strong_Breakdown": {
        "label": "Strong 20-Bar Breakdown",
        "bias": "Bearish",
        "priority": 1,
        "base_score": 26,
        "family": "breakdown",
    },
    "Breakout": {
        "label": "20-Bar Breakout",
        "bias": "Bullish",
        "priority": 2,
        "base_score": 18,
        "family": "breakout",
    },
    "Breakdown": {
        "label": "20-Bar Breakdown",
        "bias": "Bearish",
        "priority": 2,
        "base_score": 18,
        "family": "breakdown",
    },
    "Bullish_Engulfing": {
        "label": "Bullish Engulfing",
        "bias": "Bullish",
        "priority": 3,
        "base_score": 15,
        "family": "engulfing",
    },
    "Bearish_Engulfing": {
        "label": "Bearish Engulfing",
        "bias": "Bearish",
        "priority": 3,
        "base_score": 15,
        "family": "engulfing",
    },
    "Inside_Bar_Failure_Bullish": {
        "label": "Inside Bar Failure Bullish Reversal",
        "bias": "Bullish",
        "priority": 4,
        "base_score": 11,
        "family": "inside_bar_failure",
    },
    "Inside_Bar_Failure_Bearish": {
        "label": "Inside Bar Failure Bearish Reversal",
        "bias": "Bearish",
        "priority": 4,
        "base_score": 11,
        "family": "inside_bar_failure",
    },
    "Bullish_Pin_Bar": {
        "label": "Bullish Pin Bar",
        "bias": "Bullish",
        "priority": 5,
        "base_score": 10,
        "family": "pin_bar",
    },
    "Shooting_Star": {
        "label": "Shooting Star",
        "bias": "Bearish",
        "priority": 5,
        "base_score": 10,
        "family": "pin_bar",
    },
    "Inside_Bar": {
        "label": "Inside Bar",
        "bias": "Neutral",
        "priority": 6,
        "base_score": 0,
        "family": "inside_bar",
    },
}


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bullish engulfing candles."""
    pattern_df = df.copy()
    previous_open = pattern_df["Open"].shift(1)
    previous_close = pattern_df["Close"].shift(1)
    previous_bearish = pattern_df["Is_Bearish"].shift(1, fill_value=False)
    previous_significant = pattern_df["Is_Significant_Candle"].shift(1, fill_value=False)

    pattern_df["Bullish_Engulfing"] = (
        previous_bearish
        & pattern_df["Is_Bullish"]
        & (pattern_df["Open"] <= previous_close)
        & (pattern_df["Close"] >= previous_open)
        & (pattern_df["Is_Significant_Candle"] | previous_significant)
    )
    return pattern_df


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bearish engulfing candles."""
    pattern_df = df.copy()
    previous_open = pattern_df["Open"].shift(1)
    previous_close = pattern_df["Close"].shift(1)
    previous_bullish = pattern_df["Is_Bullish"].shift(1, fill_value=False)
    previous_significant = pattern_df["Is_Significant_Candle"].shift(1, fill_value=False)

    pattern_df["Bearish_Engulfing"] = (
        previous_bullish
        & pattern_df["Is_Bearish"]
        & (pattern_df["Open"] >= previous_close)
        & (pattern_df["Close"] <= previous_open)
        & (pattern_df["Is_Significant_Candle"] | previous_significant)
    )
    return pattern_df


def detect_bullish_pin_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bullish 15-minute pin bars."""
    pattern_df = df.copy()
    candle_range = pattern_df["Candle_Range"].replace(0, np.nan)
    close_location = (pattern_df["Close"] - pattern_df["Low"]) / candle_range
    pattern_df["Bullish_Pin_Bar"] = (
        (pattern_df["Lower_Wick_Ratio"] >= 0.55)
        & (pattern_df["Body_Ratio"] <= 0.35)
        & (close_location >= 0.60)
        & pattern_df["Is_Significant_Candle"]
    )
    return pattern_df


def detect_shooting_star(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 15-minute shooting star candles."""
    pattern_df = df.copy()
    candle_range = pattern_df["Candle_Range"].replace(0, np.nan)
    close_location = (pattern_df["Close"] - pattern_df["Low"]) / candle_range
    pattern_df["Shooting_Star"] = (
        (pattern_df["Upper_Wick_Ratio"] >= 0.55)
        & (pattern_df["Body_Ratio"] <= 0.35)
        & (close_location <= 0.40)
        & pattern_df["Is_Significant_Candle"]
    )
    return pattern_df


def detect_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Flag inside bars after a meaningful preceding bar."""
    pattern_df = df.copy()
    previous_significant = pattern_df["Is_Significant_Candle"].shift(1, fill_value=False)
    pattern_df["Inside_Bar"] = (
        (pattern_df["High"] < pattern_df["High"].shift(1))
        & (pattern_df["Low"] > pattern_df["Low"].shift(1))
        & (pattern_df["Is_Significant_Candle"] | previous_significant)
    )
    return pattern_df


def detect_inside_bar_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Flag inside bar failures and keep a bullish/bearish direction breakdown."""
    pattern_df = df.copy()
    previous_inside_bar = pattern_df["Inside_Bar"].shift(1, fill_value=False)
    previous_high = pattern_df["High"].shift(1)
    previous_low = pattern_df["Low"].shift(1)
    broke_above = pattern_df["High"] > previous_high
    broke_below = pattern_df["Low"] < previous_low
    closed_back_inside = (pattern_df["Close"] <= previous_high) & (
        pattern_df["Close"] >= previous_low
    )
    dual_side_sweep = previous_inside_bar & broke_above & broke_below

    bearish_failure = (
        previous_inside_bar
        & broke_above
        & ~broke_below
        & (closed_back_inside | pattern_df["Is_Bearish"])
        & pattern_df["Is_Significant_Candle"]
    )
    bullish_failure = (
        previous_inside_bar
        & broke_below
        & ~broke_above
        & (closed_back_inside | pattern_df["Is_Bullish"])
        & pattern_df["Is_Significant_Candle"]
    )

    bearish_failure = bearish_failure | (
        dual_side_sweep & pattern_df["Is_Bearish"] & pattern_df["Is_Significant_Candle"]
    )
    bullish_failure = bullish_failure | (
        dual_side_sweep & pattern_df["Is_Bullish"] & pattern_df["Is_Significant_Candle"]
    )

    pattern_df["Inside_Bar_Failure_Bearish"] = bearish_failure
    pattern_df["Inside_Bar_Failure_Bullish"] = bullish_failure
    pattern_df["Inside_Bar_Failure"] = bearish_failure | bullish_failure
    return pattern_df


def detect_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 20-bar breakouts and stronger volume-confirmed breakouts."""
    pattern_df = df.copy()
    pattern_df["Breakout"] = (
        (pattern_df["Close"] > pattern_df["Rolling_High_20_Bars"])
        & (pattern_df["Volume"] >= pattern_df["Volume_MA_20_Bars"])
    )
    pattern_df["Strong_Breakout"] = pattern_df["Breakout"] & pattern_df["Strong_Volume"]
    return pattern_df


def detect_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 20-bar breakdowns and stronger volume-confirmed breakdowns."""
    pattern_df = df.copy()
    pattern_df["Breakdown"] = (
        (pattern_df["Close"] < pattern_df["Rolling_Low_20_Bars"])
        & (pattern_df["Volume"] >= pattern_df["Volume_MA_20_Bars"])
    )
    pattern_df["Strong_Breakdown"] = pattern_df["Breakdown"] & pattern_df["Strong_Volume"]
    return pattern_df


def classify_intraday_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Classify the intraday trend from moving-average structure."""
    pattern_df = df.copy()
    pattern_df["Trend"] = np.select(
        [
            (pattern_df["Close"] > pattern_df["MA_20_Bars"])
            & (pattern_df["MA_20_Bars"] > pattern_df["MA_50_Bars"]),
            (pattern_df["Close"] < pattern_df["MA_20_Bars"])
            & (pattern_df["MA_20_Bars"] < pattern_df["MA_50_Bars"]),
        ],
        ["Uptrend", "Downtrend"],
        default="Neutral",
    )
    return pattern_df


def resolve_pattern_conflicts(
    raw_patterns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the highest-priority pattern per candle unless extra patterns share its bias."""
    grouped_patterns: dict[str, list[dict[str, Any]]] = {}

    for pattern in raw_patterns:
        grouped_patterns.setdefault(pattern["datetime"], []).append(pattern)

    resolved_patterns: list[dict[str, Any]] = []
    ignored_patterns_count = 0

    for pattern_group in grouped_patterns.values():
        sorted_group = sorted(
            pattern_group,
            key=lambda item: (
                item["priority"],
                -abs(item["weighted_score"]),
                item["pattern"],
            ),
        )
        best_pattern = sorted_group[0]
        resolved_patterns.append(best_pattern)

        for pattern in sorted_group[1:]:
            same_direction = (
                best_pattern["signal"] != "Neutral"
                and pattern["signal"] == best_pattern["signal"]
            )
            same_family = pattern["family"] == best_pattern["family"]

            if same_direction and not same_family:
                resolved_patterns.append(pattern)
            else:
                ignored_patterns_count += 1

    resolved_patterns = sorted(
        resolved_patterns,
        key=lambda item: (item["candles_ago"], item["priority"], -abs(item["weighted_score"])),
    )
    return resolved_patterns, ignored_patterns_count
