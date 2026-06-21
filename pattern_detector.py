"""Rule-based candlestick and chart pattern detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bullish engulfing candles."""
    pattern_df = df.copy()
    previous_open = pattern_df["Open"].shift(1)
    previous_close = pattern_df["Close"].shift(1)
    previous_bearish = pattern_df["Is_Bearish"].shift(1, fill_value=False)

    pattern_df["Bullish_Engulfing"] = (
        previous_bearish
        & pattern_df["Is_Bullish"]
        & (pattern_df["Open"] <= previous_close)
        & (pattern_df["Close"] >= previous_open)
    )
    return pattern_df


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bearish engulfing candles."""
    pattern_df = df.copy()
    previous_open = pattern_df["Open"].shift(1)
    previous_close = pattern_df["Close"].shift(1)
    previous_bullish = pattern_df["Is_Bullish"].shift(1, fill_value=False)

    pattern_df["Bearish_Engulfing"] = (
        previous_bullish
        & pattern_df["Is_Bearish"]
        & (pattern_df["Open"] >= previous_close)
        & (pattern_df["Close"] <= previous_open)
    )
    return pattern_df


def detect_bullish_pin_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Flag bullish pin bars with optional context filters."""
    pattern_df = df.copy()
    candle_range = pattern_df["Candle_Range"].replace(0, np.nan)
    close_location = (pattern_df["Close"] - pattern_df["Low"]) / candle_range
    short_term_decline = pattern_df["Close"] < pattern_df["Close"].shift(3)
    near_rolling_low = (
        ((pattern_df["Low"] - pattern_df["Rolling_Low_20"]).abs() / pattern_df["Rolling_Low_20"])
        <= 0.02
    )

    base_rule = (
        (pattern_df["Lower_Wick_Ratio"] >= 0.55)
        & (pattern_df["Body_Ratio"] <= 0.35)
        & (close_location >= 0.60)
    )

    pattern_df["Bullish_Pin_Bar"] = base_rule & (
        short_term_decline | near_rolling_low | pattern_df["Rolling_Low_20"].isna()
    )
    return pattern_df


def detect_shooting_star(df: pd.DataFrame) -> pd.DataFrame:
    """Flag shooting star candles with optional context filters."""
    pattern_df = df.copy()
    candle_range = pattern_df["Candle_Range"].replace(0, np.nan)
    close_location = (pattern_df["Close"] - pattern_df["Low"]) / candle_range
    short_term_rise = pattern_df["Close"] > pattern_df["Close"].shift(3)
    near_rolling_high = (
        (
            (pattern_df["High"] - pattern_df["Rolling_High_20"]).abs()
            / pattern_df["Rolling_High_20"]
        )
        <= 0.02
    )

    base_rule = (
        (pattern_df["Upper_Wick_Ratio"] >= 0.55)
        & (pattern_df["Body_Ratio"] <= 0.35)
        & (close_location <= 0.40)
    )

    pattern_df["Shooting_Star"] = base_rule & (
        short_term_rise | near_rolling_high | pattern_df["Rolling_High_20"].isna()
    )
    return pattern_df


def detect_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Flag inside bars."""
    pattern_df = df.copy()
    pattern_df["Inside_Bar"] = (
        (pattern_df["High"] < pattern_df["High"].shift(1))
        & (pattern_df["Low"] > pattern_df["Low"].shift(1))
    )
    return pattern_df


def detect_inside_day_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Flag inside day failures and keep a bullish/bearish direction breakdown."""
    pattern_df = df.copy()
    previous_inside_bar = pattern_df["Inside_Bar"].shift(1, fill_value=False)
    previous_high = pattern_df["High"].shift(1)
    previous_low = pattern_df["Low"].shift(1)

    bearish_failure = (
        previous_inside_bar
        & (pattern_df["High"] > previous_high)
        & (pattern_df["Close"] < previous_high)
        & pattern_df["Is_Bearish"]
    )
    bullish_failure = (
        previous_inside_bar
        & (pattern_df["Low"] < previous_low)
        & (pattern_df["Close"] > previous_low)
        & pattern_df["Is_Bullish"]
    )

    pattern_df["Inside_Day_Failure_Bearish"] = bearish_failure
    pattern_df["Inside_Day_Failure_Bullish"] = bullish_failure
    pattern_df["Inside_Day_Failure"] = bearish_failure | bullish_failure
    return pattern_df


def detect_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 20-day breakouts confirmed by above-average volume."""
    pattern_df = df.copy()
    pattern_df["Breakout"] = (
        (pattern_df["Close"] > pattern_df["Rolling_High_20"])
        & (pattern_df["Volume"] > pattern_df["Volume_MA_20"])
    )
    return pattern_df


def detect_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 20-day breakdowns confirmed by above-average volume."""
    pattern_df = df.copy()
    pattern_df["Breakdown"] = (
        (pattern_df["Close"] < pattern_df["Rolling_Low_20"])
        & (pattern_df["Volume"] > pattern_df["Volume_MA_20"])
    )
    return pattern_df


def detect_double_bottom(
    df: pd.DataFrame,
    tolerance: float = 0.03,
    min_separation: int = 10,
) -> pd.DataFrame:
    """Flag possible double bottoms on the second low."""
    pattern_df = df.copy()
    pattern_df["Double_Bottom"] = False

    local_lows = (
        (pattern_df["Low"] < pattern_df["Low"].shift(1))
        & (pattern_df["Low"] <= pattern_df["Low"].shift(-1))
    )
    low_indices = pattern_df.index[local_lows.fillna(False)].tolist()

    for right_index in range(1, len(low_indices)):
        current_low_index = low_indices[right_index]

        for left_index in range(right_index - 1, -1, -1):
            previous_low_index = low_indices[left_index]
            separation = current_low_index - previous_low_index

            if separation < min_separation:
                continue

            first_low = pattern_df.at[previous_low_index, "Low"]
            second_low = pattern_df.at[current_low_index, "Low"]
            average_low = (first_low + second_low) / 2
            lows_match = abs(first_low - second_low) / average_low <= tolerance

            if not lows_match:
                continue

            bounce_high = pattern_df.loc[previous_low_index:current_low_index, "High"].max()
            bounced = bounce_high >= max(first_low, second_low) * (1 + tolerance)

            if bounced:
                pattern_df.at[current_low_index, "Double_Bottom"] = True
                break

    return pattern_df


def detect_double_top(
    df: pd.DataFrame,
    tolerance: float = 0.03,
    min_separation: int = 10,
) -> pd.DataFrame:
    """Flag possible double tops on the second high."""
    pattern_df = df.copy()
    pattern_df["Double_Top"] = False

    local_highs = (
        (pattern_df["High"] > pattern_df["High"].shift(1))
        & (pattern_df["High"] >= pattern_df["High"].shift(-1))
    )
    high_indices = pattern_df.index[local_highs.fillna(False)].tolist()

    for right_index in range(1, len(high_indices)):
        current_high_index = high_indices[right_index]

        for left_index in range(right_index - 1, -1, -1):
            previous_high_index = high_indices[left_index]
            separation = current_high_index - previous_high_index

            if separation < min_separation:
                continue

            first_high = pattern_df.at[previous_high_index, "High"]
            second_high = pattern_df.at[current_high_index, "High"]
            average_high = (first_high + second_high) / 2
            highs_match = abs(first_high - second_high) / average_high <= tolerance

            if not highs_match:
                continue

            pullback_low = pattern_df.loc[previous_high_index:current_high_index, "Low"].min()
            pulled_back = pullback_low <= min(first_high, second_high) * (1 - tolerance)

            if pulled_back:
                pattern_df.at[current_high_index, "Double_Top"] = True
                break

    return pattern_df


def classify_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Classify the prevailing trend from moving-average structure."""
    pattern_df = df.copy()
    pattern_df["Trend"] = np.select(
        [
            (pattern_df["Close"] > pattern_df["MA_20"])
            & (pattern_df["MA_20"] > pattern_df["MA_50"]),
            (pattern_df["Close"] < pattern_df["MA_20"])
            & (pattern_df["MA_20"] < pattern_df["MA_50"]),
        ],
        ["Uptrend", "Downtrend"],
        default="Neutral",
    )
    return pattern_df
