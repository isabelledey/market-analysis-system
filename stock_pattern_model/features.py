"""Feature engineering for 15-minute candlestick analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_pattern_model.market_data import REQUIRED_COLUMNS
from stock_pattern_model.exceptions import DataValidationError


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday bar features used by the rule-based pattern detector."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Input DataFrame is missing required columns: {missing_columns}")

    feature_df = df.copy()
    safe_range = (feature_df["High"] - feature_df["Low"]).replace(0, np.nan)
    upper_reference = np.maximum(feature_df["Open"], feature_df["Close"])
    lower_reference = np.minimum(feature_df["Open"], feature_df["Close"])

    feature_df["Candle_Body"] = (feature_df["Close"] - feature_df["Open"]).abs()
    feature_df["Candle_Range"] = feature_df["High"] - feature_df["Low"]
    feature_df["Upper_Wick"] = feature_df["High"] - upper_reference
    feature_df["Lower_Wick"] = lower_reference - feature_df["Low"]
    feature_df["Body_Ratio"] = feature_df["Candle_Body"] / safe_range
    feature_df["Upper_Wick_Ratio"] = feature_df["Upper_Wick"] / safe_range
    feature_df["Lower_Wick_Ratio"] = feature_df["Lower_Wick"] / safe_range
    feature_df["Is_Bullish"] = feature_df["Close"] > feature_df["Open"]
    feature_df["Is_Bearish"] = feature_df["Close"] < feature_df["Open"]
    feature_df["Bar_Return"] = feature_df["Close"].pct_change(fill_method=None)
    feature_df["Continuous_MA_20"] = feature_df["Close"].rolling(window=20, min_periods=20).mean()
    feature_df["Continuous_MA_50"] = feature_df["Close"].rolling(window=50, min_periods=50).mean()
    feature_df["MA_20_Bars"] = feature_df["Continuous_MA_20"]
    feature_df["MA_50_Bars"] = feature_df["Continuous_MA_50"]
    feature_df["Avg_Range_20_Bars"] = (
        feature_df["Candle_Range"].rolling(window=20, min_periods=20).mean()
    )
    feature_df["Rolling_Volume_Baseline_20"] = (
        feature_df["Volume"].rolling(window=20, min_periods=20).mean()
    )
    feature_df["Volume_MA_20_Bars"] = feature_df["Rolling_Volume_Baseline_20"]
    feature_df["Range_Strength"] = feature_df["Candle_Range"] / feature_df["Avg_Range_20_Bars"]
    feature_df["Is_Significant_Candle"] = (
        (feature_df["Candle_Range"] >= 0.8 * feature_df["Avg_Range_20_Bars"])
        | (feature_df["Volume"] >= feature_df["Rolling_Volume_Baseline_20"])
    )
    feature_df["Strong_Range"] = (
        feature_df["Candle_Range"] >= 1.2 * feature_df["Avg_Range_20_Bars"]
    )
    feature_df["Volatility_20_Bars"] = (
        feature_df["Bar_Return"].rolling(window=20, min_periods=20).std()
    )
    feature_df["Rolling_High_20_Bars"] = (
        feature_df["High"].rolling(window=20, min_periods=20).max().shift(1)
    )
    feature_df["Rolling_Low_20_Bars"] = (
        feature_df["Low"].rolling(window=20, min_periods=20).min().shift(1)
    )
    feature_df["Trading_Date"] = feature_df["Datetime"].dt.date

    grouped_by_date = feature_df.groupby("Trading_Date", sort=False)
    feature_df["Session_MA_20"] = (
        grouped_by_date["Close"]
        .transform(lambda series: series.rolling(window=20, min_periods=1).mean())
    )
    feature_df["Session_High"] = grouped_by_date["High"].cummax()
    feature_df["Session_Low"] = grouped_by_date["Low"].cummin()
    feature_df["Session_Open"] = grouped_by_date["Open"].transform("first")
    feature_df["Distance_From_Session_High"] = (
        (feature_df["Close"] - feature_df["Session_High"]) / feature_df["Session_High"]
    )
    feature_df["Distance_From_Session_Low"] = (
        (feature_df["Close"] - feature_df["Session_Low"]) / feature_df["Session_Low"]
    )
    feature_df["Time_Of_Day"] = feature_df["Datetime"].dt.strftime("%H:%M")
    feature_df["Time_Of_Day_Volume_Baseline"] = (
        feature_df.groupby("Time_Of_Day", sort=False)["Volume"]
        .transform(lambda series: series.shift(1).rolling(window=5, min_periods=3).mean())
    )
    feature_df["Volume_Baseline_Source"] = np.where(
        feature_df["Time_Of_Day_Volume_Baseline"].notna(),
        "time_of_day",
        "rolling_20",
    )
    feature_df["Volume_Baseline"] = feature_df["Time_Of_Day_Volume_Baseline"].fillna(
        feature_df["Rolling_Volume_Baseline_20"]
    )
    feature_df["Volume_Strength"] = feature_df["Volume"] / feature_df["Volume_Baseline"]
    feature_df["Strong_Volume"] = feature_df["Volume"] >= 1.2 * feature_df["Volume_Baseline"]

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df
