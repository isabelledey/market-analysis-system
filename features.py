"""Feature engineering for candlestick and chart-pattern analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features used by the rule-based pattern detector."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

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
    feature_df["Daily_Return"] = feature_df["Close"].pct_change(fill_method=None)
    feature_df["MA_20"] = feature_df["Close"].rolling(window=20, min_periods=20).mean()
    feature_df["MA_50"] = feature_df["Close"].rolling(window=50, min_periods=50).mean()
    feature_df["MA_200"] = feature_df["Close"].rolling(window=200, min_periods=200).mean()
    feature_df["Volume_MA_20"] = (
        feature_df["Volume"].rolling(window=20, min_periods=20).mean()
    )
    feature_df["Volatility_20"] = (
        feature_df["Daily_Return"].rolling(window=20, min_periods=20).std()
    )
    feature_df["Rolling_High_20"] = (
        feature_df["High"].rolling(window=20, min_periods=20).max().shift(1)
    )
    feature_df["Rolling_Low_20"] = (
        feature_df["Low"].rolling(window=20, min_periods=20).min().shift(1)
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df
