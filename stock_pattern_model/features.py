"""Feature engineering for 15-minute candlestick analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.market_data import REQUIRED_COLUMNS
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_END
from stock_pattern_model.session_utils import DEFAULT_REGULAR_SESSION_START
from stock_pattern_model.session_utils import pattern_session_key_series
from stock_pattern_model.session_utils import session_date_series
from stock_pattern_model.session_utils import session_segment_series


def _past_rolling_mean(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def add_features(
    df: pd.DataFrame,
    *,
    exchange_timezone: str | None = None,
    regular_session_start: str = DEFAULT_REGULAR_SESSION_START,
    regular_session_end: str = DEFAULT_REGULAR_SESSION_END,
) -> pd.DataFrame:
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
    feature_df["Avg_Range_20_Bars"] = _past_rolling_mean(
        feature_df["Candle_Range"],
        window=20,
        min_periods=5,
    )
    feature_df["Avg_Range_20_Bars"] = feature_df["Avg_Range_20_Bars"].fillna(
        feature_df["Candle_Range"].shift(1).expanding(min_periods=1).mean()
    )
    feature_df["Rolling_Volume_Baseline_20"] = _past_rolling_mean(
        feature_df["Volume"],
        window=20,
        min_periods=5,
    )
    feature_df["Rolling_Volume_Baseline_20"] = feature_df["Rolling_Volume_Baseline_20"].fillna(
        feature_df["Volume"].shift(1).expanding(min_periods=1).mean()
    )
    feature_df["Volume_MA_20_Bars"] = feature_df["Rolling_Volume_Baseline_20"]
    range_baseline = feature_df["Avg_Range_20_Bars"].where(feature_df["Avg_Range_20_Bars"] > 0)
    volume_baseline_20 = feature_df["Rolling_Volume_Baseline_20"].where(feature_df["Rolling_Volume_Baseline_20"] > 0)
    feature_df["Range_Strength"] = feature_df["Candle_Range"] / range_baseline
    feature_df["Is_Significant_Candle"] = (
        (
            range_baseline.notna()
            & (feature_df["Candle_Range"] >= 0.8 * range_baseline)
        )
        | (
            volume_baseline_20.notna()
            & (feature_df["Volume"] >= volume_baseline_20)
        )
    )
    feature_df["Strong_Range"] = (
        range_baseline.notna()
        & (feature_df["Candle_Range"] > (1.2 * range_baseline) + 1e-9)
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
    feature_df["Session_Date"] = session_date_series(
        feature_df["Datetime"],
        exchange_timezone=exchange_timezone,
    )
    feature_df["Session_Segment"] = session_segment_series(
        feature_df["Datetime"],
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    feature_df["Pattern_Session_Key"] = pattern_session_key_series(
        feature_df["Datetime"],
        exchange_timezone=exchange_timezone,
        regular_session_start=regular_session_start,
        regular_session_end=regular_session_end,
    )
    feature_df["Trading_Date"] = pd.to_datetime(feature_df["Datetime"]).dt.date

    grouped_by_date = feature_df.groupby("Session_Date", sort=False)
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
    volume_baseline = feature_df["Volume_Baseline"].where(feature_df["Volume_Baseline"] > 0)
    feature_df["Volume_Strength"] = feature_df["Volume"] / volume_baseline
    feature_df["Strong_Volume"] = (
        volume_baseline.notna()
        & (feature_df["Volume"] > (1.2 * volume_baseline) + 1e-9)
    )

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df
