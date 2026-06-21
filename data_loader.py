"""Utilities for downloading and cleaning intraday stock data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(columns: pd.Index) -> list[str]:
    """Convert yfinance column labels into plain strings."""
    flattened: list[str] = []

    for column in columns:
        if isinstance(column, tuple):
            flattened.append(str(column[0]))
        else:
            flattened.append(str(column))

    return flattened


def load_stock_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "15m",
) -> pd.DataFrame:
    """Download intraday OHLCV data for one symbol and return a clean DataFrame."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol must be a non-empty string.")

    if interval == "15m" and period.endswith("y"):
        raise ValueError(
            "yfinance 15-minute intraday data cannot be requested with year-based periods. "
            "Use a recent period such as '1mo' because intraday history is limited."
        )

    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}' with period='{period}' and interval='{interval}'."
        )

    data = data.copy()
    data.columns = _flatten_columns(data.columns)
    data = data.reset_index()

    if "Datetime" not in data.columns:
        first_column = data.columns[0]
        data = data.rename(columns={first_column: "Datetime"})

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Downloaded data for '{symbol}' is missing required columns: {missing_columns}"
        )

    clean_df = data.loc[:, REQUIRED_COLUMNS].copy()
    clean_df["Datetime"] = pd.to_datetime(clean_df["Datetime"])

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.dropna(subset=numeric_columns)
    clean_df = clean_df.sort_values("Datetime").reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(f"All rows for symbol '{symbol}' were empty after cleaning.")

    return clean_df
