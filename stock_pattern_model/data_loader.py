"""Backward-compatible wrappers around the provider-based market-data layer."""

from __future__ import annotations

import pandas as pd

from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.market_data import REQUIRED_COLUMNS
from stock_pattern_model.market_data import YFinanceProvider


def load_stock_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "15m",
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    exchange_timezone: str | None = None,
    strict_data: bool = True,
    timeout_seconds: float = 10.0,
    retry_attempts: int = 3,
    cache_dir: str | None = None,
    cache_ttl_seconds: int = 3600,
    bypass_cache: bool = False,
) -> pd.DataFrame:
    """Download intraday OHLCV data using the default Yahoo Finance provider."""
    provider = YFinanceProvider(
        config=MarketDataConfig(
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            strict_data=strict_data,
            exchange_timezone=exchange_timezone,
        )
    )
    payload = provider.load(
        symbol=symbol,
        interval=interval,
        period=period,
        start=start,
        end=end,
        exchange_timezone=exchange_timezone,
        strict_data=strict_data,
        bypass_cache=bypass_cache,
    )
    return payload.dataframe.copy()
