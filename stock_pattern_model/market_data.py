"""Market-data providers, validation, and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import MarketDataPayload
from stock_pattern_model.exceptions import CacheError
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.exceptions import InvalidInstrumentError
from stock_pattern_model.exceptions import MarketDataError
from stock_pattern_model.exceptions import MissingDataFileError
from stock_pattern_model.exceptions import MarketDataProviderError


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
COLUMN_ALIASES = {
    "datetime": "Datetime",
    "date": "Datetime",
    "timestamp": "Datetime",
    "time": "Datetime",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


class MarketDataProvider(Protocol):
    """Provider interface for loading market data from different sources."""

    def load(
        self,
        *,
        symbol: str,
        interval: str,
        period: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        exchange_timezone: str | None = None,
        as_of: pd.Timestamp | None = None,
        strict_data: bool = True,
        bypass_cache: bool = False,
    ) -> MarketDataPayload:
        """Return validated market data plus metadata."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column name variations into canonical names."""
    normalized_df = df.copy()
    rename_map: dict[str, str] = {}
    seen_targets: dict[str, str] = {}

    for column in normalized_df.columns:
        normalized_key = "".join(char for char in str(column).strip().lower() if char.isalnum())
        target = COLUMN_ALIASES.get(normalized_key)
        if target is None:
            continue
        existing = seen_targets.get(target)
        if existing is not None and existing != column:
            raise DataValidationError(
                f"Ambiguous column normalization for '{target}': {existing!r} and {column!r}."
            )
        seen_targets[target] = str(column)
        rename_map[column] = target

    normalized_df = normalized_df.rename(columns=rename_map)
    return normalized_df


def _localize_datetime_series(
    series: pd.Series,
    exchange_timezone: str | None,
) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("Datetime column contains invalid timestamp values.")

    if parsed.dt.tz is None:
        if exchange_timezone is None:
            raise DataValidationError(
                "Timezone-naive timestamps require an explicit exchange timezone."
            )
        timezone = ZoneInfo(exchange_timezone)
        return parsed.dt.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )

    return parsed


def _count_irregular_gaps(datetimes: pd.Series, interval: str) -> int:
    if len(datetimes) < 2:
        return 0

    expected_gap = pd.to_timedelta(interval)
    gap_count = 0
    previous = datetimes.shift(1)
    for current_time, previous_time in zip(datetimes.iloc[1:], previous.iloc[1:]):
        if current_time.date() != previous_time.date():
            continue
        if current_time - previous_time != expected_gap:
            gap_count += 1
    return gap_count


def validate_market_data(
    df: pd.DataFrame,
    *,
    interval: str,
    exchange_timezone: str | None = None,
    as_of: pd.Timestamp | None = None,
    strict_data: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and optionally clean a market-data DataFrame."""
    normalized_df = normalize_columns(df)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized_df.columns]
    if missing_columns:
        raise DataValidationError(f"Input data is missing required columns: {missing_columns}")

    validated_df = normalized_df.loc[:, REQUIRED_COLUMNS].copy()
    validated_df["Datetime"] = _localize_datetime_series(
        validated_df["Datetime"],
        exchange_timezone=exchange_timezone,
    )
    row_count = len(validated_df)
    warnings: list[str] = []
    cleaning_actions: list[str] = []

    duplicate_count = int(validated_df["Datetime"].duplicated().sum())
    is_sorted = bool(validated_df["Datetime"].is_monotonic_increasing)
    if not is_sorted:
        warnings.append("Input timestamps were unsorted.")
        if strict_data:
            raise DataValidationError("Input timestamps are not sorted in ascending order.")
        validated_df = validated_df.sort_values("Datetime").reset_index(drop=True)
        cleaning_actions.append("sorted_timestamps")

    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate timestamps.")
        if strict_data:
            raise DataValidationError(f"Found {duplicate_count} duplicate timestamps.")
        validated_df = validated_df.drop_duplicates(subset=["Datetime"], keep="last").reset_index(drop=True)
        cleaning_actions.append("dropped_duplicate_timestamps")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        validated_df[column] = pd.to_numeric(validated_df[column], errors="coerce")

    missing_value_count = int(validated_df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())
    if missing_value_count:
        warnings.append(f"Found {missing_value_count} missing OHLCV values.")
        if strict_data:
            raise DataValidationError(f"Found {missing_value_count} missing OHLCV values.")
        validated_df = validated_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        cleaning_actions.append("dropped_missing_ohlcv_rows")

    invalid_ohlc_mask = (
        (validated_df["High"] < validated_df["Open"])
        | (validated_df["High"] < validated_df["Close"])
        | (validated_df["High"] < validated_df["Low"])
        | (validated_df["Low"] > validated_df["Open"])
        | (validated_df["Low"] > validated_df["Close"])
        | (validated_df["Volume"] < 0)
        | (validated_df["Open"] <= 0)
        | (validated_df["High"] <= 0)
        | (validated_df["Low"] <= 0)
        | (validated_df["Close"] <= 0)
    )
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_count:
        warnings.append(f"Found {invalid_ohlc_count} rows with invalid OHLCV values.")
        if strict_data:
            raise DataValidationError(f"Found {invalid_ohlc_count} rows with invalid OHLCV values.")
        validated_df = validated_df.loc[~invalid_ohlc_mask].reset_index(drop=True)
        cleaning_actions.append("dropped_invalid_ohlcv_rows")

    validated_df = validated_df.sort_values("Datetime").reset_index(drop=True)
    irregular_gap_count = _count_irregular_gaps(validated_df["Datetime"], interval)
    if irregular_gap_count:
        warnings.append(
            f"Detected {irregular_gap_count} irregular same-session gaps relative to interval {interval}."
        )

    completed_row_count = len(validated_df)
    if as_of is not None:
        normalized_as_of = pd.Timestamp(as_of)
        if normalized_as_of.tzinfo is None:
            raise DataValidationError("as_of must be timezone-aware.")
        completed_row_count = int(
            (validated_df["Datetime"] + pd.to_timedelta(interval) <= normalized_as_of).sum()
        )

    report = DataQualityReport(
        row_count=row_count,
        completed_row_count=completed_row_count,
        duplicate_count=duplicate_count,
        missing_value_count=missing_value_count,
        invalid_ohlc_count=invalid_ohlc_count,
        irregular_gap_count=irregular_gap_count,
        warnings=warnings,
        cleaning_actions=cleaning_actions,
    )
    return validated_df.reset_index(drop=True), report


class FileDataProvider:
    """Load market data from local CSV or Parquet files."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(
        self,
        *,
        symbol: str,
        interval: str,
        period: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        exchange_timezone: str | None = None,
        as_of: pd.Timestamp | None = None,
        strict_data: bool = True,
        bypass_cache: bool = False,
    ) -> MarketDataPayload:
        del period, bypass_cache
        if not self.file_path.exists():
            raise MissingDataFileError(f"Data file not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        try:
            if suffix == ".csv":
                raw_df = pd.read_csv(self.file_path)
            elif suffix == ".parquet":
                raw_df = pd.read_parquet(self.file_path)
            else:
                raise MarketDataProviderError(
                    f"Unsupported file extension '{self.file_path.suffix}'. Use CSV or Parquet."
                )
        except ImportError as error:
            raise MarketDataProviderError(
                f"Parquet support requires an installed parquet engine for {self.file_path}."
            ) from error
        except OSError as error:
            raise MarketDataProviderError(f"Could not read data file: {self.file_path}") from error

        validated_df, report = validate_market_data(
            raw_df,
            interval=interval,
            exchange_timezone=exchange_timezone,
            as_of=as_of,
            strict_data=strict_data,
        )

        if start is not None:
            validated_df = validated_df.loc[validated_df["Datetime"] >= pd.Timestamp(start)].reset_index(drop=True)
        if end is not None:
            validated_df = validated_df.loc[validated_df["Datetime"] < pd.Timestamp(end)].reset_index(drop=True)

        exchange_timezone_name = (
            exchange_timezone or str(pd.to_datetime(validated_df["Datetime"]).dt.tz)
        )
        return MarketDataPayload(
            dataframe=validated_df,
            quality_report=report,
            exchange_timezone=exchange_timezone_name,
            metadata={"source": "file", "file_path": str(self.file_path), "symbol": symbol},
        )


class YFinanceProvider:
    """Load market data from Yahoo Finance with validation, retry, and cache support."""

    def __init__(
        self,
        *,
        config: MarketDataConfig | None = None,
        sleep_fn=time.sleep,
    ) -> None:
        self.config = config or MarketDataConfig()
        self.config.validate()
        self._sleep_fn = sleep_fn

    def load(
        self,
        *,
        symbol: str,
        interval: str,
        period: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        exchange_timezone: str | None = None,
        as_of: pd.Timestamp | None = None,
        strict_data: bool = True,
        bypass_cache: bool = False,
    ) -> MarketDataPayload:
        if not symbol or not isinstance(symbol, str):
            raise InvalidInstrumentError("symbol must be a non-empty string.")
        if period and start is not None and end is not None:
            raise MarketDataProviderError("Use either period or start/end, not all three together.")

        cache_path = self._cache_file(symbol, interval, period, start, end)
        if self.config.use_cache and not bypass_cache and cache_path and self._is_cache_fresh(cache_path):
            cached_payload = self._load_cache(cache_path)
            if cached_payload is not None:
                LOGGER.debug("Loaded market data for %s from cache %s", symbol, cache_path)
                return cached_payload

        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                raw_df, metadata = self._download(symbol, interval, period, start, end)
                validated_df, report = validate_market_data(
                    raw_df,
                    interval=interval,
                    exchange_timezone=exchange_timezone,
                    as_of=as_of,
                    strict_data=strict_data,
                )
                payload = MarketDataPayload(
                    dataframe=validated_df,
                    quality_report=report,
                    exchange_timezone=str(pd.to_datetime(validated_df["Datetime"]).dt.tz),
                    metadata=metadata,
                )
                if self.config.use_cache and not bypass_cache and cache_path:
                    self._write_cache(cache_path, payload)
                return payload
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt >= self.config.retry_attempts:
                    break
                backoff = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                LOGGER.debug(
                    "YFinanceProvider attempt %s/%s failed for %s: %s. Retrying in %.2fs",
                    attempt,
                    self.config.retry_attempts,
                    symbol,
                    error,
                    backoff,
                )
                self._sleep_fn(backoff)

        raise MarketDataProviderError(
            f"Failed to load market data for {symbol} from Yahoo Finance."
        ) from last_error

    def _download(
        self,
        symbol: str,
        interval: str,
        period: str | None,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            ticker = yf.Ticker(symbol)
            history_kwargs: dict[str, Any] = {
                "interval": interval,
                "auto_adjust": False,
                "prepost": False,
                "actions": False,
                "timeout": self.config.timeout_seconds,
            }
            if period is not None:
                history_kwargs["period"] = period
            if start is not None:
                history_kwargs["start"] = start
            if end is not None:
                history_kwargs["end"] = end
            data = ticker.history(**history_kwargs)
            if data.empty:
                raise MarketDataError(
                    f"No data returned for symbol '{symbol}' with interval='{interval}'."
                )
            data = data.reset_index()
            metadata: dict[str, Any] = {"source": "yfinance", "symbol": symbol}
            try:
                fast_info = getattr(ticker, "fast_info", None)
                if fast_info:
                    metadata["fast_info"] = dict(fast_info)
            except Exception:  # noqa: BLE001
                metadata["fast_info"] = None
            return data, metadata
        except Exception as error:  # noqa: BLE001
            raise MarketDataProviderError(
                f"Yahoo Finance request failed for symbol '{symbol}'."
            ) from error

    def _cache_file(
        self,
        symbol: str,
        interval: str,
        period: str | None,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> Path | None:
        cache_root = self.config.cache_path()
        if cache_root is None:
            return None
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = json.dumps(
            {
                "symbol": symbol,
                "interval": interval,
                "period": str(period),
                "start": str(start),
                "end": str(end),
            },
            sort_keys=True,
        ).encode("utf-8")
        filename = hashlib.sha256(cache_key).hexdigest()
        return cache_root / f"{filename}.parquet"

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        age_seconds = time.time() - cache_path.stat().st_mtime
        return age_seconds <= self.config.cache_ttl_seconds

    def _load_cache(self, cache_path: Path) -> MarketDataPayload | None:
        metadata_path = cache_path.with_suffix(".json")
        try:
            dataframe = pd.read_parquet(cache_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            report = DataQualityReport(
                row_count=len(dataframe),
                completed_row_count=len(dataframe),
                duplicate_count=0,
                missing_value_count=0,
                invalid_ohlc_count=0,
                irregular_gap_count=0,
                warnings=["Loaded from cache."],
                cleaning_actions=["cache_hit"],
            )
            exchange_timezone = str(pd.to_datetime(dataframe["Datetime"]).dt.tz)
            return MarketDataPayload(
                dataframe=dataframe,
                quality_report=report,
                exchange_timezone=exchange_timezone,
                metadata=metadata,
            )
        except ImportError as error:
            raise CacheError("Parquet cache requires a parquet engine to read cached data.") from error
        except OSError as error:
            raise CacheError(f"Could not read cache file: {cache_path}") from error

    def _write_cache(self, cache_path: Path, payload: MarketDataPayload) -> None:
        metadata_path = cache_path.with_suffix(".json")
        try:
            payload.dataframe.to_parquet(cache_path, index=False)
            metadata_path.write_text(json.dumps(payload.metadata, indent=2), encoding="utf-8")
        except ImportError as error:
            raise CacheError("Parquet cache requires a parquet engine to write cached data.") from error
        except OSError as error:
            raise CacheError(f"Could not write cache file: {cache_path}") from error
