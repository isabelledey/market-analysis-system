from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_pattern_model.analysis import analyze_stock
from stock_pattern_model.features import add_features
from stock_pattern_model.market_data import FileDataProvider
from stock_pattern_model.market_data import YFinanceProvider
from stock_pattern_model.market_data import validate_market_data
from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import MarketDataPayload
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.exceptions import MarketDataProviderError


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
EXCHANGE_TZ = ZoneInfo("America/New_York")


def make_df(length: int = 25, start: str = "2026-07-10 09:30", tz=EXCHANGE_TZ) -> pd.DataFrame:
    datetimes = pd.date_range(start=start, periods=length, freq="15min", tz=tz)
    rows = []
    for index, timestamp in enumerate(datetimes):
        rows.append(
            {
                "Datetime": timestamp,
                "Open": 100.0 + index * 0.1,
                "High": 100.6 + index * 0.1,
                "Low": 99.6 + index * 0.1,
                "Close": 100.1 + index * 0.1,
                "Volume": 1000 + index * 10,
            }
        )
    return pd.DataFrame(rows)


def test_csv_loading() -> None:
    provider = FileDataProvider(FIXTURE_DIR / "sample_ohlcv.csv")

    payload = provider.load(
        symbol="AAPL",
        interval="15m",
        exchange_timezone="America/New_York",
        strict_data=True,
    )

    assert len(payload.dataframe) == 25
    assert payload.exchange_timezone == "America/New_York"


def test_parquet_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")
    sample_df = make_df()
    monkeypatch.setattr(pd, "read_parquet", lambda path: sample_df.copy())
    provider = FileDataProvider(parquet_path)

    payload = provider.load(
        symbol="AAPL",
        interval="15m",
        exchange_timezone="America/New_York",
        strict_data=True,
    )

    assert len(payload.dataframe) == len(sample_df)


def test_column_normalization(tmp_path: Path) -> None:
    csv_path = tmp_path / "normalized.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-07-10 09:30:00,100,101,99,100.5,1000\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(csv_path)

    payload = provider.load(
        symbol="AAPL",
        interval="15m",
        exchange_timezone="America/New_York",
        strict_data=True,
    )

    assert list(payload.dataframe.columns) == ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def test_missing_column_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        "Datetime,Open,High,Low,Close\n"
        "2026-07-10 09:30:00,100,101,99,100.5\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(csv_path)

    with pytest.raises(DataValidationError):
        provider.load(
            symbol="AAPL",
            interval="15m",
            exchange_timezone="America/New_York",
            strict_data=True,
        )


def test_invalid_ohlc_rejection(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "Datetime,Open,High,Low,Close,Volume\n"
        "2026-07-10 09:30:00,100,99,98,100.5,1000\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(csv_path)

    with pytest.raises(DataValidationError):
        provider.load(
            symbol="AAPL",
            interval="15m",
            exchange_timezone="America/New_York",
            strict_data=True,
        )


def test_duplicate_timestamp_reporting() -> None:
    duplicate_df = make_df(length=3)
    duplicate_df.loc[2, "Datetime"] = duplicate_df.loc[1, "Datetime"]

    _, report = validate_market_data(
        duplicate_df,
        interval="15m",
        strict_data=False,
    )

    assert report.duplicate_count == 1
    assert "dropped_duplicate_timestamps" in report.cleaning_actions


def test_missing_bar_warning() -> None:
    gap_df = make_df(length=4).drop(index=2).reset_index(drop=True)

    _, report = validate_market_data(
        gap_df,
        interval="15m",
        strict_data=True,
    )

    assert report.irregular_gap_count == 1
    assert any("irregular same-session gaps" in warning for warning in report.warnings)


def test_timezone_naive_input_requires_exchange_timezone(tmp_path: Path) -> None:
    csv_path = tmp_path / "naive.csv"
    csv_path.write_text(
        "Datetime,Open,High,Low,Close,Volume\n"
        "2026-07-10 09:30:00,100,101,99,100.5,1000\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(csv_path)

    with pytest.raises(DataValidationError):
        provider.load(symbol="AAPL", interval="15m", strict_data=True)


def test_daylight_saving_conversion(tmp_path: Path) -> None:
    csv_path = tmp_path / "dst.csv"
    csv_path.write_text(
        "Datetime,Open,High,Low,Close,Volume\n"
        "2026-03-06 09:30:00,100,101,99,100.5,1000\n"
        "2026-03-09 09:30:00,101,102,100,101.5,1100\n",
        encoding="utf-8",
    )
    provider = FileDataProvider(csv_path)

    payload = provider.load(
        symbol="AAPL",
        interval="15m",
        exchange_timezone="America/New_York",
        strict_data=True,
    )

    assert payload.dataframe.loc[0, "Datetime"].isoformat().endswith("-05:00")
    assert payload.dataframe.loc[1, "Datetime"].isoformat().endswith("-04:00")


def test_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cached_df = make_df()
    cached_payload = MarketDataPayload(
        dataframe=cached_df,
        quality_report=DataQualityReport(
            row_count=len(cached_df),
            completed_row_count=len(cached_df),
            duplicate_count=0,
            missing_value_count=0,
            invalid_ohlc_count=0,
            irregular_gap_count=0,
            warnings=["Loaded from cache."],
            cleaning_actions=["cache_hit"],
        ),
        exchange_timezone="America/New_York",
        metadata={"source": "cache"},
    )
    provider = YFinanceProvider(config=MarketDataConfig(cache_dir=str(tmp_path)))
    monkeypatch.setattr(provider, "_is_cache_fresh", lambda path: True)
    monkeypatch.setattr(provider, "_load_cache", lambda path: cached_payload)
    monkeypatch.setattr(provider, "_download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not be called")))

    payload = provider.load(symbol="AAPL", interval="15m", period="1mo")

    assert payload.metadata["source"] == "cache"


def test_expired_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = YFinanceProvider(config=MarketDataConfig(cache_dir=str(tmp_path)))
    sample_df = make_df()
    monkeypatch.setattr(provider, "_is_cache_fresh", lambda path: False)
    monkeypatch.setattr(provider, "_write_cache", lambda path, payload: None)
    monkeypatch.setattr(provider, "_download", lambda *args, **kwargs: (sample_df.copy(), {"source": "yfinance"}))

    payload = provider.load(symbol="AAPL", interval="15m", period="1mo")

    assert payload.metadata["source"] == "yfinance"


def test_retry_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_df = make_df()
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky_download(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise MarketDataProviderError("temporary failure")
        return sample_df.copy(), {"source": "yfinance"}

    provider = YFinanceProvider(
        config=MarketDataConfig(retry_attempts=3, retry_backoff_seconds=0.1, use_cache=False),
        sleep_fn=sleeps.append,
    )
    monkeypatch.setattr(provider, "_download", flaky_download)

    payload = provider.load(symbol="AAPL", interval="15m", period="1mo")

    assert len(payload.dataframe) == len(sample_df)
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_offline_file_based_analysis() -> None:
    fixture_path = FIXTURE_DIR / "sample_ohlcv.csv"

    result = analyze_stock(
        "AAPL",
        interval="15m",
        data_file=str(fixture_path),
        exchange_timezone="America/New_York",
        no_cache=True,
    )

    assert result["symbol"] == "AAPL"
    assert result["data_quality_report"]["row_count"] == 25


def test_continuous_versus_session_reset_features() -> None:
    first_session = make_df(length=20, start="2026-07-10 09:30")
    second_session = make_df(length=5, start="2026-07-11 09:30")
    combined = pd.concat([first_session, second_session], ignore_index=True)

    feature_df = add_features(combined)
    second_session_first_row = feature_df.loc[20]

    assert second_session_first_row["Continuous_MA_20"] != second_session_first_row["Session_MA_20"]


def test_time_of_day_volume_fallback_behavior() -> None:
    session_frames = []
    for session_index in range(4):
        session_frames.append(make_df(length=4, start=f"2026-07-{10 + session_index:02d} 09:30"))
    combined = pd.concat(session_frames, ignore_index=True)

    feature_df = add_features(combined)

    assert feature_df.loc[0, "Volume_Baseline_Source"] == "rolling_20"
    assert feature_df.loc[len(feature_df) - 1, "Volume_Baseline_Source"] == "time_of_day"
