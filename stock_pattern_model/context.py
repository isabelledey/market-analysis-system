"""Authoritative instrument and analysis context models."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stock_pattern_model.datetime_utils import ensure_timezone_aware
from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.session_utils import normalize_session_mode
from stock_pattern_model.session_utils import session_mode_requires_extended_hours
from stock_pattern_model.session_utils import session_segments_for_mode


EXCHANGE_PROFILE_BY_CODE: dict[str, dict[str, Any]] = {
    "NASDAQ": {
        "exchange": "NASDAQ",
        "mic": "XNAS",
        "exchange_timezone": "America/New_York",
        "exchange_calendar": "NASDAQ",
        "regular_session_start": "09:30",
        "regular_session_end": "16:00",
        "currency": "USD",
    },
    "NYSE": {
        "exchange": "NYSE",
        "mic": "XNYS",
        "exchange_timezone": "America/New_York",
        "exchange_calendar": "NYSE",
        "regular_session_start": "09:30",
        "regular_session_end": "16:00",
        "currency": "USD",
    },
    "AMEX": {
        "exchange": "AMEX",
        "mic": "XASE",
        "exchange_timezone": "America/New_York",
        "exchange_calendar": "AMEX",
        "regular_session_start": "09:30",
        "regular_session_end": "16:00",
        "currency": "USD",
    },
    "TASE": {
        "exchange": "TASE",
        "mic": "XTAE",
        "exchange_timezone": "Asia/Jerusalem",
        "exchange_calendar": "TASE",
        "regular_session_start": "09:45",
        "regular_session_end": "17:25",
        "currency": "ILS",
    },
    "LSE": {
        "exchange": "LSE",
        "mic": "XLON",
        "exchange_timezone": "Europe/London",
        "exchange_calendar": "LSE",
        "regular_session_start": "08:00",
        "regular_session_end": "16:30",
        "currency": "GBP",
    },
    "TSX": {
        "exchange": "TSX",
        "mic": "XTSE",
        "exchange_timezone": "America/Toronto",
        "exchange_calendar": "TSX",
        "regular_session_start": "09:30",
        "regular_session_end": "16:00",
        "currency": "CAD",
    },
    "ASX": {
        "exchange": "ASX",
        "mic": "XASX",
        "exchange_timezone": "Australia/Sydney",
        "exchange_calendar": "ASX",
        "regular_session_start": "10:00",
        "regular_session_end": "16:00",
        "currency": "AUD",
    },
}

SYMBOL_SUFFIX_PROFILE_MAP: dict[str, str] = {
    ".TA": "TASE",
    ".L": "LSE",
    ".TO": "TSX",
    ".V": "TSX",
    ".AX": "ASX",
}

EXCHANGE_ALIASES: dict[str, str] = {
    "NMS": "NASDAQ",
    "NAS": "NASDAQ",
    "NASDAQGS": "NASDAQ",
    "NASDAQGM": "NASDAQ",
    "NASDAQCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "AMEX",
    "TLV": "TASE",
    "TLV STOCK EXCHANGE": "TASE",
}


def _normalize_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return None
    return text


def _upper_or_none(value: Any) -> str | None:
    text = _normalize_blank(value)
    return text.upper() if text is not None else None


def _profile_from_symbol(symbol: str) -> dict[str, Any]:
    for suffix, profile_code in SYMBOL_SUFFIX_PROFILE_MAP.items():
        if symbol.upper().endswith(suffix):
            return dict(EXCHANGE_PROFILE_BY_CODE[profile_code])
    return {}


def _profile_from_exchange(exchange: str | None) -> dict[str, Any]:
    normalized_exchange = _upper_or_none(exchange)
    if normalized_exchange is None:
        return {}
    normalized_exchange = EXCHANGE_ALIASES.get(normalized_exchange, normalized_exchange)
    return dict(EXCHANGE_PROFILE_BY_CODE.get(normalized_exchange, {}))


def _provider_metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, "", "Unknown"):
            return value
    return None


def _flatten_provider_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    flat = dict(metadata)
    for nested_key in ("history_metadata", "fast_info", "info", "instrument_metadata"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                flat.setdefault(key, value)
    return flat


def instrument_metadata_from_sources(
    *,
    symbol: str,
    instrument: ResolvedInstrument | None = None,
    provider_metadata: dict[str, Any] | None = None,
    exchange_timezone_override: str | None = None,
    default_input_identifier: str | None = None,
) -> "InstrumentMetadata":
    """Merge resolver metadata, provider metadata, suffix hints, and explicit overrides."""

    flat_provider_metadata = _flatten_provider_metadata(provider_metadata)
    canonical_symbol = _upper_or_none(
        _provider_metadata_value(
            flat_provider_metadata,
            "symbol",
            "provider_symbol",
        )
    ) or _upper_or_none(symbol) or "UNKNOWN"

    suffix_profile = _profile_from_symbol(canonical_symbol)
    base_exchange = instrument.exchange if instrument is not None else None
    provider_exchange = _provider_metadata_value(
        flat_provider_metadata,
        "exchange",
        "fullExchangeName",
        "exchangeName",
        "market",
    )
    exchange_profile = _profile_from_exchange(provider_exchange) or _profile_from_exchange(base_exchange)
    merged_profile = dict(suffix_profile)
    merged_profile.update(exchange_profile)

    resolved_exchange_timezone = _normalize_blank(exchange_timezone_override)
    metadata_warnings: list[str] = []
    provider_timezone = _provider_metadata_value(
        flat_provider_metadata,
        "exchangeTimezoneName",
        "timezone",
    )
    instrument_timezone = instrument.exchange_timezone if instrument is not None else None
    inferred_timezone = (
        resolved_exchange_timezone
        or _normalize_blank(provider_timezone)
        or _normalize_blank(instrument_timezone)
        or _normalize_blank(merged_profile.get("exchange_timezone"))
    )

    if (
        resolved_exchange_timezone is not None
        and provider_timezone is not None
        and str(resolved_exchange_timezone) != str(provider_timezone)
    ):
        metadata_warnings.append(
            "Explicit exchange-timezone override differs from provider metadata and is being used for this run."
        )
    if inferred_timezone is None:
        metadata_warnings.append(
            "Exchange timezone could not be resolved; session analysis may be unreliable."
        )
    if _normalize_blank(merged_profile.get("exchange_calendar")) is None:
        metadata_warnings.append(
            "Exchange calendar could not be resolved; holiday and early-close handling may be incomplete."
        )

    return InstrumentMetadata(
        input_identifier=(
            instrument.input_identifier
            if instrument is not None
            else default_input_identifier or canonical_symbol
        ),
        canonical_symbol=canonical_symbol,
        security_number=instrument.security_number if instrument is not None else None,
        name=(
            _normalize_blank(
                _provider_metadata_value(
                    flat_provider_metadata,
                    "longName",
                    "shortName",
                    "instrument_name",
                    "name",
                )
            )
            or (instrument.name if instrument is not None else None)
            or canonical_symbol
        ),
        exchange=(
            _normalize_blank(provider_exchange)
            or _normalize_blank(base_exchange)
            or _normalize_blank(merged_profile.get("exchange"))
        ),
        mic=_normalize_blank(_provider_metadata_value(flat_provider_metadata, "mic")) or _normalize_blank(merged_profile.get("mic")),
        currency=(
            _normalize_blank(
                _provider_metadata_value(
                    flat_provider_metadata,
                    "currency",
                    "financialCurrency",
                )
            )
            or (instrument.currency if instrument is not None else None)
            or _normalize_blank(merged_profile.get("currency"))
        ),
        quote_type=_normalize_blank(
            _provider_metadata_value(
                flat_provider_metadata,
                "quoteType",
                "instrumentType",
            )
        ),
        exchange_timezone=inferred_timezone,
        exchange_calendar=_normalize_blank(merged_profile.get("exchange_calendar")),
        regular_session_start=_normalize_blank(merged_profile.get("regular_session_start")),
        regular_session_end=_normalize_blank(merged_profile.get("regular_session_end")),
        warnings=tuple(metadata_warnings),
        sources=tuple(
            source
            for source, present in (
                ("resolver", instrument is not None),
                ("provider", bool(flat_provider_metadata)),
                ("suffix_mapping", bool(suffix_profile)),
                ("exchange_mapping", bool(exchange_profile)),
                ("explicit_override", resolved_exchange_timezone is not None),
            )
            if present
        ),
        provider_metadata=flat_provider_metadata,
    )


def dataframe_identity(df: pd.DataFrame) -> str:
    """Build a stable, low-cost identity token for one normalized DataFrame."""
    if df.empty:
        return "empty"
    datetimes = pd.to_datetime(df["Datetime"])
    digest = sha256()
    digest.update(str(len(df)).encode("utf-8"))
    digest.update(str(datetimes.iloc[0].value).encode("utf-8"))
    digest.update(str(datetimes.iloc[-1].value).encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class InstrumentMetadata:
    """Resolved instrument metadata shared across the analysis pipeline."""

    input_identifier: str
    canonical_symbol: str
    security_number: str | None = None
    name: str | None = None
    exchange: str | None = None
    mic: str | None = None
    currency: str | None = None
    quote_type: str | None = None
    exchange_timezone: str | None = None
    exchange_calendar: str | None = None
    regular_session_start: str | None = None
    regular_session_end: str | None = None
    warnings: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    provider_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradingSession:
    """Canonical exchange and display session range for one trading session."""

    trading_date: str
    session_id: str
    segment: str
    exchange_timezone: str
    exchange_start: pd.Timestamp
    exchange_end: pd.Timestamp
    display_timezone: str
    display_start: pd.Timestamp
    display_end: pd.Timestamp
    session_mode: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("exchange_start", "exchange_end", "display_start", "display_end"):
            payload[key] = ensure_timezone_aware(payload[key]).isoformat(timespec="minutes")
        return payload


@dataclass(frozen=True)
class AnalysisContext:
    """Authoritative analysis context used by providers, analysis, and reporting."""

    instrument: InstrumentMetadata
    provider: str
    interval: str
    requested_period: str | None
    requested_start: str | pd.Timestamp | None
    requested_end: str | pd.Timestamp | None
    session_mode: str
    included_segments: tuple[str, ...]
    exchange_timezone: str | None
    display_timezone: str
    exchange_calendar: str | None
    regular_session_start: str
    regular_session_end: str
    include_extended_hours: bool
    adjusted: bool
    analysis_time: pd.Timestamp | None = None
    latest_completed_candle_start: pd.Timestamp | None = None
    latest_completed_candle_end: pd.Timestamp | None = None
    dataframe_identity: str | None = None
    cache_config: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    def with_runtime_state(
        self,
        *,
        analysis_time: pd.Timestamp | None = None,
        latest_completed_candle_start: pd.Timestamp | None = None,
        latest_completed_candle_end: pd.Timestamp | None = None,
        dataframe_identity_value: str | None = None,
        warnings: list[str] | tuple[str, ...] | None = None,
    ) -> "AnalysisContext":
        return AnalysisContext(
            instrument=self.instrument,
            provider=self.provider,
            interval=self.interval,
            requested_period=self.requested_period,
            requested_start=self.requested_start,
            requested_end=self.requested_end,
            session_mode=self.session_mode,
            included_segments=self.included_segments,
            exchange_timezone=self.exchange_timezone,
            display_timezone=self.display_timezone,
            exchange_calendar=self.exchange_calendar,
            regular_session_start=self.regular_session_start,
            regular_session_end=self.regular_session_end,
            include_extended_hours=self.include_extended_hours,
            adjusted=self.adjusted,
            analysis_time=analysis_time if analysis_time is not None else self.analysis_time,
            latest_completed_candle_start=(
                latest_completed_candle_start
                if latest_completed_candle_start is not None
                else self.latest_completed_candle_start
            ),
            latest_completed_candle_end=(
                latest_completed_candle_end
                if latest_completed_candle_end is not None
                else self.latest_completed_candle_end
            ),
            dataframe_identity=(
                dataframe_identity_value
                if dataframe_identity_value is not None
                else self.dataframe_identity
            ),
            cache_config=dict(self.cache_config or {}),
            warnings=tuple(warnings if warnings is not None else self.warnings),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("analysis_time", "latest_completed_candle_start", "latest_completed_candle_end"):
            value = payload.get(key)
            if value is not None:
                payload[key] = ensure_timezone_aware(value).isoformat(timespec="minutes")
        payload["instrument"] = self.instrument.to_dict()
        return payload


def build_analysis_context(
    *,
    symbol: str,
    interval: str,
    display_timezone: str,
    session_mode: str,
    instrument: ResolvedInstrument | None = None,
    provider: str = "unknown",
    provider_metadata: dict[str, Any] | None = None,
    requested_period: str | None = None,
    requested_start: str | pd.Timestamp | None = None,
    requested_end: str | pd.Timestamp | None = None,
    exchange_timezone_override: str | None = None,
    regular_session_start: str | None = None,
    regular_session_end: str | None = None,
    cache_config: dict[str, Any] | None = None,
    adjusted: bool = False,
) -> AnalysisContext:
    """Build the single shared context for one analysis or evaluation run."""

    ZoneInfo(display_timezone)
    normalized_session_mode = normalize_session_mode(session_mode)
    instrument_metadata = instrument_metadata_from_sources(
        symbol=symbol,
        instrument=instrument,
        provider_metadata=provider_metadata,
        exchange_timezone_override=exchange_timezone_override,
        default_input_identifier=symbol,
    )

    effective_regular_start = (
        _normalize_blank(regular_session_start)
        or instrument_metadata.regular_session_start
        or "09:30"
    )
    effective_regular_end = (
        _normalize_blank(regular_session_end)
        or instrument_metadata.regular_session_end
        or "16:00"
    )
    warnings = list(instrument_metadata.warnings)

    return AnalysisContext(
        instrument=instrument_metadata,
        provider=provider,
        interval=interval,
        requested_period=requested_period,
        requested_start=requested_start,
        requested_end=requested_end,
        session_mode=normalized_session_mode,
        included_segments=session_segments_for_mode(normalized_session_mode),
        exchange_timezone=instrument_metadata.exchange_timezone,
        display_timezone=display_timezone,
        exchange_calendar=instrument_metadata.exchange_calendar,
        regular_session_start=effective_regular_start,
        regular_session_end=effective_regular_end,
        include_extended_hours=session_mode_requires_extended_hours(normalized_session_mode),
        adjusted=adjusted,
        cache_config=dict(cache_config or {}),
        warnings=tuple(warnings),
    )
