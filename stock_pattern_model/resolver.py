"""Instrument resolution for tickers and Israeli security numbers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

from stock_pattern_model.domain import ResolvedInstrument
from stock_pattern_model.exceptions import InvalidMappingFileError
from stock_pattern_model.exceptions import InvalidInstrumentError
from stock_pattern_model.exceptions import MissingMappingFileError
from stock_pattern_model.exceptions import UnknownSecurityNumberError


def normalize_identifier(value: str) -> str:
    """Normalize a CLI instrument identifier."""
    normalized = value.strip().upper()
    if not normalized:
        raise InvalidInstrumentError("Instrument identifier must not be empty.")
    return normalized


def is_numeric_security_number(value: str) -> bool:
    """Return True when the identifier is a numeric Israeli security number."""
    return value.isdigit()


class InstrumentResolver(Protocol):
    """Resolver interface so other providers can be added later."""

    def resolve(self, identifier: str, mapping_file: str | None = None) -> ResolvedInstrument:
        """Resolve an input identifier into a normalized instrument."""


class CsvInstrumentResolver:
    """Resolve numeric Israeli security numbers from a local CSV mapping file."""

    REQUIRED_HEADERS = (
        "security_number",
        "yahoo_symbol",
        "name",
        "exchange",
        "currency",
        "timezone",
    )

    def resolve(self, identifier: str, mapping_file: str | None = None) -> ResolvedInstrument:
        normalized = normalize_identifier(identifier)

        if is_numeric_security_number(normalized):
            return self._resolve_security_number(normalized, mapping_file)

        if normalized.endswith(".TA"):
            return ResolvedInstrument(
                input_identifier=identifier,
                symbol=normalized,
                security_number=None,
                name=normalized,
                exchange="TASE",
                currency="ILS",
                exchange_timezone="Asia/Jerusalem",
            )

        return ResolvedInstrument(
            input_identifier=identifier,
            symbol=normalized,
            security_number=None,
            name=normalized,
            exchange="Unknown",
            currency="Unknown",
            exchange_timezone="America/New_York",
        )

    def _resolve_security_number(
        self,
        security_number: str,
        mapping_file: str | None,
    ) -> ResolvedInstrument:
        if not mapping_file:
            raise MissingMappingFileError(
                "A mapping file is required for Israeli security numbers.\n"
                "Use: --mapping-file data/tase_securities.csv"
            )

        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            raise MissingMappingFileError(f"Mapping file not found: {mapping_file}")

        try:
            with mapping_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise InvalidMappingFileError("Mapping file is empty.")
                missing_headers = [
                    header for header in self.REQUIRED_HEADERS if header not in reader.fieldnames
                ]
                if missing_headers:
                    raise InvalidMappingFileError(
                        f"Mapping file is missing required columns: {missing_headers}"
                    )

                for row in reader:
                    if row["security_number"].strip() == security_number:
                        symbol = row["yahoo_symbol"].strip()
                        if not symbol:
                            raise InvalidMappingFileError(
                                f"Security number {security_number} has no yahoo_symbol mapping."
                            )
                        return ResolvedInstrument(
                            input_identifier=security_number,
                            symbol=symbol.strip().upper(),
                            security_number=security_number,
                            name=row["name"].strip() or symbol.strip().upper(),
                            exchange=row["exchange"].strip() or "Unknown",
                            currency=row["currency"].strip() or "Unknown",
                            exchange_timezone=row["timezone"].strip() or "Asia/Jerusalem",
                        )
        except OSError as error:
            raise MissingMappingFileError(f"Could not read mapping file: {mapping_file}") from error

        raise UnknownSecurityNumberError(
            f"Unknown Israeli security number: {security_number}"
        )
