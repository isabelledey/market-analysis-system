"""Explicit exception hierarchy for stock pattern analysis."""

from __future__ import annotations


class StockPatternError(Exception):
    """Base class for package-specific failures."""


class InvalidInstrumentError(StockPatternError):
    """Raised when an instrument identifier is missing or malformed."""


class UnknownSecurityNumberError(InvalidInstrumentError):
    """Raised when a numeric Israeli security number is not found in the mapping."""


class MappingFileError(StockPatternError):
    """Raised when the security-number mapping file is missing or invalid."""


class MissingMappingFileError(MappingFileError):
    """Raised when a required mapping file was not provided or does not exist."""


class InvalidMappingFileError(MappingFileError):
    """Raised when a mapping file exists but its contents are invalid."""


class MarketDataError(StockPatternError):
    """Raised when market data could not be loaded."""


class MarketDataProviderError(MarketDataError):
    """Raised when a specific market-data provider fails."""


class MissingDataFileError(MarketDataProviderError):
    """Raised when a required offline market-data file does not exist."""


class CacheError(MarketDataError):
    """Raised when cached market data cannot be read or written."""


class DataValidationError(StockPatternError):
    """Raised when downloaded or provided data is missing required structure."""


class NoCompletedBarsError(DataValidationError):
    """Raised when all available candles are still incomplete at the analysis cutoff."""


class ConfigurationError(StockPatternError):
    """Raised when a configuration value is invalid."""


class OutputFileError(StockPatternError):
    """Raised when CLI output cannot be written to disk."""
