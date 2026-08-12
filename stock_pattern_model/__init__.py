"""Public package interface for stock pattern analysis."""

from __future__ import annotations

from stock_pattern_model.analysis import analyze_dataframe, analyze_stock
from stock_pattern_model.config import (
    HistoricalEvaluationConfig,
    MarketDataConfig,
    ScoringConfig,
)
from stock_pattern_model.context import (
    AnalysisContext,
    InstrumentMetadata,
    TradingSession,
)
from stock_pattern_model.domain import (
    DataQualityReport,
    HistoricalEvaluationResult,
    HistoricalPerformanceSummary,
    HistoricalSignalOutcome,
    HistoricalSignalRecord,
    MarketDataPayload,
    PatternEvent,
    PatternFamily,
    PatternStatus,
)
from stock_pattern_model.evaluation import (
    collect_historical_signals_from_dataframe,
    evaluate_historical_dataframe,
    evaluate_historical_stock,
)
from stock_pattern_model.exceptions import (
    CacheError,
    ConfigurationError,
    DataValidationError,
    InvalidInstrumentError,
    InvalidMappingFileError,
    MarketDataError,
    MarketDataProviderError,
    MissingDataFileError,
    MissingMappingFileError,
    NoCompletedBarsError,
    OutputFileError,
    StockPatternError,
    UnknownSecurityNumberError,
)
from stock_pattern_model.market_data import (
    FileDataProvider,
    MarketDataProvider,
    YFinanceProvider,
)
from stock_pattern_model.pattern_detector import PatternRegistry
from stock_pattern_model.resolver import CsvInstrumentResolver, InstrumentResolver
from stock_pattern_model.scoring import ScoringService

__all__ = [
    "AnalysisContext",
    "CacheError",
    "ConfigurationError",
    "CsvInstrumentResolver",
    "DataQualityReport",
    "DataValidationError",
    "FileDataProvider",
    "HistoricalEvaluationConfig",
    "HistoricalEvaluationResult",
    "HistoricalPerformanceSummary",
    "HistoricalSignalOutcome",
    "HistoricalSignalRecord",
    "InstrumentMetadata",
    "InstrumentResolver",
    "InvalidInstrumentError",
    "InvalidMappingFileError",
    "MarketDataConfig",
    "MarketDataError",
    "MarketDataPayload",
    "MarketDataProvider",
    "MarketDataProviderError",
    "MissingDataFileError",
    "MissingMappingFileError",
    "NoCompletedBarsError",
    "OutputFileError",
    "PatternEvent",
    "PatternFamily",
    "PatternRegistry",
    "PatternStatus",
    "ScoringConfig",
    "ScoringService",
    "StockPatternError",
    "TradingSession",
    "UnknownSecurityNumberError",
    "YFinanceProvider",
    "analyze_dataframe",
    "analyze_stock",
    "collect_historical_signals_from_dataframe",
    "evaluate_historical_dataframe",
    "evaluate_historical_stock",
]
