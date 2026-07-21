"""Public package interface for stock pattern analysis."""

from __future__ import annotations

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.analysis import analyze_stock
from stock_pattern_model.config import MarketDataConfig
from stock_pattern_model.config import HistoricalEvaluationConfig
from stock_pattern_model.config import ScoringConfig
from stock_pattern_model.context import AnalysisContext
from stock_pattern_model.context import InstrumentMetadata
from stock_pattern_model.context import TradingSession
from stock_pattern_model.domain import DataQualityReport
from stock_pattern_model.domain import HistoricalEvaluationResult
from stock_pattern_model.domain import HistoricalPerformanceSummary
from stock_pattern_model.domain import HistoricalSignalOutcome
from stock_pattern_model.domain import HistoricalSignalRecord
from stock_pattern_model.domain import MarketDataPayload
from stock_pattern_model.domain import PatternEvent
from stock_pattern_model.domain import PatternFamily
from stock_pattern_model.domain import PatternStatus
from stock_pattern_model.evaluation import collect_historical_signals_from_dataframe
from stock_pattern_model.evaluation import evaluate_historical_dataframe
from stock_pattern_model.evaluation import evaluate_historical_stock
from stock_pattern_model.exceptions import CacheError
from stock_pattern_model.exceptions import ConfigurationError
from stock_pattern_model.exceptions import DataValidationError
from stock_pattern_model.exceptions import InvalidMappingFileError
from stock_pattern_model.exceptions import InvalidInstrumentError
from stock_pattern_model.exceptions import MarketDataError
from stock_pattern_model.exceptions import MarketDataProviderError
from stock_pattern_model.exceptions import MissingDataFileError
from stock_pattern_model.exceptions import MissingMappingFileError
from stock_pattern_model.exceptions import NoCompletedBarsError
from stock_pattern_model.exceptions import OutputFileError
from stock_pattern_model.exceptions import StockPatternError
from stock_pattern_model.exceptions import UnknownSecurityNumberError
from stock_pattern_model.market_data import FileDataProvider
from stock_pattern_model.market_data import MarketDataProvider
from stock_pattern_model.market_data import YFinanceProvider
from stock_pattern_model.pattern_detector import PatternRegistry
from stock_pattern_model.resolver import CsvInstrumentResolver
from stock_pattern_model.resolver import InstrumentResolver
from stock_pattern_model.scoring import ScoringService


__all__ = [
    "CsvInstrumentResolver",
    "ConfigurationError",
    "CacheError",
    "DataQualityReport",
    "DataValidationError",
    "AnalysisContext",
    "HistoricalEvaluationConfig",
    "HistoricalEvaluationResult",
    "HistoricalPerformanceSummary",
    "HistoricalSignalOutcome",
    "HistoricalSignalRecord",
    "FileDataProvider",
    "InstrumentMetadata",
    "InstrumentResolver",
    "InvalidMappingFileError",
    "InvalidInstrumentError",
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
