"""Deprecated compatibility wrapper for the packaged pattern detector registry.

Import from ``stock_pattern_model.pattern_detector`` instead.
"""

from __future__ import annotations

from stock_pattern_model.pattern_detector import DEFAULT_PATTERN_REGISTRY
from stock_pattern_model.pattern_detector import PATTERN_DETAILS
from stock_pattern_model.pattern_detector import PatternRegistry
from stock_pattern_model.pattern_detector import classify_intraday_trend
from stock_pattern_model.pattern_detector import detect_patterns
from stock_pattern_model.pattern_detector import resolve_pattern_conflicts


__all__ = [
    "DEFAULT_PATTERN_REGISTRY",
    "PATTERN_DETAILS",
    "PatternRegistry",
    "classify_intraday_trend",
    "detect_patterns",
    "resolve_pattern_conflicts",
]
