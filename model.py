"""Backward-compatible wrappers for the packaged analysis API."""

from __future__ import annotations

from stock_pattern_model.analysis import analyze_dataframe
from stock_pattern_model.analysis import analyze_stock
from stock_pattern_model.exceptions import NoCompletedBarsError as NoCompletedCandlesError


__all__ = [
    "NoCompletedCandlesError",
    "analyze_dataframe",
    "analyze_stock",
]
