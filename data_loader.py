"""Deprecated compatibility wrapper for the packaged market-data layer.

Import from ``stock_pattern_model.data_loader`` instead.
"""

from __future__ import annotations

from stock_pattern_model.data_loader import REQUIRED_COLUMNS
from stock_pattern_model.data_loader import load_stock_data


__all__ = ["REQUIRED_COLUMNS", "load_stock_data"]
