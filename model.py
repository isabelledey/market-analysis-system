"""High-level stock analysis model built on rule-based pattern detection."""

from __future__ import annotations

from typing import Any

from data_loader import load_stock_data
from features import add_features
from pattern_detector import classify_trend
from pattern_detector import detect_bearish_engulfing
from pattern_detector import detect_breakdown
from pattern_detector import detect_breakout
from pattern_detector import detect_bullish_engulfing
from pattern_detector import detect_bullish_pin_bar
from pattern_detector import detect_double_bottom
from pattern_detector import detect_double_top
from pattern_detector import detect_inside_bar
from pattern_detector import detect_inside_day_failure
from pattern_detector import detect_shooting_star


RECENCY_WEIGHTS = {
    0: 1.00,
    1: 0.85,
    2: 0.70,
    3: 0.55,
    4: 0.40,
}

PATTERN_SCORES = {
    "Bullish_Engulfing": 18,
    "Bearish_Engulfing": -18,
    "Bullish_Pin_Bar": 14,
    "Shooting_Star": -14,
    "Inside_Day_Failure_Bullish": 10,
    "Inside_Day_Failure_Bearish": -10,
    "Breakout": 20,
    "Breakdown": -20,
    "Double_Bottom": 22,
    "Double_Top": -22,
}

PATTERN_LABELS = {
    "Bullish_Engulfing": "Bullish Engulfing",
    "Bearish_Engulfing": "Bearish Engulfing",
    "Bullish_Pin_Bar": "Bullish Pin Bar",
    "Shooting_Star": "Shooting Star",
    "Inside_Bar": "Inside Bar",
    "Inside_Day_Failure_Bullish": "Inside Day Failure Bullish Reversal",
    "Inside_Day_Failure_Bearish": "Inside Day Failure Bearish Reversal",
    "Breakout": "20-Day Breakout",
    "Breakdown": "20-Day Breakdown",
    "Double_Bottom": "Double Bottom",
    "Double_Top": "Double Top",
}

TREND_SCORES = {
    "Uptrend": 15,
    "Downtrend": -15,
    "Neutral": 0,
}

MAX_NORMALIZED_SCORE = 70.0


def _run_pattern_pipeline(symbol: str):
    """Load the raw data and run all transformations in order."""
    df = load_stock_data(symbol=symbol)
    df = add_features(df)
    df = detect_bullish_engulfing(df)
    df = detect_bearish_engulfing(df)
    df = detect_bullish_pin_bar(df)
    df = detect_shooting_star(df)
    df = detect_inside_bar(df)
    df = detect_inside_day_failure(df)
    df = detect_breakout(df)
    df = detect_breakdown(df)
    df = detect_double_bottom(df)
    df = detect_double_top(df)
    df = classify_trend(df)
    return df


def _collect_recent_patterns(df, lookback_days: int = 5) -> tuple[list[dict[str, Any]], float]:
    """Collect recent pattern hits and convert them into a weighted score."""
    recent_df = df.tail(lookback_days).copy()
    recent_patterns: list[dict[str, Any]] = []
    total_score = 0.0

    for offset, (_, row) in enumerate(recent_df.iloc[::-1].iterrows()):
        recency_weight = RECENCY_WEIGHTS.get(offset, 0.25)
        signal_date = row["Date"].date().isoformat()

        if bool(row.get("Inside_Bar", False)):
            recent_patterns.append(
                {
                    "pattern": PATTERN_LABELS["Inside_Bar"],
                    "date": signal_date,
                    "signal": "Neutral",
                    "weighted_score": 0.0,
                }
            )

        for column_name, base_score in PATTERN_SCORES.items():
            if bool(row.get(column_name, False)):
                weighted_score = round(base_score * recency_weight, 2)
                total_score += weighted_score
                recent_patterns.append(
                    {
                        "pattern": PATTERN_LABELS[column_name],
                        "date": signal_date,
                        "signal": "Bullish" if base_score > 0 else "Bearish",
                        "weighted_score": weighted_score,
                    }
                )

    return recent_patterns, total_score


def _build_explanation(
    symbol: str,
    latest_date: str,
    latest_close: float,
    trend: str,
    detected_patterns: list[dict[str, Any]],
    overall_bias: str,
    confidence_score: float,
    raw_score: float,
) -> str:
    """Create a short human-readable explanation of the current signal."""
    if detected_patterns:
        pattern_names = ", ".join(
            f"{item['pattern']} ({item['date']})" for item in detected_patterns
        )
        pattern_summary = f"Recent pattern activity: {pattern_names}."
    else:
        pattern_summary = "No active chart or candlestick patterns were detected in the last 5 trading days."

    return (
        f"{symbol} closed at {latest_close:.2f} on {latest_date}. "
        f"The current trend classification is {trend}. "
        f"{pattern_summary} "
        f"The weighted rule-based score is {raw_score:.2f}, which maps to a {overall_bias.lower()} bias "
        f"with confidence {confidence_score:.1f}/100."
    )


def analyze_stock(symbol: str) -> dict[str, Any]:
    """Analyze one symbol and return a structured summary."""
    df = _run_pattern_pipeline(symbol)

    latest_row = df.iloc[-1]
    trend = str(latest_row["Trend"])
    detected_patterns, pattern_score = _collect_recent_patterns(df, lookback_days=5)
    total_score = pattern_score + TREND_SCORES.get(trend, 0)

    if total_score > 20:
        overall_bias = "Bullish"
    elif total_score < -20:
        overall_bias = "Bearish"
    else:
        overall_bias = "Neutral"

    confidence_score = min(100.0, round(abs(total_score) / MAX_NORMALIZED_SCORE * 100, 1))
    latest_date = latest_row["Date"].date().isoformat()
    latest_close = float(latest_row["Close"])
    explanation = _build_explanation(
        symbol=symbol,
        latest_date=latest_date,
        latest_close=latest_close,
        trend=trend,
        detected_patterns=detected_patterns,
        overall_bias=overall_bias,
        confidence_score=confidence_score,
        raw_score=total_score,
    )

    return {
        "symbol": symbol.upper(),
        "latest_date": latest_date,
        "latest_close": round(latest_close, 2),
        "trend": trend,
        "detected_patterns": detected_patterns,
        "overall_bias": overall_bias,
        "confidence_score": confidence_score,
        "explanation": explanation,
    }
