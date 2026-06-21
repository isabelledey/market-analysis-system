"""High-level 15-minute stock analysis model built on rule-based detection."""

from __future__ import annotations

from typing import Any
from typing import Optional

from data_loader import load_stock_data
from features import add_features
from pattern_detector import PATTERN_DETAILS
from pattern_detector import classify_intraday_trend
from pattern_detector import detect_bearish_engulfing
from pattern_detector import detect_breakdown
from pattern_detector import detect_breakout
from pattern_detector import detect_bullish_engulfing
from pattern_detector import detect_bullish_pin_bar
from pattern_detector import detect_inside_bar
from pattern_detector import detect_inside_bar_failure
from pattern_detector import detect_shooting_star
from pattern_detector import resolve_pattern_conflicts


TREND_SCORES = {
    "Uptrend": 15,
    "Downtrend": -15,
    "Neutral": 0,
}

LOOKBACK_BARS = 12


def _get_recency_weight(candles_ago: int) -> float:
    """Weight newer 15-minute bars more heavily than older ones."""
    if candles_ago == 0:
        return 1.0
    if 1 <= candles_ago <= 3:
        return 0.85
    if 4 <= candles_ago <= 6:
        return 0.65
    return 0.40


def _run_pattern_pipeline(symbol: str):
    """Load the raw intraday data and run all transformations in order."""
    df = load_stock_data(symbol=symbol)
    df = add_features(df)
    df = detect_bullish_engulfing(df)
    df = detect_bearish_engulfing(df)
    df = detect_bullish_pin_bar(df)
    df = detect_shooting_star(df)
    df = detect_inside_bar(df)
    df = detect_inside_bar_failure(df)
    df = detect_breakout(df)
    df = detect_breakdown(df)
    df = classify_intraday_trend(df)
    return df


def _collect_recent_patterns(
    df,
    lookback_bars: int = LOOKBACK_BARS,
) -> list[dict[str, Any]]:
    """Collect recent pattern hits before same-candle conflict resolution."""
    recent_df = df.tail(lookback_bars).copy()
    recent_patterns: list[dict[str, Any]] = []

    for offset, (_, row) in enumerate(recent_df.iloc[::-1].iterrows()):
        recency_weight = _get_recency_weight(offset)
        signal_datetime = row["Datetime"].isoformat()
        range_strength = row.get("Range_Strength")
        volume_strength = row.get("Volume_Strength")
        if range_strength is None or range_strength != range_strength:
            range_strength = 0.0
        if volume_strength is None or volume_strength != volume_strength:
            volume_strength = 0.0
        signal_strength = max(
            float(range_strength),
            float(volume_strength),
        )

        for column_name, metadata in PATTERN_DETAILS.items():
            if bool(row.get(column_name, False)):
                weighted_score = round(metadata["base_score"] * recency_weight, 2)
                recent_patterns.append(
                    {
                        "pattern_column": column_name,
                        "pattern": metadata["label"],
                        "datetime": signal_datetime,
                        "signal": metadata["bias"],
                        "candles_ago": offset,
                        "priority": metadata["priority"],
                        "base_score": metadata["base_score"],
                        "family": metadata["family"],
                        "weighted_score": weighted_score,
                        "signal_strength": round(signal_strength, 2),
                        "volume_confirmed": bool(row.get("Volume_Strength", 0) >= 1.0),
                        "strong_signal": bool(
                            row.get("Strong_Volume", False)
                            or row.get("Strong_Range", False)
                            or column_name.startswith("Strong_")
                        ),
                    }
                )

    return recent_patterns


def _score_patterns(
    resolved_patterns: list[dict[str, Any]],
    trend: str,
) -> dict[str, float]:
    """Separate bullish, bearish, and total scores with a small trend contribution."""
    bullish_score = sum(
        pattern["weighted_score"]
        for pattern in resolved_patterns
        if pattern["signal"] == "Bullish"
    )
    bearish_score = sum(
        abs(pattern["weighted_score"])
        for pattern in resolved_patterns
        if pattern["signal"] == "Bearish"
    )
    neutral_patterns_count = sum(
        1 for pattern in resolved_patterns if pattern["signal"] == "Neutral"
    )

    trend_adjustment = TREND_SCORES.get(trend, 0)
    if trend_adjustment > 0:
        bullish_score += trend_adjustment
    elif trend_adjustment < 0:
        bearish_score += abs(trend_adjustment)

    total_score = bullish_score - bearish_score
    return {
        "bullish_score": round(bullish_score, 2),
        "bearish_score": round(bearish_score, 2),
        "neutral_patterns_count": neutral_patterns_count,
        "total_score": round(total_score, 2),
    }


def classify_market_state(
    df,
    detected_patterns: list[dict[str, Any]],
    score: dict[str, float],
) -> str:
    """Classify the current intraday market state from trend and recent signals."""
    latest_trend = str(df.iloc[-1]["Trend"])
    total_score = score["total_score"]
    bullish_score = score["bullish_score"]
    bearish_score = score["bearish_score"]
    latest_pattern = detected_patterns[0] if detected_patterns else None

    if latest_pattern and latest_pattern["pattern_column"] in {"Strong_Breakout", "Breakout"}:
        return "Breakout Attempt"

    if latest_pattern and latest_pattern["pattern_column"] in {"Strong_Breakdown", "Breakdown"}:
        return "Breakdown Attempt"

    if abs(total_score) <= 8 and bullish_score > 0 and bearish_score > 0:
        return "Choppy"

    if latest_trend == "Uptrend" and bearish_score > bullish_score:
        if any(
            pattern["signal"] == "Bearish" and pattern["strong_signal"]
            for pattern in detected_patterns[:3]
        ):
            return "Reversal Attempt"

    if latest_trend == "Downtrend" and bullish_score > bearish_score:
        if any(
            pattern["signal"] == "Bullish" and pattern["strong_signal"]
            for pattern in detected_patterns[:3]
        ):
            return "Reversal Attempt"

    if latest_trend == "Uptrend" and bullish_score > bearish_score:
        return "Trending Bullish"

    if latest_trend == "Downtrend" and bearish_score > bullish_score:
        return "Trending Bearish"

    return "Neutral"


def _calculate_confidence(
    trend: str,
    market_state: str,
    latest_pattern: Optional[dict[str, Any]],
    top_patterns: list[dict[str, Any]],
    bullish_score: float,
    bearish_score: float,
    total_score: float,
) -> float:
    """Estimate confidence from score separation, trend alignment, and signal quality."""
    score_sum = bullish_score + bearish_score
    score_balance = abs(bullish_score - bearish_score) / score_sum if score_sum else 0.0
    confidence = min(55.0, abs(total_score) * 1.8) + (score_balance * 20.0)

    if market_state == "Choppy":
        confidence -= 25.0

    if trend == "Neutral" and market_state not in {"Breakout Attempt", "Breakdown Attempt"}:
        confidence -= 10.0

    if latest_pattern:
        dominant_bias_count = sum(
            1
            for pattern in top_patterns
            if pattern["signal"] == latest_pattern["signal"]
            and pattern["signal"] != "Neutral"
        )

        if latest_pattern["strong_signal"]:
            confidence += 10.0

        if latest_pattern["volume_confirmed"]:
            confidence += 5.0

        if dominant_bias_count >= 2:
            confidence += 8.0

        if (
            (trend == "Uptrend" and latest_pattern["signal"] == "Bullish")
            or (trend == "Downtrend" and latest_pattern["signal"] == "Bearish")
        ):
            confidence += 10.0

    if abs(bullish_score - bearish_score) <= 6:
        confidence -= 12.0

    if market_state == "Neutral" and not top_patterns:
        confidence = min(confidence, 20.0)

    return round(max(5.0, min(100.0, confidence)), 1)


def _build_explanation(
    symbol: str,
    latest_datetime: str,
    latest_close: float,
    interval: str,
    trend: str,
    market_state: str,
    top_patterns: list[dict[str, Any]],
    overall_bias: str,
    confidence_score: float,
    bullish_score: float,
    bearish_score: float,
    total_score: float,
    ignored_patterns_count: int,
) -> str:
    """Create a short human-readable explanation of the current signal."""
    if top_patterns:
        pattern_names = ", ".join(
            f"{item['pattern']} ({item['datetime']})" for item in top_patterns
        )
        pattern_summary = f"Top recent patterns: {pattern_names}."
    else:
        pattern_summary = "No meaningful intraday patterns survived the recent filtering window."

    return (
        f"{symbol} last traded at {latest_close:.2f} on {latest_datetime} using {interval} candles. "
        f"The current intraday trend is {trend} and the market state is {market_state}. "
        f"{pattern_summary} "
        f"Bullish score is {bullish_score:.2f}, bearish score is {bearish_score:.2f}, "
        f"and total score is {total_score:.2f}. "
        f"{ignored_patterns_count} lower-priority conflicting patterns were ignored. "
        f"This maps to a {overall_bias.lower()} bias with confidence {confidence_score:.1f}/100."
    )


def analyze_stock(symbol: str) -> dict[str, Any]:
    """Analyze one symbol using 15-minute intraday candles."""
    df = _run_pattern_pipeline(symbol)

    latest_row = df.iloc[-1]
    trend = str(latest_row["Trend"])
    interval = "15m"
    raw_patterns = _collect_recent_patterns(df, lookback_bars=LOOKBACK_BARS)
    resolved_patterns, ignored_patterns_count = resolve_pattern_conflicts(raw_patterns)
    score = _score_patterns(resolved_patterns, trend)
    bullish_score = score["bullish_score"]
    bearish_score = score["bearish_score"]
    total_score = score["total_score"]
    market_state = classify_market_state(df, resolved_patterns, score)
    latest_pattern = resolved_patterns[0] if resolved_patterns else None
    top_patterns = sorted(
        resolved_patterns,
        key=lambda item: (-abs(item["weighted_score"]), item["candles_ago"], item["priority"]),
    )[:3]

    if total_score > 20:
        overall_bias = "Bullish"
    elif total_score < -20:
        overall_bias = "Bearish"
    else:
        overall_bias = "Neutral"

    confidence_score = _calculate_confidence(
        trend=trend,
        market_state=market_state,
        latest_pattern=latest_pattern,
        top_patterns=top_patterns,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        total_score=total_score,
    )
    latest_datetime = latest_row["Datetime"].isoformat()
    latest_close = float(latest_row["Close"])
    explanation = _build_explanation(
        symbol=symbol,
        latest_datetime=latest_datetime,
        latest_close=latest_close,
        interval=interval,
        trend=trend,
        market_state=market_state,
        top_patterns=top_patterns,
        overall_bias=overall_bias,
        confidence_score=confidence_score,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        total_score=total_score,
        ignored_patterns_count=ignored_patterns_count,
    )

    return {
        "symbol": symbol.upper(),
        "latest_datetime": latest_datetime,
        "latest_close": round(latest_close, 2),
        "interval": interval,
        "trend": trend,
        "market_state": market_state,
        "overall_bias": overall_bias,
        "confidence_score": confidence_score,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "total_score": total_score,
        "top_patterns": top_patterns,
        "ignored_patterns_count": ignored_patterns_count,
        "explanation": explanation,
    }
