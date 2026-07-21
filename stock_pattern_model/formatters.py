"""Formatting helpers for CLI text and JSON output."""

from __future__ import annotations

import json
from typing import Any


def format_analysis_json(result: dict[str, Any]) -> str:
    """Render a result dictionary as pretty JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)


def format_analysis_text(
    result: dict[str, Any],
    include_all_patterns: bool = False,
) -> str:
    """Render an analysis result as human-readable text."""
    instrument = result.get("instrument", {})
    pattern_key = "all_detected_patterns" if include_all_patterns else "top_patterns"
    patterns = result.get(pattern_key, [])
    lines = [
        f"Instrument: {instrument.get('symbol', result.get('symbol', 'UNKNOWN'))}",
        f"Input Identifier: {instrument.get('input_identifier', result.get('symbol', 'UNKNOWN'))}",
        f"Resolved Symbol: {result.get('symbol', instrument.get('symbol', 'UNKNOWN'))}",
        f"Security Number: {instrument.get('security_number') or 'None'}",
        f"Name: {instrument.get('name') or 'Unknown'}",
        f"Exchange: {instrument.get('exchange', 'Unknown')}",
        f"Currency: {instrument.get('currency', 'Unknown')}",
        f"Interval: {result.get('interval', 'Unknown')}",
        f"Analysis Time: {result.get('analysis_time', result.get('as_of', 'Unknown'))}",
        f"Exchange Timezone: {result.get('exchange_timezone', 'Unknown')}",
        f"Display Timezone: {result.get('display_timezone', 'Unknown')}",
        f"Latest Completed Candle Start: {result.get('latest_bar_start', 'Unknown')}",
        f"Latest Completed Candle End: {result.get('latest_bar_end', 'Unknown')}",
        f"Latest Close: {result.get('latest_close', 'Unknown')}",
        f"Data Quality: {result.get('data_quality_report', {}).get('completed_row_count', 'Unknown')} "
        f"completed rows / {result.get('data_quality_report', {}).get('row_count', 'Unknown')} total rows",
        f"Trend: {result.get('trend', 'Unknown')}",
        f"Trend Horizon: {result.get('trend_horizon', 'Unknown')}",
        f"Market State: {result.get('market_state', 'Unknown')}",
        f"Overall Bias: {result.get('overall_bias', 'Unknown')}",
        f"Bullish Score: {result.get('bullish_score', 'Unknown')}",
        f"Bearish Score: {result.get('bearish_score', 'Unknown')}",
        f"Rule Confidence: {result.get('rule_confidence', 'Unknown')}",
        f"Trend Score: {result.get('trend_score', 'Unknown')}",
        f"Trend Signal Contribution: {result.get('trend_signal_score', 'Unknown')}",
        f"Pattern Score: {result.get('pattern_score', 'Unknown')}",
        f"Volume Score: {result.get('volume_score', 'Unknown')}",
        f"Net Signal Score: {result.get('net_signal_score', 'Unknown')}",
        f"Short-Term Trend: {result.get('short_term_trend', 'Unknown')} ({result.get('short_term_trend_score', 'Unknown')})",
        f"Medium-Term Trend: {result.get('medium_term_trend', 'Unknown')} ({result.get('medium_term_trend_score', 'Unknown')})",
        f"Long-Term Trend: {result.get('long_term_trend', 'Unknown')} ({result.get('long_term_trend_score', 'Unknown')})",
        "Patterns:",
    ]

    if patterns:
        for pattern in patterns:
            lines.append(f"  Name: {pattern['pattern_name']}")
            lines.append(f"  Family: {pattern.get('pattern_family', 'unknown')}")
            lines.append(f"  Status: {pattern.get('status', 'confirmed')}")
            lines.append(f"  State: {pattern.get('event_state', 'unknown')}")
            lines.append(f"  Bias: {pattern['bias']}")
            lines.append(f"  Pattern Start: {pattern.get('pattern_start_display', pattern.get('bar_start_display', 'Unknown'))}")
            lines.append(f"  Pattern Completion: {pattern.get('pattern_end_display', pattern.get('bar_end_display', 'Unknown'))}")
            lines.append(f"  Detected at: {pattern['detected_at_display']}")
            lines.append(f"  Display Timezone: {pattern.get('display_timezone', 'Unknown')}")
            lines.append(f"  Signal Strength: {pattern.get('signal_strength', 'Unknown')}")
            lines.append(f"  Weighted Score: {pattern.get('weighted_score', 'Unknown')}")
            lines.append(f"  Reason: {pattern['detection_reason']}")
            lines.append(f"  Evidence group: {pattern.get('evidence_group', 'unknown')}")
    else:
        lines.append("  None")

    warnings = result.get("warnings") or []
    lines.append("Warnings:")
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  None")

    structured = result.get("structured_explanation") or {}
    lines.append("Explanation:")
    lines.append(f"  {structured.get('summary', result.get('explanation', ''))}")
    if structured.get("trend_evidence"):
        lines.append("Trend Evidence:")
        for item in structured["trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bullish_evidence"):
        lines.append("Bullish Evidence:")
        for item in structured["bullish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bearish_evidence"):
        lines.append("Bearish Evidence:")
        for item in structured["bearish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("conflicts"):
        lines.append("Conflicts:")
        for item in structured["conflicts"]:
            lines.append(f"  - {item}")
    if structured.get("data_warnings"):
        lines.append("Data Warnings:")
        for item in structured["data_warnings"]:
            lines.append(f"  - {item}")
    lines.append("Bias Rationale:")
    lines.append(f"  {structured.get('reason_for_bias', '')}")
    lines.append("Confidence Rationale:")
    lines.append(
        f"  {structured.get('reason_for_confidence', result.get('explanation', ''))} "
        "Rule confidence is an uncalibrated rule-strength score, not a probability."
    )
    return "\n".join(lines)
