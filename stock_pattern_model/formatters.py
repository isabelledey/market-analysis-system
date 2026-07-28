"""Formatting helpers for CLI text and JSON output."""

from __future__ import annotations

import json
from typing import Any


def _append_transition_lines(lines: list[str], pattern: dict[str, Any], *, indent: str) -> None:
    for label, key in (
        ("State Updated", "state_updated_at_display"),
        ("Retest Time", "retest_at_display"),
        ("Rejection Time", "rejection_at_display"),
        ("Reclaimed Time", "reclaimed_at_display"),
        ("Failed Time", "failed_at_display"),
        ("Invalidated Time", "invalidated_at_display"),
        ("Expired Time", "expired_at_display"),
    ):
        value = pattern.get(key)
        if value:
            lines.append(f"{indent}{label}: {value}")


def format_analysis_json(result: dict[str, Any]) -> str:
    """Render a result dictionary as pretty JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)


def format_analysis_text(
    result: dict[str, Any],
    include_all_patterns: bool = False,
    pattern_history_mode: str = "session",
    history_limit: int | None = None,
) -> str:
    """Render an analysis result as human-readable text."""
    instrument = result.get("instrument", {})
    current_patterns = result.get("current_relevant_patterns") or []
    session_history = result.get("session_pattern_history") or []
    current_contributing = result.get("current_contributing_evidence") or []
    awaiting_confirmation = result.get("awaiting_confirmation_evidence") or []
    current_conflicting = result.get("current_conflicting_evidence") or []
    current_neutral = result.get("current_neutral_evidence") or []
    historical_lifecycle = result.get("historical_lifecycle_events") or []
    recent_non_contributing = result.get("recent_non_contributing_tracked_events") or []
    historical_summary = result.get("historical_lifecycle_summary") or {}
    raw_pattern_key = "all_detected_patterns" if include_all_patterns else "top_patterns"
    raw_patterns = result.get(raw_pattern_key, [])
    if history_limit is not None:
        session_history = session_history[:history_limit]
    total_history = result.get("session_history_total", len(session_history))
    shown_history = len(session_history)
    history_count_suffix = "" if total_history == shown_history else f" of {total_history}"
    lines = [
        f"Instrument: {instrument.get('symbol', result.get('symbol', 'UNKNOWN'))}",
        f"Input Identifier: {instrument.get('input_identifier', result.get('symbol', 'UNKNOWN'))}",
        f"Resolved Symbol: {result.get('symbol', instrument.get('symbol', 'UNKNOWN'))}",
        f"Security Number: {instrument.get('security_number') or 'None'}",
        f"Name: {instrument.get('name') or 'Unknown'}",
        f"Exchange: {instrument.get('exchange') or 'Unknown'}",
        f"Exchange Calendar: {result.get('exchange_calendar') or 'Unknown'}",
        f"Currency: {instrument.get('currency') or 'Unknown'}",
        f"Interval: {result.get('interval', 'Unknown')}",
        f"Analysis Time: {result.get('analysis_time', result.get('as_of', 'Unknown'))}",
        f"Exchange Timezone: {result.get('exchange_timezone') or 'Unknown'}",
        f"Display Timezone: {result.get('display_timezone') or 'Unknown'}",
        f"Session Mode: {result.get('session_mode') or 'Unknown'}",
        f"Included Segments: {', '.join(result.get('included_segments', [])) or 'Unknown'}",
        f"Excluded Segments: {', '.join(result.get('excluded_segments', [])) or 'None'}",
        f"Latest Completed Candle Start: {result.get('latest_bar_start', 'Unknown')}",
        f"Latest Completed Candle End: {result.get('latest_bar_end', 'Unknown')}",
        f"Latest Close: {result.get('latest_close', 'Unknown')}",
        f"Data Quality: {result.get('data_quality_report', {}).get('completed_row_count', 'Unknown')} "
        f"completed rows / {result.get('data_quality_report', {}).get('row_count', 'Unknown')} total rows",
        f"Trend: {result.get('trend', 'Unknown')}",
        f"Trend Horizon: {result.get('trend_horizon', 'Unknown')}",
        f"Market State: {result.get('market_state', 'Unknown')}",
        f"Overall Bias: {result.get('overall_bias', 'Unknown')}",
        f"Bullish Pattern Score: {result.get('bullish_pattern_score', result.get('bullish_score', 'Unknown'))}",
        f"Bearish Pattern Score: {result.get('bearish_pattern_score', result.get('bearish_score', 'Unknown'))}",
        f"Rule Confidence: {result.get('rule_confidence', 'Unknown')}",
        f"Trend Score: {result.get('trend_score', 'Unknown')}",
        f"Trend Signal Contribution: {result.get('trend_signal_score', 'Unknown')}",
        f"Pattern Score: {result.get('pattern_score', 'Unknown')}",
        f"Volume Score: {result.get('volume_score', 'Unknown')}",
        f"Net Signal Score: {result.get('net_signal_score', 'Unknown')}",
        f"Short-Term Trend: {result.get('short_term_trend', 'Unknown')} ({result.get('short_term_trend_score', 'Unknown')})",
        f"Medium-Term Trend: {result.get('medium_term_trend', 'Unknown')} ({result.get('medium_term_trend_score', 'Unknown')})",
        f"Long-Term Trend: {result.get('long_term_trend', 'Unknown')} ({result.get('long_term_trend_score', 'Unknown')})",
        f"Current Contributing Evidence Count: {len(current_contributing)}",
        f"Awaiting Confirmation Count: {len(awaiting_confirmation)}",
        f"Current Conflicting Evidence Count: {len(current_conflicting)}",
        f"Current Neutral Evidence Count: {len(current_neutral)}",
        f"Recent Non-Contributing Tracked Event Count: {len(recent_non_contributing)}",
        f"Historical Lifecycle Event Count: {len(historical_lifecycle)}",
    ]

    def append_event_section(title: str, patterns: list[dict[str, Any]]) -> None:
        lines.append(f"{title} ({len(patterns)}):")
        if not patterns:
            lines.append("  None")
            return
        for pattern in patterns:
            labels = ", ".join(pattern.get("pattern_labels", [pattern.get("primary_pattern_name", "Unknown")]))
            lines.append(f"  Name: {pattern.get('primary_pattern_name', 'Unknown')}")
            lines.append(f"  Pattern Labels: {labels}")
            if pattern.get("raw_geometry_labels"):
                lines.append(f"  Raw Geometry Labels: {', '.join(pattern['raw_geometry_labels'])}")
            lines.append(f"  Family: {pattern.get('family', 'unknown')}")
            lines.append(f"  State: {pattern.get('state', 'unknown')}")
            lines.append(f"  Status: {pattern.get('status', 'unknown')}")
            lines.append(f"  Bias: {pattern.get('bias', 'Unknown')}")
            lines.append(f"  Geometry: {pattern.get('geometry_label', 'unknown')}")
            lines.append(f"  Context Quality: {pattern.get('context_quality', 'unknown')}")
            lines.append(f"  Pattern Start: {pattern.get('pattern_start_display', 'Unknown')}")
            lines.append(f"  Setup Completion: {pattern.get('setup_completion_display', pattern.get('pattern_completion_display', 'Unknown'))}")
            lines.append(f"  Pattern Completion: {pattern.get('pattern_completion_display', 'Unknown')}")
            lines.append(f"  Detected at: {pattern.get('detected_at_display', 'Unknown')}")
            if pattern.get("confirmation_at_display"):
                lines.append(f"  Confirmation Time: {pattern['confirmation_at_display']}")
            _append_transition_lines(lines, pattern, indent="  ")
            lines.append(f"  Display Timezone: {pattern.get('display_timezone', 'Unknown')}")
            lines.append(f"  Signal Strength: {pattern.get('signal_strength', 'Unknown')}")
            lines.append(f"  Pattern Score Contribution: {pattern.get('pattern_score_contribution', 0.0)}")
            lines.append(f"  Volume Score Contribution: {pattern.get('volume_score_contribution', 0.0)}")
            lines.append(f"  Combined Event Contribution: {pattern.get('combined_event_contribution', pattern.get('current_weighted_score', 0.0))}")
            lines.append(f"  Recency Weight: {pattern.get('recency_weight', 'Unknown')}")
            lines.append(f"  Score Eligible: {'Yes' if pattern.get('score_eligible') else 'No'}")
            lines.append(f"  Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}")
            if pattern.get("invalidation_condition"):
                lines.append(f"  Invalidation Condition: {pattern['invalidation_condition']}")
            if pattern.get("current_score_exclusion_reason"):
                lines.append(f"  Current Score Exclusion Reason: {pattern['current_score_exclusion_reason']}")
            if pattern.get("overlap_note"):
                lines.append(f"  Overlap Note: {pattern['overlap_note']}")
            if pattern.get("related_note"):
                lines.append(f"  Relationship Note: {pattern['related_note']}")

    append_event_section("Current Contributing Evidence", current_contributing)
    append_event_section("Awaiting Confirmation", awaiting_confirmation)
    append_event_section("Conflicting Evidence", current_conflicting)
    append_event_section("Current Neutral / Informational Evidence", current_neutral)
    append_event_section("Recent Non-Contributing Tracked Events", recent_non_contributing)

    lines.append(f"Deprecated Current Relevant Patterns ({len(current_patterns)}):")
    if current_patterns:
        lines.append("  This alias is deprecated and mirrors the current display collections above.")
    else:
        lines.append("  None")

    if pattern_history_mode in {"session", "all"}:
        relevant_session = result.get("relevant_session", {})
        lines.append(
            f"Historical Session Detections ({shown_history} shown{history_count_suffix}):"
        )
        lines.append(
            f"  Relevant Session: {relevant_session.get('exchange_date', 'Unknown')} "
            f"({relevant_session.get('session_start_display', 'Unknown')} to "
            f"{relevant_session.get('session_end_display', 'Unknown')})"
        )
        lines.append(f"  Session Mode: {relevant_session.get('session_mode', result.get('session_mode', 'Unknown'))}")
        lines.append(
            f"  Included Segments: {', '.join(relevant_session.get('included_segments', result.get('included_segments', []))) or 'Unknown'}"
        )
        if total_history != shown_history:
            lines.append(f"  Showing {shown_history} of {total_history} detected events")
        if session_history:
            for index, pattern in enumerate(session_history, start=1):
                lines.append(f"  {index}. {pattern.get('primary_pattern_name', 'Unknown')}")
                lines.append(f"     Pattern Labels: {', '.join(pattern.get('pattern_labels', []))}")
                lines.append(f"     Detected at: {pattern.get('detected_at_display', 'Unknown')}")
                lines.append(f"     State: {pattern.get('state', 'unknown')}")
                _append_transition_lines(lines, pattern, indent="     ")
                lines.append(
                    f"     Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}"
                )
                if pattern.get("invalidation_condition"):
                    lines.append(f"     Invalidation Condition: {pattern['invalidation_condition']}")
                if pattern.get("current_score_exclusion_reason"):
                    lines.append(
                        f"     Current Score Exclusion Reason: {pattern['current_score_exclusion_reason']}"
                    )
                if pattern.get("overlap_note"):
                    lines.append(f"     Overlap Note: {pattern['overlap_note']}")
                if pattern.get("related_note"):
                    lines.append(f"     Relationship Note: {pattern['related_note']}")
        else:
            lines.append("  None")

    lines.append("Historical Lifecycle Summary:")
    if historical_summary:
        lines.append(f"  Total Historical Lifecycle Events: {historical_summary.get('count', len(historical_lifecycle))}")
        if historical_summary.get("by_state"):
            lines.append(
                "  By State: "
                + ", ".join(f"{state}={count}" for state, count in historical_summary["by_state"].items())
            )
        if historical_summary.get("by_family"):
            lines.append(
                "  By Family: "
                + ", ".join(f"{family}={count}" for family, count in historical_summary["by_family"].items())
            )
    else:
        lines.append("  None")

    if include_all_patterns:
        lines.append(f"All Historical Detected Pattern Labels ({len(raw_patterns)}):")
        if raw_patterns:
            for pattern in raw_patterns:
                lines.append(f"  Name: {pattern['pattern_name']}")
                lines.append(f"  Family: {pattern.get('pattern_family', 'unknown')}")
                lines.append(f"  Status: {pattern.get('status', 'confirmed')}")
                lines.append(f"  State: {pattern.get('event_state', 'unknown')}")
                lines.append(f"  Bias: {pattern['bias']}")
                lines.append(
                    f"  Pattern Start: {pattern.get('pattern_start_display', pattern.get('bar_start_display', 'Unknown'))}"
                )
                lines.append(
                    f"  Setup Completion: {pattern.get('setup_completion_display', pattern.get('pattern_end_display', 'Unknown'))}"
                )
                lines.append(
                    f"  Pattern Completion: {pattern.get('pattern_end_display', pattern.get('bar_end_display', 'Unknown'))}"
                )
                lines.append(f"  Detected at: {pattern['detected_at_display']}")
                if pattern.get("confirmation_at_display"):
                    lines.append(f"  Confirmation Time: {pattern['confirmation_at_display']}")
                lines.append(f"  Included in Current Score: {'Yes' if pattern.get('included_in_current_score') else 'No'}")
                if pattern.get("invalidation_condition"):
                    lines.append(f"  Invalidation Condition: {pattern['invalidation_condition']}")
                if pattern.get("exclusion_reason"):
                    lines.append(f"  Exclusion Reason: {pattern['exclusion_reason']}")

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
    if structured.get("supporting_trend_evidence"):
        lines.append("Supporting Trend Evidence:")
        for item in structured["supporting_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("conflicting_trend_evidence"):
        lines.append("Conflicting Trend Evidence:")
        for item in structured["conflicting_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("neutral_trend_evidence"):
        lines.append("Neutral Trend Evidence:")
        for item in structured["neutral_trend_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bullish_evidence"):
        lines.append("Bullish Evidence:")
        for item in structured["bullish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("bearish_evidence"):
        lines.append("Bearish Evidence:")
        for item in structured["bearish_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("current_pattern_evidence"):
        lines.append("Current Pattern Evidence:")
        for item in structured["current_pattern_evidence"]:
            lines.append(f"  - {item}")
    if structured.get("session_context"):
        lines.append("Session Context:")
        for item in structured["session_context"]:
            lines.append(f"  - {item}")
    if structured.get("lifecycle_note"):
        lines.append("Lifecycle Note:")
        lines.append(f"  {structured['lifecycle_note']}")
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
    if structured.get("confidence_breakdown"):
        lines.append("Confidence Breakdown:")
        for key, value in structured["confidence_breakdown"].items():
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)
